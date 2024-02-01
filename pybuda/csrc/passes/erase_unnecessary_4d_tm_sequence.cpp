// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{

static bool is_reshape(graphlib::Node const *node)
{
    graphlib::OpNode const *op = dynamic_cast<graphlib::OpNode const *>(node);
    return op and op->op_name() == "reshape";
}

static bool is_transpose(graphlib::Node const *node)
{
    graphlib::OpNode const *op = dynamic_cast<graphlib::OpNode const *>(node);
    return op and op->op_name() == "transpose";
}

static bool involves_w_dim(graphlib::OpNode const * op) 
{
    auto shape = op->shape().as_vector();
    if (op->op_name() == "reshape")
    {
        int w_dim = shape.size() - 4;
        return shape.size() >= 4 and shape[w_dim] > 1;
    } else if (op->op_name() == "transpose") {
        int _dim0 = op->op_type().get_attr_as<int>("dim0");
        if (_dim0 > 0)
            _dim0 -= shape.size();
        int _dim1 = op->op_type().get_attr_as<int>("dim1");
        if (_dim1 > 0)
            _dim1 -= shape.size();
        if (_dim0 > _dim1)
            std::swap(_dim0, _dim1); // _dim1 > _dim0
        return _dim0 == -4 and _dim1 == -3;
    }

    return false;
}

static bool output_shape_matches(graphlib::OpNode const * op, graphlib::Shape const & first_input_shape) 
{
    if (op->op_name() != "reshape")
        return false;

    auto input_shape_v = first_input_shape.as_vector();
    auto output_shape = op->shape().as_vector();
    if (input_shape_v.size() != output_shape.size())
        return false;
    return std::equal(input_shape_v.begin(), input_shape_v.end(), output_shape.begin());
}

static std::vector<graphlib::Node *> path_of_transpose_and_reshape(graphlib::Graph *graph, graphlib::Node *initial_node)
{
    std::vector<graphlib::Node *> path;

    // set the input shape of the first reshape
    graphlib::OpNode const *init_op = dynamic_cast<graphlib::OpNode const *>(initial_node);
    graphlib::Node *reshape_input = graph->data_operands(init_op)[0];
    graphlib::Shape first_input_shape = reshape_input->shape(); 

    // move ptr to reshape op 
    path.push_back(dynamic_cast<graphlib::OpNode *>(initial_node));
    graphlib::Node *iter = initial_node; 
    while (true)
    {

        if (path.size() >= 3) {
            path.clear();
            break;
        }

        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(iter);
        TT_ASSERT(op);

        std::vector<graphlib::Node *> users = graph->data_users(op);
        if (users.size() > 1) {
            path.clear();
            break;
        }
        graphlib::OpNode *user = dynamic_cast<graphlib::OpNode *>(users[0]);
        if (not user) {
            path.clear();
            break;
        }
        
        if (is_reshape(user) and output_shape_matches(user, first_input_shape)) {
            path.push_back(user);
            break; 
        }
        else if (is_transpose(user) and involves_w_dim(user)) {
            path.push_back(user);
        }
        else {
            path.clear();
            break;
        }
        iter = user;
    }

    if (path.size() < 3)
    {
	path.clear();
    }

    return path;
}


static void commute_4d_tm_ops(graphlib::Graph *graph, std::vector<graphlib::Node *> path)
{
    graphlib::OpNode *first = path.front()->as<graphlib::OpNode>();

    TT_ASSERT(is_reshape(first));
    TT_ASSERT(path.size() == 3);
    auto first_reshape_out_shape = first->shape().as_vector();
    int fourth_dim = first_reshape_out_shape[0];
    int third_dim = first_reshape_out_shape[1];
    int orig_third_dim = third_dim * fourth_dim;   
 
    // path is supposed to be (random op) --> reshape(1) -> transpose -> reshape(2)
    //  - create multiple select nodes and interleave op
    //  - connect input/output of the select ops
    //  - connect output of the interleave op with output of reshape(2)
    //  - remove old nodes

    // create select ops
    std::vector<graphlib::PyOpNode *> new_select_nodes;
    for (int i = 0; i < fourth_dim; i++) {
        graphlib::OpType op_type("select", {-3, i*third_dim, third_dim, orig_third_dim});
        std::string op_name = first->name() + "_replaced_select.";
        op_name += std::to_string(i);
        graphlib::PyOpNode *new_node = graph->add_node(
            graphlib::create_node<graphlib::PyOpNode>(op_name, op_type), graph->get_subgraph_id_for_node(first->id())); 
        new_node->set_shape(graphlib::Shape::create({1, unsigned(third_dim), first_reshape_out_shape[2], first_reshape_out_shape[3]}));
        new_node->set_output_df(first->output_df());
        new_select_nodes.push_back(new_node);
    } 

    // create interleave op
    graphlib::OpType op_type("interleave", {-3, 1});
    std::string op_name = first->name() + "_replaced_interleave.0";
    graphlib::PyOpNode *new_interleave_node = graph->add_node(
        graphlib::create_node<graphlib::PyOpNode>(op_name, op_type), graph->get_subgraph_id_for_node(first->id()));  
    new_interleave_node->set_shape(graphlib::Shape::create({1, unsigned(orig_third_dim), first_reshape_out_shape[2], first_reshape_out_shape[3]}));
    new_interleave_node->set_output_df(first->output_df());

    // connect inputs of the select ops and remove old output edge and outputs to interleave op 
    graphlib::Edge input_edge = graph->operand_data_edges(first)[0];
    for (int idx = 0; idx < fourth_dim; ++idx) {
        auto select_node = new_select_nodes[idx]; 
        auto new_edge0 = graphlib::Edge(
            input_edge.producer_node_id,
            0,
            select_node->id(),
            0,
            input_edge.edge_type); 
        graph->add_edge(new_edge0); 
        graph->copy_edge_attributes(input_edge, new_edge0);
    	
        auto new_edge1 = graphlib::Edge(
            select_node->id(),
            0,
            new_interleave_node->id(),
            idx,
            input_edge.edge_type);      
        graph->add_edge(new_edge1); 
        graph->copy_edge_attributes(input_edge, new_edge1);
    }

    // connect output of the reshape(2) with output of interleave
    graphlib::OpNode *last = path.back()->as<graphlib::OpNode>();
    for (graphlib::Edge &user : graph->user_edges(last)) {
        if (user.edge_type == graphlib::EdgeType::kData) {
            auto new_edge = graphlib::Edge(
                    new_interleave_node->id(),
                    (graphlib::PortId)0,
                    user.consumer_node_id,
                    user.consumer_input_port_id,
                    user.edge_type);
            graph->add_edge(new_edge);
            graph->copy_edge_attributes(user, new_edge);
        }
    } 

    graph->remove_node(path[1]);
    graph->remove_node(path[2]);
}

// It's supposed to detect:
// (op, output shape A) --> reshape() --> transpose(-4,-3,-1) --> reshape(shape A)
// and replace it with:
// (op, output shape A) --> select ops(dim=-3) --> interleave(dim=-3) 
static bool are_removable_4d_tm_sequence(graphlib::Graph *graph, graphlib::Edge edge)
{
    graphlib::Node *producer = graph->node_by_id(edge.producer_node_id); 
    bool doesnt_fork = graph->data_operands(producer).size() == 1; 
    bool commuted = false;
    if (doesnt_fork) {
       if (is_reshape(producer)) { 
           std::vector<graphlib::Node *> removing_path = path_of_transpose_and_reshape(graph, producer);  
           if (!removing_path.empty()) {
                commute_4d_tm_ops(graph, removing_path);
                commuted = true;
           }
       }
    }
    return commuted;
}

void erase_unnecessary_4d_tm_sequence(graphlib::Graph *graph)
{
    bool updated = true;
    while (updated)
    {
        updated = false;
        for (auto *node : graphlib::topological_sort(*graph))
        {
            auto user_edges = graph->user_data_edges(node);
            if (user_edges.size() > 1 or user_edges.empty())
                continue; 

            bool has_bcast = graph->get_edge_attributes(user_edges[0])->has_broadcast_dims();
            if (not has_bcast and are_removable_4d_tm_sequence(graph, user_edges[0]))
            {
                log_debug(LogGraphCompiler, "Bypass reshape: {}", node->name());
                auto change_rank = [graph](graphlib::Edge new_edge, graphlib::Edge)
                {
                    if (not is_reshape(graph->node_by_id(new_edge.consumer_node_id)))
                        handle_change_rank(graph, new_edge);
                };
                bypass_node(graph, node, true, change_rank);
                updated = true;

                break;
            }
        }
    }
}

}  // namespace tt::passes
