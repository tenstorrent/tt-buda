// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/move_index_to_mm_weights.hpp"

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "passes/passes_utils.hpp"
#include "passes/commute_utils.hpp"
#include "passes/consteval.hpp"
#include "utils/logger.hpp"
#include "reportify/reportify.hpp"

namespace tt::passes
{

bool is_type(graphlib::Node *n, std::string type)
{
    graphlib::OpNode *node = dynamic_cast<graphlib::OpNode *>(n);
    if (not node)
        return false;

    if (node->op_type().op != type)
        return false;
    return true;
}

// Check if all of passed user_nodes are index op
bool are_all_users_index(std::vector<graphlib::Node *> &users)
{
    int dim_common = -1, length_common = -1;
    for (graphlib::Node *n : users)
    {
       if (not (is_type(n, "index")))
           return false; 

       // Check attribute
       auto attrs = dynamic_cast<graphlib::OpNode *>(n)->op_attrs(); 
       int dim = std::get<int>(attrs[0]);
       int length = std::get<int>(attrs[2]) - std::get<int>(attrs[1]);
       if (length_common == -1)
       {
            length_common = length;
            dim_common = dim;
       }
       if (dim != dim_common or length != length_common)
           return false;
    }

    return true;
}

// Look for target sequence from given node
std::vector<graphlib::Node*> path_satisfying_condition(graphlib::Graph *graph, graphlib::Node *node)
{
    std::vector<graphlib::Node*> path;

    // Check users
    graphlib::Node *iter = node; 
    while (true)
    {
        std::vector<graphlib::Node *> users = graph->data_users(iter);

        // Output node does not satisfy condition
        if (users.empty())
            return {};

        // Cease the search once after hitting index ops
        if (is_type(iter, "reshape") and is_type(users[0], "index"))
        {
            if (not are_all_users_index(users))
            {
                path.clear();
            }
            else
            {
                path.push_back(iter); 
            }
            break;    
        }

        // Only accept ops with #users == 1, except index ops
        if (users.size() > 1)
            return {}; 
 
        // Check consumer chain is matmul -> reshape -> add -> reshape -> index
        if ((is_type(iter, "transpose") and is_type(users[0], "matmul")) or 
            (is_type(iter, "matmul") and is_type(users[0], "reshape")) or 
            (is_type(iter, "matmul") and is_type(users[0], "add")) or
            (is_type(iter, "reshape") and is_type(users[0], "add")) or
            (is_type(iter, "add") and is_type(users[0], "reshape")))
        {
            path.push_back(iter);
            iter = users[0];
        }
        else
        {
            path.clear();
            break;
        }
    }

    return path;
}

size_t get_non_param_port(graphlib::Graph *graph, graphlib::Node *node)
{
    auto operands = graph->data_operands(node);
    for (size_t i = 0; i < operands.size(); ++i)
    {
        graphlib::Node *n = operands[i];
        if (n->node_type() == graphlib::NodeType::kInput)
            continue;
        if (is_type(n, "select") and graph->data_operands(n)[0]->node_type() == graphlib::NodeType::kInput)
            continue;
        return i;
    }
    return 0;
}

graphlib::Node * duplicate_path(
    graphlib::Graph *graph, 
    std::vector<graphlib::Node *> &path,
    graphlib::Node *bias_node,
    graphlib::Node *weight_node,
    graphlib::OpNode *index_op
)
{
    graphlib::Node *prev_node = weight_node;
    int index = std::get<int>(index_op->op_attrs()[1]);

    // Iterate through nodes in the path
    for (size_t i = 0; i < path.size(); ++i)
    {
        graphlib::Node *org_node = path[i];
        const std::string name = org_node->name() + "_clone_path_" + std::to_string(index);
        auto *clone = graph->add_node(org_node->clone(name), graph->get_subgraph_id_for_node(org_node->id()));
        size_t port_id = (is_type(org_node, "matmul")) ? 1-get_non_param_port(graph, org_node) : get_non_param_port(graph, org_node);
        auto new_edge = graphlib::Edge(prev_node->id(), graph->data_users(prev_node).size(), clone->id(), port_id, graphlib::EdgeType::kData);
        graph->add_edge(new_edge);
        prev_node = clone;

        // For add op, we need to add edge between bias param and the node as well
        if (is_type(clone, "add"))
        {
            new_edge = graphlib::Edge(bias_node->id(), graph->data_users(bias_node).size(), clone->id(), 1-port_id, graphlib::EdgeType::kData); 
            graph->add_edge(new_edge);
            break;
        }
        else if (is_type(clone, "matmul"))
        {
            graphlib::Node * input_node = graph->data_operands(org_node)[0];
            if (is_type(input_node, "select"))
                input_node = graph->data_operands(org_node)[1];
            new_edge = graphlib::Edge(input_node->id(), graph->data_users(input_node).size(), clone->id(), 1-port_id, graphlib::EdgeType::kData); 
            graph->add_edge(new_edge);
        }
    }

    return prev_node; 
}

std::vector<graphlib::OpType::Attr> recalculate_index_attr(
    graphlib::Graph *graph, 
    graphlib::Node *next_node,
    std::vector<graphlib::OpType::Attr> &old_attr
)
{
    // retrieve attr values of original index op 
    int dim = std::get<int>(old_attr[0]); 
    int start = std::get<int>(old_attr[1]);
    int length = std::get<int>(old_attr[2]);
    int stride = std::get<int>(old_attr[3]);

    // calculate the corresponding slicing for the operand 
    auto operand_shape = next_node->shape();
    std::vector<graphlib::OpType::Attr> new_attr = {dim, start, length, stride};
    if (is_type(next_node, "transpose"))
    {
        auto transpose_op_type = next_node->as<graphlib::OpNode>()->op_type();
        int dim0 = transpose_op_type.get_attr_as<int>("dim0");
        if (dim0 >= 0)
            dim0 -= operand_shape.size();
        int dim1 = transpose_op_type.get_attr_as<int>("dim1");
        if (dim1 >= 0)
            dim1 -= operand_shape.size();
        if (dim0 == dim)
            new_attr = {dim1, start, length, stride};
        else if (dim1 == dim)
            new_attr = {dim0, start, length, stride};
    }
    else if (is_type(next_node, "reshape"))
    {
        auto reshape_in_shape = graph->data_operands(next_node)[0]->shape(); 

        // update attribute if dimension is changed by reshape
        if (operand_shape[dim] != reshape_in_shape[dim])
        {
            int new_start = start, new_length = length, new_stride = stride;
            for (int i = dim+1; i <= -1; ++i)
            {
                 new_length *= operand_shape[i]; 
            }
            new_start *= new_length;
            new_stride *= new_length;
            new_attr = {-1, new_start, new_length, new_stride};
        }
    }
    
    return new_attr; 
}

std::vector<graphlib::OpType::Attr> index2select(graphlib::Graph *graph, graphlib::OpNode *index_op)
{
    auto old_attr = index_op->op_attrs();
    int dim = std::get<int>(old_attr[0]); 
    int start = std::get<int>(old_attr[1]);
    int length = std::get<int>(old_attr[2]) - start;
    int stride = graph->data_operands(index_op)[0]->shape()[dim];
    std::vector<graphlib::OpType::Attr> new_attr = {dim, start, length, stride};
    return new_attr;
}

bool reached_target_node(
    graphlib::Graph * graph,
    graphlib::Node* n,
    graphlib::Node* target_node
)
{
    auto operands = graph->data_operands(n);

    // select ops are supposed to be inserted after weight/bias ops
    if (is_type(n, "transpose") or is_type(n, "matmul") or is_type(n, "add"))
    {
        for (auto op : operands)
        {
            if (op == target_node)
                return true;
            
            if (is_type(op, "select") and graph->data_operands(op)[0] == target_node)
                return true;
        }
    }
    return false; 
}

graphlib::Node* insert_index_op(
    graphlib::Graph *graph,
    graphlib::OpNode *index_op, 
    graphlib::Edge &user_edge,
    std::vector<graphlib::Node*> &path
)
{
    // clone new select op
    graphlib::Node *param_node = graph->node_by_id(user_edge.producer_node_id);
    auto name = index_op->name() + "_commute_clone_" + param_node->name(); 
    auto *index_clone = graph->add_node(index_op->clone(name), graph->get_subgraph_id_for_node(index_op->id())); 
    graphlib::OpNode *new_select_op = dynamic_cast<graphlib::OpNode *>(index_clone);

    // convert index attr to select attr
    std::vector<graphlib::OpType::Attr> new_attr = index2select(graph, index_op);

    // recalculate attribute of the select op
    for (auto itr = path.rbegin(); itr != path.rend(); ++itr)
    {
        graphlib::Node *n = (*itr);
        new_attr = recalculate_index_attr(graph, n, new_attr);
        if (reached_target_node(graph, n, param_node))
            break;
    }

    // add the node to graph
    new_select_op->change_op_type("select", new_attr); 
    insert_node_on_edge(graph, user_edge, index_clone);
    return new_select_op;
}

// return the first add node in path
graphlib::Node* get_bias_user_node(std::vector<graphlib::Node*> &path)
{
    for (auto n : path)
    {
       if (is_type(n, "add"))
           return n;
    }
    return nullptr;
}

// get weight/bias parameter node and the edge connected to it
std::tuple<graphlib::Node*, graphlib::Edge> get_param_operand_info(graphlib::Graph *graph, graphlib::Node* node)
{
    auto node_user_edges = graph->operand_edges(node);
    for (auto e : node_user_edges)
    {
        graphlib::Node *user = graph->node_by_id(e.producer_node_id);
        if (user->node_type() == graphlib::NodeType::kInput and user->as<graphlib::InputNode>()->input_type() == graphlib::InputNodeType::Parameter)
            return std::make_tuple(user, e);
    }
    return std::make_tuple(nullptr, graphlib::Edge());
}

// update attribute of reshape ops in path, except the last reshape
void update_reshape_attributes(graphlib::Graph *graph, graphlib::Node * weight_select, graphlib::Node *stop_node)
{
    TT_ASSERT(is_type(weight_select, "select"), "weight-select node with unexpected type");
    auto attrs = weight_select->as<graphlib::OpNode>()->op_attrs();
    graphlib::Node *iter = graph->data_users(weight_select)[0];
    int num_index_ops = graph->data_users(stop_node).size();

    while(iter != stop_node)
    {
        attrs = recalculate_index_attr(graph, iter, attrs);
        if (is_type(iter, "reshape"))
        {
             int current_dim = std::get<int>(attrs[0]);
             graphlib::Shape new_shape = iter->shape();
             new_shape[current_dim] /= num_index_ops;
             update_reshape_attr(iter->as<graphlib::OpNode>(), new_shape);
        }

        iter = graph->data_users(iter)[0];
    }
}

void commute(graphlib::Graph *graph, std::vector<graphlib::Node *> &path)
{ 
    TT_ASSERT(path.size() == 5 or path.size() == 3);
    graphlib::Node *weight_user_node = path.front();
    graphlib::Node *bias_user_node = get_bias_user_node(path);
    graphlib::OpNode *index_producer = path.back()->as<graphlib::OpNode>();
    std::vector<graphlib::Node *> index_ops = graph->data_users(index_producer);

    // retrieve weight/bias param nodes
    auto [weight_node, weight_user_edge] = get_param_operand_info(graph, weight_user_node);
    auto [bias_node, bias_user_edge] = get_param_operand_info(graph, bias_user_node);
    graphlib::Node *add_node = bias_user_node;

    // Duplicate matmul -> add path for (N-1) times, where N = index_ops.size()
    // and move the index-op to right after transpose and bias param as select op
    for (size_t i = 0; i < index_ops.size(); ++i)
    {
        graphlib::OpNode *index_op = dynamic_cast<graphlib::OpNode *>(index_ops[i]); 

        if (i == 0)
        { 
            // no need for path duplication for the 1st index op, just insert index ops and update reshape attributes
            auto new_select_op_weight = insert_index_op(graph, index_op, weight_user_edge, path);
            insert_index_op(graph, index_op, bias_user_edge, path);
            update_reshape_attributes(graph, new_select_op_weight, index_producer);
        }
        else // (i > 0)
        {
            // Duplicate path for 2nd ~ index ops  
            add_node = duplicate_path(graph, path, bias_node, weight_node, index_op);
            weight_user_edge = graph->user_data_edges(weight_node).back(); 
            bias_user_edge = graph->user_data_edges(bias_node).back();
            insert_index_op(graph, index_op, weight_user_edge, path);
            auto new_select_op_add = insert_index_op(graph, index_op, bias_user_edge, path);

            // copy bcast to new select node
            graphlib::Node *first_index_node = graph->data_operands(bias_user_node)[0];
            if (not is_type(first_index_node, "select"))
                first_index_node = graph->data_operands(bias_user_node)[1];
            auto edge_with_org_tm = graph->get_edges(bias_node, first_index_node)[0];
            auto edge_with_new_tm = graph->get_edges(bias_node, new_select_op_add)[0];
            graph->copy_edge_attributes(edge_with_org_tm, edge_with_new_tm);
        }
 
        // connect add and reshape 
        graphlib::replace_node(graph, index_op, add_node, true);
    }

    // Bypass the reshape right before index-ops as well 
    graph->remove_node(index_producer);
    recalculate_shapes(graph); 
}

// consteval
void merge_index_ops(graphlib::Graph *graph, graphlib::Node *input_node)
{
    // merge multiple index ops, if they are identical including
    std::vector<graphlib::Node *> input_users = graph->data_users(input_node);
    std::unordered_map<int, graphlib::Node*> index_attr_to_node;
    for (auto *index_op : input_users)
    {
        if (not is_type(index_op, "select"))
            return;

        // check if there's index-op with same attr
        auto attrs = dynamic_cast<graphlib::OpNode *>(index_op)->op_attrs();
        int start_index = std::get<int>(attrs[1]);
        auto it = index_attr_to_node.find(start_index);
        if (it != index_attr_to_node.end())
            graphlib::replace_node(graph, index_op, it->second, true);
        else
            index_attr_to_node.insert(std::make_pair(start_index, index_op));
    }
}

void move_index_to_mm_weights(graphlib::Graph *graph) 
{
    // This pass aims to move index-ops up to be right after weight op, which consists of:
    // (1) detects index ops that have the same weight node in their producer chains,
    // (2) move those index ops to be immediate consumer op the weight op
    // (3) fracture the weight op to multiple weight ops by constevaling
    //  - NOTE: considering past-cache case, index ops that have the same attribute values
    //          will have the same fractured weight node as their poroducer node

    // Explore graph to find input nodes including parameters
    std::vector<graphlib::Node *> input_nodes;
    for (auto *node : graphlib::topological_sort(*graph))
    {
        graphlib::InputNode *input_node = dynamic_cast<graphlib::InputNode *>(node);
        if (not input_node)
            continue;
        input_nodes.push_back(node);
    }

    // Search through input nodes, if the consumers shape symmetry tree until index nodes
    // the input node is marked as the node that we can move those index nodes to  
    bool commuted = false;
    for (auto *input_node : input_nodes)
    { 
        // Check if the input-node satisfies the condition
        std::vector<graphlib::Node *> input_users = graph->data_users(input_node);
        std::vector<std::vector<graphlib::Node *>> paths; 
        for (graphlib::Node * n : input_users)
        {
            auto path = path_satisfying_condition(graph, n);
            if (path.empty())
                break;
            paths.push_back(path);
        }  

        if (paths.empty())
             continue;

        // Commute
        for (auto &path : paths)
        {
            commute(graph, path);
        }

        // Merge equivalent index ops
        merge_index_ops(graph, input_node);
        commuted = true;
    }
 
    // Consteval
    if (commuted)
        run_consteval_graph_pass(graph);
    reportify::dump_graph(graph->name(), "post_move_index", graph);
}

}  // namespace tt::passes
