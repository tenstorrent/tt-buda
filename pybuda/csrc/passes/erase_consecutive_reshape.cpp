// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/erase_consecutive_reshape.hpp"

#include <pybind11/pybind11.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"
#include "passes/commute_utils.hpp"
#include "passes/passes_utils.hpp"
namespace tt::passes
{

static bool is_reshape(graphlib::Node const *node)
{
    graphlib::OpNode const *op = dynamic_cast<graphlib::OpNode const *>(node);
    return op and op->op_name() == "reshape";
}

static bool is_narrow(graphlib::Node const *node)
{
    graphlib::OpNode const *op = dynamic_cast<graphlib::OpNode const *>(node);
    return op and op->op_name() == "narrow";
}

// A transpose with all dims but one dim equal to 1 is equavalent to a reshape
// Or if transpose dims are consecutive and one of the dims is 1, then its equivalent to a reshape
static bool is_reshape_transpose(graphlib::Node const *node)
{
    graphlib::OpNode const *op = dynamic_cast<graphlib::OpNode const *>(node);
    if (not op or op->op_name() != "transpose")
        return false;

    auto shape = op->shape();
    int _dim0 = op->op_type().get_attr_as<int>("dim0");
    int _dim1 = op->op_type().get_attr_as<int>("dim1");
    if (_dim0 > _dim1)
        std::swap(_dim0, _dim1);

    if (_dim0 + 1 == _dim1) {
        // Consecutive transpose dims
        return shape[_dim0] == 1 or shape[_dim1] == 1;
    } else {
        auto shape_vector = shape.as_vector();
        std::uint32_t max_dim = 0;
        std::uint32_t volume = 1;
        for (auto dim : shape_vector)
        {
            max_dim = std::max(max_dim, dim);
            volume *= dim;
        }
        return max_dim == volume;
    }
}

static std::vector<graphlib::Node *> path_to_reshape_after_communable_unaries(graphlib::Graph *graph, graphlib::Node *initial_node)
{
    std::vector<graphlib::Node *> path;
    
    path.push_back(dynamic_cast<graphlib::OpNode *>(initial_node));
    graphlib::Node *iter = initial_node;
    while (true)
    {
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
        if (graphlib::is_eltwise_unary(user) or graphlib::is_eltwise_binary(user)) {
            path.push_back(op);
        }
        else if (is_reshape(user)) {
            path.push_back(op);
            path.push_back(user);
            break;
        }
        else {
            path.clear();
            break;
        }
        iter = user;
    }

    return path;
}

static void commute_eltwise_ops(graphlib::Graph *graph, std::vector<graphlib::Node *> path)
{
    graphlib::OpNode *first = path.front()->as<graphlib::OpNode>();

    TT_ASSERT(is_reshape(first) or is_reshape_transpose(first));
    graphlib::Node *reshape_input = graph->data_operands(first)[0];
    graphlib::Shape commute_shape = reshape_input->shape();
    graphlib::OpNode *last = path.back()->as<graphlib::OpNode>();

    // path is reshape -> eltwise -> eltwise ... -> reshape
    // don't need to alter first reshape as it's getting removed
    for (std::size_t i = 0; i < path.size()-1; ++i) {
        graphlib::Node *node = path[i];
        node->set_shape(commute_shape);

        // Set the shape to the desired final shape for this whole path
        log_trace(LogGraphCompiler, "Commuting: {} to: {}", node->name(), commute_shape);
        if (graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node))
        {
            op->add_golden_transform(first->op_type());

            // Handle the other operand if it's eltwise-binary op  (taken from erase-inverse-ops)
            if (graphlib::is_eltwise_binary(op))
            {
		std::vector<graphlib::Edge> edges = graph->operand_data_edges(op);
                graphlib::Edge current_edge = (edges[0].producer_node_id == path[i-1]->id()) ? edges[0] : edges[1];
                auto current_edge_tms = graph->get_edge_attributes(current_edge)->get_tms();

                for (graphlib::OpType &op_type : current_edge_tms) {
                    if (op_type.op == "broadcast")
                    {
                        int bcast_dim = std::get<int>(op_type.attr[0]);
                        int volume = std::get<int>(op_type.attr[1]);
                        if (bcast_dim < 0)
                            bcast_dim += commute_shape.size();

                        commute_shape[bcast_dim] *= volume;
                    }
                }

                graphlib::Edge another_operand_edge = (edges[0].producer_node_id == path[i-1]->id()) ? edges[1] : edges[0]; // operand not in the current path
           
                auto name = last->name() + "_operand_commute_clone" + std::to_string(another_operand_edge.edge_creation_id);
                graphlib::Node *clone = graph->add_node(last->clone(name), graph->get_subgraph_id_for_node(last->id()));
                graphlib::OpNode *added_op = dynamic_cast<graphlib::OpNode *>(clone);
                added_op->as<graphlib::TaggedNode>()->tag("dont_erase", true);
                log_trace(LogGraphCompiler, "  Operand commute clone: {} -> between {} and {} ", name, added_op->name(), graph->node_by_id(another_operand_edge.producer_node_id)->name());
            
                update_reshape_attr(added_op, commute_shape);
                clone->set_shape(commute_shape);
                log_trace(LogGraphCompiler, "  Operand commute clone shape: {}", commute_shape);
           
                // TODO: fix the bug that bc disapeears after lower-reinterpret-cast 
                insert_node_on_edge(graph, another_operand_edge, clone);
                handle_change_rank(graph, clone);
                try_commute_bcast_through_clone(graph, added_op);
                if (graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(graph->data_operands(clone)[0]))
                    try_consteval_input_no_operand_forks(graph, input, true);   
            } 
        } 
    }
}

static bool are_consecutive_reshape(graphlib::Graph *graph, graphlib::Edge edge, bool commute_eltwise)
{
    graphlib::Node *producer = graph->node_by_id(edge.producer_node_id);
    graphlib::Node *consumer = graph->node_by_id(edge.consumer_node_id);
    bool doesnt_fork = graph->data_operands(producer).size() == 1;
    bool consecutive_reshape = false;
    if (doesnt_fork) {
        if (!commute_eltwise) {
            consecutive_reshape = (is_reshape(producer) or is_reshape_transpose(producer)) and is_reshape(consumer);
        }
        else {
            if (is_reshape(producer) or is_reshape_transpose(producer)) {
                std::vector<graphlib::Node *> path_to_reshape = path_to_reshape_after_communable_unaries(graph, producer);
                if (!path_to_reshape.empty()) {
                    commute_eltwise_ops(graph, path_to_reshape);
                    consecutive_reshape = true;
                }
            }
        }
    }
    return consecutive_reshape;
}

static bool is_nop_reshape(graphlib::Graph *graph, graphlib::Node *node)
{
    if (not is_reshape(node))
        return false;

    graphlib::Node *operand = graph->data_operands(node)[0];
    return (operand->node_type() != graphlib::NodeType::kInput) and operand->shape() == node->shape();
}

static bool is_nop_narrow(graphlib::Graph *graph, graphlib::Node *node)
{
    if (not is_narrow(node))
        return false;

    graphlib::Node *operand = graph->data_operands(node)[0];
    return (operand->node_type() != graphlib::NodeType::kInput) and operand->shape() == node->shape();
}

bool erase_consecutive_reshape(graphlib::Graph *graph, bool commute_eltwise)
{
    bool updated = true;
    bool updated_anything = false;
    while (updated)
    {
        updated = false;
        for (auto *node : graphlib::topological_sort(*graph))
        {
            auto user_edges = graph->user_data_edges(node);
            if (user_edges.size() > 1 or user_edges.empty())
                continue;

            if (node->as<graphlib::TaggedNode>()->tag_value_or("dont_erase", false))
                continue;

            // TODO: relax this, but it causes a lot of edges cases
            bool has_bcast = graph->get_edge_attributes(user_edges[0])->has_broadcast_dims();
            if (not has_bcast and (are_consecutive_reshape(graph, user_edges[0], commute_eltwise) or is_nop_reshape(graph, node)))
            {
                log_trace(LogGraphCompiler, "Bypass reshape: {}", node->name());
                auto change_rank = [graph](graphlib::Edge new_edge, graphlib::Edge old_edge)
                {
                    auto original_tms = graph->get_edge_attributes(old_edge)->get_tms();
                    auto new_tms = graph->get_edge_attributes(new_edge)->get_tms();
                    new_tms.insert(new_tms.end(), original_tms.begin(), original_tms.end());
                    graph->get_edge_attributes(new_edge)->set_tms(new_tms);
                    if (not is_reshape(graph->node_by_id(new_edge.consumer_node_id)))
                        handle_change_rank(graph, new_edge);
                };
                bypass_node(graph, node, true, change_rank);
                updated = true;
                updated_anything = true;
                break;
            }
        }
    }
    return updated_anything;
}

void bypass_nop_tms(graphlib::Graph *graph)
{
    bool updated = true;
    while (updated)
    {
        updated = false;
        for (auto *node : graphlib::topological_sort(*graph))
        {
            auto user_edges = graph->user_data_edges(node);
            if (user_edges.empty())
                continue;

            if (is_nop_reshape(graph, node))
            {
                log_trace(LogGraphCompiler, "Bypass NOP reshape: {}", node->name());
                auto change_rank = [graph](graphlib::Edge new_edge, graphlib::Edge)
                {
                    if (not is_reshape(graph->node_by_id(new_edge.consumer_node_id)))
                        handle_change_rank(graph, new_edge);
                };
                bypass_node(graph, node, true, change_rank);
                updated = true;
                break;
            } else if (is_nop_narrow(graph, node))
            {
                log_trace(LogGraphCompiler, "Bypass NOP narrow: {}", node->name());
                bypass_node(graph, node, true);
                updated = true;
                break;
            }
        }
    }
}
}  // namespace tt::passes
