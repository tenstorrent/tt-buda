// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/replace_incommutable_patterns.hpp"

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"
#include "passes/passes_utils.hpp"
#include "passes/commute_utils.hpp"
#include "passes/print_graph.hpp"
#include "reportify/reportify.hpp"

namespace tt::passes
{

// This function will hoist a bcast through a path of ops given that
// the final node in the path has one user edge which contains the bcast on dim.
//
// This function will hoist the bcast up until but not including the first op
static bool hoist_bcast_through_path(
    graphlib::Graph *graph,
    std::vector<graphlib::OpNode*> path,
    int bcast_dim)
{
    if (path.size() < 2)
        return true;

    auto* last = path.back();
    
    TT_ASSERT(graph->data_users(last).size() == 1, "Must have one user edge");
    auto last_user_edge = graph->user_data_edges(last)[0];

    int bcast_volume = -1;
    for (auto &op_type : graph->get_edge_attributes(last_user_edge)->get_tms()) {
        if (op_type.op == "broadcast") {
            int dim = std::get<int>(op_type.attr[0]);
            if (dim == bcast_dim) {
                bcast_volume = std::get<int>(op_type.attr[1]);

                // Just incase, remove both
                graph->get_edge_attributes(last_user_edge)->remove_broadcast_dim(bcast_dim);
                graph->get_edge_attributes(last_user_edge)->remove_broadcast_dim(bcast_dim);
                break;
            }
        }
    }
    TT_ASSERT(bcast_volume != -1, "Broadcast on specified dim does not exist on final node user edge.");

    for (auto i = path.size()-1; i > 0; i--) {
        auto* op = path[i];
        TT_ASSERT(graph->data_users(op).size() == 1, "Must have one user edge");
        TT_ASSERT(is_elementwise(op), "Must be elementwise op");

        auto shape = op->shape();
        TT_ASSERT(shape[bcast_dim] == 1, "Cannot broadcast on dims > 1");
        shape[bcast_dim] = bcast_volume;

        op->set_shape(shape);

        // Place broadcast on operand forks, except for the previous in the path
        for (auto operand_edge : graph->operand_data_edges(op)) {
            if (graph->node_by_id(operand_edge.producer_node_id) == path[i-1])
                continue;
            graph->get_edge_attributes(operand_edge)->set_broadcast_dim(bcast_dim, bcast_volume, false);
        }
    }

    // Place bcase of first user edge
    auto* first = path[0];
    TT_ASSERT(graph->data_users(first).size() == 1, "Must have one user edge");
    auto first_user_edge = graph->user_data_edges(first)[0];
    graph->get_edge_attributes(first_user_edge)->set_broadcast_dim(bcast_dim, bcast_volume, false);
    return true;
}

static bool is_incommutable_reduce_avg_reduce_avg_bcast_on_incommutable_dim(
    graphlib::Graph *graph,
    graphlib::OpNode *op,
    graphlib::Shape commute_shape,
    graphlib::Shape clone_shape) 
{
    bool is_incommutable_reduce_avg_reduce_avg_bcast_on_incommutable_dim = true;
    is_incommutable_reduce_avg_reduce_avg_bcast_on_incommutable_dim &= op->op_name() == "reduce_avg";
    if (not is_incommutable_reduce_avg_reduce_avg_bcast_on_incommutable_dim)
        return false;
    int reduce_dim = std::get<int>(op->op_attrs()[0]);

    is_incommutable_reduce_avg_reduce_avg_bcast_on_incommutable_dim &= reduce_dim == -2;
    is_incommutable_reduce_avg_reduce_avg_bcast_on_incommutable_dim &= commute_shape[reduce_dim] == clone_shape[reduce_dim] * clone_shape[reduce_dim-1];
    if (not is_incommutable_reduce_avg_reduce_avg_bcast_on_incommutable_dim)
        return false;

    auto* next_op = dynamic_cast<graphlib::OpNode*>(graph->data_users(op)[0]);
    is_incommutable_reduce_avg_reduce_avg_bcast_on_incommutable_dim &= next_op and next_op->op_name() == "reduce_avg" and graph->data_users(op).size() == 1;
    if (next_op) {
        std::vector<graphlib::OpNode*> next_ops;
        next_ops.push_back(next_op);
        bool contains_y_bcast = false;
        int next_reduce_dim = std::get<int>(next_op->op_attrs()[0]);

        // Since this patten specifically picks up a reduce_avg on y dim, we are now looking for the inverse broadcast on the y dim
        while (not contains_y_bcast) {
            is_incommutable_reduce_avg_reduce_avg_bcast_on_incommutable_dim &= next_reduce_dim > reduce_dim;
            if (not is_incommutable_reduce_avg_reduce_avg_bcast_on_incommutable_dim)
                return false;
        
        
            auto next_next_edge = graph->user_data_edges(next_op)[0];

            for (auto &op_type : graph->get_edge_attributes(next_next_edge)->get_tms()) {
                if (op_type.op == "broadcast") {
                    int bcast_dim = std::get<int>(op_type.attr[0]);
                    contains_y_bcast |= bcast_dim == reduce_dim and std::get<int>(op_type.attr[1]) == (int)clone_shape[reduce_dim];
                }
            }
            if (contains_y_bcast) {
                is_incommutable_reduce_avg_reduce_avg_bcast_on_incommutable_dim = true;
                break;
            }
            if (graph->data_users(next_op).size() != 1) {
                is_incommutable_reduce_avg_reduce_avg_bcast_on_incommutable_dim = false;
                break;
            }
            next_op = dynamic_cast<graphlib::OpNode*>(graph->data_users(next_op)[0]);
            if (not next_op or not is_elementwise(next_op))
                break;
            next_ops.push_back(next_op);
        }
        if (is_incommutable_reduce_avg_reduce_avg_bcast_on_incommutable_dim) {
            hoist_bcast_through_path(graph, next_ops, reduce_dim);
        }
    }
    return is_incommutable_reduce_avg_reduce_avg_bcast_on_incommutable_dim;
}

static bool is_y_dim_concat_with_changed_x_dim(
    graphlib::Graph *graph,
    graphlib::OpNode *op,
    graphlib::Shape commute_shape) 
{
    if (op->op_name() != "concatenate")
        return false;

    int concat_dim = std::get<int>(op->op_attrs()[0]);

    if (concat_dim != -2)
        return false;

    if (commute_shape[-1] == op->shape()[-1])
        return false;

    // For now, lets make sure that all operands of the concat are a reshape
    // such that every dim of the operands of the reshapes are equivalent except for -2
    for (auto operand : graph->data_operands(op)) {
        graphlib::OpNode *operand_op = dynamic_cast<graphlib::OpNode*>(operand);
        if (not operand_op or operand_op->op_name() != "reshape")
            return false;

        auto operand_operand_shape = graph->data_operands(operand_op)[0]->shape();
        for (uint32_t i = 0; i < operand_operand_shape.size(); i++) {
            if (i == operand_operand_shape.size()-2)
                continue;
            if (operand_operand_shape[i] != commute_shape[i])
                return false;
        }
    }

    return true;
}

static bool attempt_replace_downward_pattern(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    graphlib::OpNode *op,
    graphlib::Shape commute_shape,
    graphlib::Shape clone_shape)
{
    if (is_incommutable_reduce_avg_reduce_avg_bcast_on_incommutable_dim(graph, op, commute_shape, clone_shape)) {
        log_trace(LogGraphCompiler, "  Replacing incommutable reduce: {} with grouped reduce ", op->name());

        // Incoming reshape
        graphlib::Edge incoming_edge = graph->operand_data_edges(op)[0];
        auto name = initial_op->name() + "_pattern_replacement_input_commute_clone" + std::to_string(incoming_edge.edge_creation_id);
        auto *incoming_clone = graph->add_node(initial_op->clone(name), graph->get_subgraph_id_for_node(initial_op->id()));
        graphlib::OpNode *incoming_clone_op = dynamic_cast<graphlib::OpNode *>(incoming_clone);
        update_reshape_attr(incoming_clone_op, commute_shape);
        incoming_clone->set_shape(commute_shape);
        auto [edge_in, edge_out] = insert_node_on_edge(graph, incoming_edge, incoming_clone);

        // Set output df to match producer
        incoming_clone->set_output_df(graph->node_by_id(edge_in.producer_node_id)->output_df());
        // Grouped reduce
        std::vector<graphlib::Edge> outgoing_edges = graph->user_data_edges(op);
        name = op->name() + "_grouped_reduce_avg_clone_" + std::to_string(incoming_edge.edge_creation_id);
        for (graphlib::Edge outgoing_edge : outgoing_edges) {
            name += "_" + std::to_string(outgoing_edge.edge_creation_id);
        }
        int reduce_dim = std::get<int>(op->op_attrs()[0]);

        std::vector<graphlib::OpType::Attr> grouped_reduce_attrs{reduce_dim, (int)clone_shape[reduce_dim-1], true};
        op->change_op_type("grouped_reduce_avg");
        op->overwrite_op_attrs(grouped_reduce_attrs);
        auto grouped_reduce_shape = commute_shape;
        op->set_shape(grouped_reduce_shape);

        // Update next reduce shape
        auto next_op = dynamic_cast<graphlib::OpNode *>(graph->data_users(op)[0]);
        auto next_reduce_shape = commute_shape;
        auto next_reduce_dim = std::get<int>(next_op->op_attrs()[0]);

        next_reduce_shape[next_reduce_dim] = next_op->shape()[next_reduce_dim];
        next_op->set_shape(next_reduce_shape);

        // Remove broadcast on next op user edge
        for (auto next_next_edge : graph->user_data_edges(next_op))
        {    auto tms = graph->get_edge_attributes(next_next_edge)->get_tms();
            graph->get_edge_attributes(next_next_edge)->clear_broadcast_dims();

            for (auto &op_type : tms) {
                if (op_type.op == "broadcast") {
                    int bcast_dim = std::get<int>(op_type.attr[0]);
                    int volume = std::get<int>(op_type.attr[1]);
                    if (bcast_dim == reduce_dim) {
                        continue;
                    }
                    graph->get_edge_attributes(next_next_edge)->set_broadcast_dim(bcast_dim, volume, false);
                }
            }
        }
        
        // Outgoing reshape(s)
        std::vector<graphlib::Edge> next_op_outgoing_edges = graph->user_data_edges(next_op);
        for (graphlib::Edge outgoing_edge : next_op_outgoing_edges) {
            name = initial_op->name() + "_pattern_replacement_output_commute_clone" + std::to_string(outgoing_edge.edge_creation_id);
            auto *outgoing_clone = graph->add_node(initial_op->clone(name), graph->get_subgraph_id_for_node(initial_op->id()));
            graphlib::OpNode *outgoing_clone_op = dynamic_cast<graphlib::OpNode *>(outgoing_clone);
            auto outgoing_clone_shape = clone_shape;
            outgoing_clone_shape[next_reduce_dim] = next_op->shape()[next_reduce_dim];
            update_reshape_attr(outgoing_clone_op, outgoing_clone_shape);
            outgoing_clone->set_shape(outgoing_clone_shape);
            auto [edge_in, edge_out] = insert_node_on_edge(graph, outgoing_edge, outgoing_clone, true, true, 0, true);
            // Set output df to match producer
            outgoing_clone->set_output_df(graph->node_by_id(edge_in.producer_node_id)->output_df());
        }
    }
    else if (is_y_dim_concat_with_changed_x_dim(graph, op, commute_shape)) {

        // Place inverse reshapes on all operands
        auto operand_edges = graph->operand_data_edges(op);
        for (graphlib::Edge incoming_edge: operand_edges) {
            auto name = initial_op->name() + "_pattern_replacement_input_commute_clone" + std::to_string(incoming_edge.edge_creation_id);
            auto *incoming_clone = graph->add_node(initial_op->clone(name), graph->get_subgraph_id_for_node(initial_op->id()));
            graphlib::OpNode *incoming_clone_op = dynamic_cast<graphlib::OpNode *>(incoming_clone);
            auto incoming_clone_shape = commute_shape;
            int incoming_concat_dim_len = graph->node_by_id(incoming_edge.producer_node_id)->shape()[-2];
            incoming_clone_shape[-2] = incoming_concat_dim_len * clone_shape[-1] / commute_shape[-1];
            update_reshape_attr(incoming_clone_op, incoming_clone_shape);
            incoming_clone->set_shape(incoming_clone_shape);
            auto [edge_in, edge_out] = insert_node_on_edge(graph, incoming_edge, incoming_clone);
            // Set output df to match producer
            incoming_clone->set_output_df(graph->node_by_id(edge_in.producer_node_id)->output_df());
        
        }

        // Retrieve current op shape for output clones
        auto output_clone_shape = op->shape();

        // Convert op shape
        auto new_concat_shape = op->shape();
        new_concat_shape[-2] = op->shape()[-2]*op->shape()[-1] / commute_shape[-1];
        new_concat_shape[-1] = commute_shape[-1];
        op->set_shape(new_concat_shape);

        // Add golden transform
        std::vector<uint32_t> shape_vec = output_clone_shape.as_vector();
        std::vector<graphlib::OpType::Attr> golden_transform_attrs;
        for (uint32_t d : shape_vec)
        {
            golden_transform_attrs.push_back((int)d);
        }
        op->add_golden_transform(graphlib::OpType("reshape", golden_transform_attrs));

        for (graphlib::Edge outgoing_edge : graph->user_data_edges(op)) {
            auto name = initial_op->name() + "_pattern_replacement_output_commute_clone" + std::to_string(outgoing_edge.edge_creation_id);
            auto *outgoing_clone = graph->add_node(initial_op->clone(name), graph->get_subgraph_id_for_node(initial_op->id()));
            graphlib::OpNode *outgoing_clone_op = dynamic_cast<graphlib::OpNode *>(outgoing_clone);
            auto outgoing_clone_shape = output_clone_shape;
            update_reshape_attr(outgoing_clone_op, outgoing_clone_shape);
            outgoing_clone->set_shape(outgoing_clone_shape);
            auto [edge_in, edge_out] = insert_node_on_edge(graph, outgoing_edge, outgoing_clone, true, true, 0, true);
            // Set output df to match producer
            outgoing_clone->set_output_df(graph->node_by_id(edge_in.producer_node_id)->output_df());
        }
    }
    else
        return false; // If we did not change a pattern then return false
    return true;
}

static bool attempt_replace_upward_pattern(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    graphlib::OpNode *op,
    graphlib::Shape commute_shape,
    graphlib::Shape clone_shape)    
{
    (void)graph;
    (void)initial_op;
    (void)op;
    (void)commute_shape;
    (void)clone_shape;
    return false;
}

static bool attempt_replace_pattern(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    graphlib::OpNode *op,
    graphlib::Shape commute_shape,
    graphlib::Shape clone_shape,
    bool commute_up = false) 
{
    if (not commute_up)
        return attempt_replace_downward_pattern(graph, initial_op, op, commute_shape, clone_shape);
    else
        return attempt_replace_upward_pattern(graph, initial_op, op, commute_shape, clone_shape);
}


static bool find_and_replace_incommutable_patterns(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    graphlib::Shape commute_shape,
    bool commute_up = false,
    graphlib::OpNode *from = nullptr,
    graphlib::OpNode *previous_op = nullptr)
{
    graphlib::OpNode *iter = from ? from : initial_op;
    auto clone_shape = initial_op->shape();

    bool replaced_pattern = false;
    while (not replaced_pattern)
    {   
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(iter);
        TT_ASSERT(op);
        
        log_trace(LogGraphCompiler, "  checking commute past {}", op->name());

        if (previous_op) {
            if (commute_up and are_bcasts_between_ops(graph, op, previous_op)) {
                log_trace(LogGraphCompiler, "  Bcast between {} and {} prevents input commute", op->name(), previous_op->name());
                break;
            }
            else if (not commute_up)
                handle_shape_change_through_bcast(graph, initial_op, previous_op, op, &commute_shape, &clone_shape);
        }

        // If we've run into an inverse op along this path, then theres nothing to replace
        if (are_compatible_ops(graph, initial_op, op, &commute_shape))
        {
            break;
        }
        // TODO: (lpanos) I dont think is_elementwise should return true for any of these ops, but for now it does
        bool can_commute = is_elementwise(op) and op->op_name() != "concatenate" and op->op_name() != "select" and op->op_name() != "interleave";

        if (not can_commute and op != initial_op)
        {   
            if (attempt_replace_pattern(graph, initial_op, op, commute_shape, clone_shape, commute_up)) {
                replaced_pattern = true;
            }
            break;
        }

        std::vector<graphlib::Node *> next_nodes = commute_up ? graph->data_operands(op) : graph->data_users(op);
        for (std::size_t i = 1; i < next_nodes.size(); ++i)
        {
            graphlib::OpNode *next_node = dynamic_cast<graphlib::OpNode *>(next_nodes[i]);
            replaced_pattern |= next_node and find_and_replace_incommutable_patterns(graph, initial_op, commute_shape, commute_up, next_node, op);
        }

        if (replaced_pattern)
            break;

        TT_ASSERT(next_nodes.size() > 0);
        if (not commute_up) {
            graphlib::OutputNode *output = dynamic_cast<graphlib::OutputNode *>(next_nodes[0]);
            if (output)
                break;
        }
        else {
            graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(next_nodes[0]);
            if (input)
                break;
        }
            
        previous_op = op;
        iter = dynamic_cast<graphlib::OpNode *>(next_nodes[0]);
        if (not iter)
            break;
    }

    return replaced_pattern;
}


bool replace_incommutable_patterns(graphlib::Graph *graph) {
    bool updated_anything = false;
    // return false; // TODO Enable later
    for (auto *node : graphlib::topological_sort(*graph))
    {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
        if (not op)
            continue;

        if (op->op_name() != "reshape")
            continue;

        if (not find_and_replace_incommutable_patterns(graph, op, shape_of_only_operand(graph, op))) {
            if (not find_and_replace_incommutable_patterns(graph, op, shape_of_only_operand(graph, op), true))
                continue;
        }
        updated_anything = true;
        break;
    }
    reportify::dump_graph(graph->name(), "replace_incommutable_patterns", graph);
    return updated_anything;
}

} // namespace tt::passes
