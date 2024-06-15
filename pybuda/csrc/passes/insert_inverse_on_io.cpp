// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/insert_inverse_on_io.hpp"
#include "passes/commute_utils.hpp"
#include <pybind11/pybind11.h>
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"
#include "passes/passes_utils.hpp"
#include "reportify/reportify.hpp"


namespace tt::passes 
{

using IOEdgeInfo = std::pair<graphlib::Edge, std::pair<graphlib::Shape, graphlib::Shape>>;

void add_inverse_to_input_edges(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    std::vector<std::pair<graphlib::Edge, std::pair<graphlib::Shape, graphlib::Shape>>> input_edges_and_shapes
)
{
    // two reshapes are needed, one to be eliminated with inverse op, and one to make it an inverse (i.e. shape of 
    // operand == output shape of inverse). Since the two are inverse of each other, tag first so it's not erased.
    for (auto input_edge_and_shape : input_edges_and_shapes)
    {
        auto edge = input_edge_and_shape.first;
        auto frist_reshape = input_edge_and_shape.second.first;
        auto second_reshape = input_edge_and_shape.second.second;
        auto name = initial_op->name() + "_input_commute_clone" + std::to_string(edge.edge_creation_id);
        auto *clone_0 = graph->add_node(initial_op->clone(name), graph->get_subgraph_id_for_node(initial_op->id()));
        graphlib::OpNode *clone_0_op = dynamic_cast<graphlib::OpNode *>(clone_0);
        clone_0_op->as<graphlib::TaggedNode>()->tag("dont_erase");
        update_reshape_attr(clone_0_op, frist_reshape);
        clone_0->set_shape(frist_reshape);
        log_trace(LogGraphCompiler, "  Input commute clone 0: {} set to shape: {}", name, frist_reshape);
        auto [incoming_edge, outgoing_edge] = insert_node_on_edge(graph, edge, clone_0);
        convert_implicit_to_explicit_bcasts(graph, incoming_edge);

        // Set output df to match producer
        clone_0->set_output_df(graph->node_by_id(incoming_edge.producer_node_id)->output_df());
    
        name = initial_op->name() + "_input_commute_clone" + std::to_string(outgoing_edge.edge_creation_id);
        auto *clone_1 = graph->add_node(initial_op->clone(name), graph->get_subgraph_id_for_node(initial_op->id()));
        graphlib::OpNode *clone_1_op = dynamic_cast<graphlib::OpNode *>(clone_1);
        update_reshape_attr(clone_1_op, second_reshape);
        clone_1->set_shape(second_reshape);
        log_trace(LogGraphCompiler, "  Input commute clone 1: {} set to shape: {}", name, second_reshape);
        auto [incoming_edge_1, outgoing_edge_1] = insert_node_on_edge(graph, outgoing_edge, clone_1);
        handle_change_rank(graph, clone_1);

        // Set output df to match producer
        clone_1->set_output_df(graph->node_by_id(incoming_edge_1.producer_node_id)->output_df());

        auto *input = dynamic_cast<graphlib::InputNode *>(graph->node_by_id(edge.producer_node_id));

        // TODO: constevaling transposes can cause tensor shape mismatch on parameter gradients for some reason. Should fix
        if (input and input->is_constant())
        {
            try_consteval_op(graph, clone_0_op, true);
        }
        else if (input and input->is_parameter())
        {
            if (clone_0_op->op_name() == "reshape")
                try_consteval_op(graph, clone_0_op, true);
            else if (clone_1_op->op_name() == "transpose" and not graph->enable_training() and not input->requires_grad())
                try_consteval_op(graph, clone_0_op, true);
        }
    }
}

void add_inverse_to_output_edge(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    IOEdgeInfo edge_and_shape)
{
    auto edge = edge_and_shape.first;
    auto commute_shape = edge_and_shape.second.first;
    auto clone_shape = edge_and_shape.second.second;

    auto name = initial_op->name() + "_output_commute_clone" + std::to_string(edge.edge_creation_id);
    auto *clone_0 = graph->add_node(initial_op->clone(name), graph->get_subgraph_id_for_node(initial_op->id()));
    graphlib::OpNode *clone_0_op = dynamic_cast<graphlib::OpNode *>(clone_0);
    update_reshape_attr(clone_0_op, commute_shape);
    clone_0->set_shape(commute_shape);
    log_trace(LogGraphCompiler, "  Output commute clone 0: {} set to shape: {}", name, commute_shape);
    auto [incoming_edge, outgoing_edge] = insert_node_on_edge(graph, edge, clone_0);
    convert_implicit_to_explicit_bcasts(graph, incoming_edge);

    // Set output df to match producer
    clone_0->set_output_df(graph->node_by_id(incoming_edge.producer_node_id)->output_df());

    name = initial_op->name() + "_output_commute_clone" + std::to_string(outgoing_edge.edge_creation_id);
    auto *clone_1 = graph->add_node(initial_op->clone(name), graph->get_subgraph_id_for_node(initial_op->id()));
    graphlib::OpNode *clone_1_op = dynamic_cast<graphlib::OpNode *>(clone_1);
    clone_1_op->as<graphlib::TaggedNode>()->tag("dont_erase");
    update_reshape_attr(clone_1_op, clone_shape);
    clone_1->set_shape(clone_shape);
    log_trace(LogGraphCompiler, "  Output commute clone 1: {}", name);
    auto [incoming_edge_1, outgoing_edge_1] = insert_node_on_edge(graph, outgoing_edge, clone_1);
    handle_change_rank(graph, clone_1);

    // Set output df to match producer
    clone_1->set_output_df(graph->node_by_id(incoming_edge_1.producer_node_id)->output_df());
}

void add_inverse_to_tm_edges(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    std::vector<std::pair<graphlib::Edge, std::pair<graphlib::Shape, graphlib::Shape>>> input_edges_and_shapes
) {
    for (auto input_edge_and_shape : input_edges_and_shapes)
    {
        auto edge = input_edge_and_shape.first;
        auto commute_shape = input_edge_and_shape.second.first;
        auto clone_shape = input_edge_and_shape.second.second;

        auto name = initial_op->name() + "_tm_commute_clone" + std::to_string(edge.edge_creation_id);
        auto *clone_0 = graph->add_node(initial_op->clone(name), graph->get_subgraph_id_for_node(initial_op->id()));
        graphlib::OpNode *clone_0_op = dynamic_cast<graphlib::OpNode *>(clone_0);
        update_reshape_attr(clone_0_op, commute_shape);
        clone_0->set_shape(commute_shape);
        log_trace(LogGraphCompiler, "  TM commute clone 0: {} set to shape: {}", name, commute_shape);
        auto [incoming_edge, outgoing_edge] = insert_node_on_edge(graph, edge, clone_0);
        convert_implicit_to_explicit_bcasts(graph, incoming_edge);
        // Set output df to match producer
        clone_0->set_output_df(graph->node_by_id(incoming_edge.producer_node_id)->output_df());
    }
}

std::vector<IOEdgeInfo> all_edges_to_input_nodes_commutable(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    graphlib::Shape commute_shape,
    graphlib::Shape clone_shape,
    graphlib::OpNode *from,
    graphlib::OpNode *previous_op)
{
    std::vector<std::pair<graphlib::Edge, std::pair<graphlib::Shape, graphlib::Shape>>> input_edges;
    graphlib::OpNode *iter = from ? from : initial_op;

    if (dynamic_cast<graphlib::OutputNode *>(graph->data_users(initial_op)[0]))
    {
        input_edges.clear();
        return input_edges;
    }
    bool found_input = false;
    while (not found_input)
    {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(iter);
        TT_ASSERT(op);

        if (previous_op and are_bcasts_between_ops(graph, op, previous_op))
        {
            log_trace(LogGraphCompiler, "  Bcast between {} and {} prevents input commute", op->name(), previous_op->name());
            input_edges.clear();
            return input_edges;
        }

        bool all_forks_commute_to_input = true;
        std::vector<graphlib::Node *> operands = graph->data_operands(op);
        for (std::size_t i = 1; (i < operands.size()) and all_forks_commute_to_input; ++i)
        {
            graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(operands[i]);
            if (input and (input->is_constant() or input->is_parameter()))
                continue;

            graphlib::OpNode *operand = dynamic_cast<graphlib::OpNode *>(operands[i]);
            auto commute_shape_copy = commute_shape;
            auto clone_shape_copy = clone_shape;
            bool can_commute_fork = can_commute_past_op(op, initial_op, graph, &commute_shape_copy, &clone_shape_copy, true, operands[i]);
            if (can_commute_fork and operand)
            {
                auto new_edges = all_edges_to_input_nodes_commutable(graph, initial_op, commute_shape_copy, clone_shape_copy, operand, op);
                all_forks_commute_to_input &= operand and not new_edges.empty();
                for (auto &edge : new_edges)
                {
                    input_edges.push_back(edge);
                }
            }
            else
            {
                all_forks_commute_to_input = false;
            }
        }

        bool can_commute = can_commute_past_op(op, initial_op, graph, &commute_shape, &clone_shape, true, operands[0]);
        auto input_node = dynamic_cast<graphlib::InputNode *>(operands[0]);
        if (input_node)
        {
            found_input = true;
        }
        if (can_commute and found_input and all_forks_commute_to_input and op != initial_op)
        {
            input_edges.push_back(std::make_pair(graph->operand_data_edges(op)[0], std::make_pair(commute_shape, clone_shape)));
            return input_edges;
        }
        else if (op != initial_op and (not can_commute or not all_forks_commute_to_input))
        {
            input_edges.clear();
            return input_edges;
        }
        previous_op = op;
        iter = dynamic_cast<graphlib::OpNode *>(operands[0]);
        if (not iter)
            break;
    }

    input_edges.clear();
    return input_edges;
}

std::pair<bool, std::unique_ptr<IOEdgeInfo>> find_commutable_output_edge(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    graphlib::Shape commute_shape,
    graphlib::OpNode *from = nullptr,
    graphlib::OpNode *previous_op = nullptr)
{
    graphlib::OpNode *iter = from ? from : initial_op;
    auto clone_shape = initial_op->shape();

    bool found_output = false;
    while (not found_output)
    {   
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(iter);
        TT_ASSERT(op);
        
        if (previous_op) {
            handle_shape_change_through_bcast(graph, initial_op, previous_op, op, &commute_shape, &clone_shape);
        }

        bool can_commute = can_commute_past_op(op, initial_op, graph, &commute_shape, &clone_shape);
        bool found_inverse = are_compatible_ops(graph, initial_op, op, &commute_shape);

        if (found_inverse)
            return std::make_pair<bool, std::unique_ptr<IOEdgeInfo>>(true, nullptr);

        bool all_forks_commute_to_output_or_inverse = true;
        std::vector<graphlib::Node *> users = graph->data_users(op);
        for (std::size_t i = 1; (i < users.size()) and all_forks_commute_to_output_or_inverse; ++i)
        {
            graphlib::OpNode *user = dynamic_cast<graphlib::OpNode *>(users[i]);
            auto [found_output_or_inverse, edge_info] = find_commutable_output_edge(graph, initial_op, commute_shape, user, op);
            if (edge_info) { // Found an output
                return std::make_pair<bool, std::unique_ptr<IOEdgeInfo>>((bool)found_output_or_inverse, std::move(edge_info));
            } 
            else if (not found_output_or_inverse) {
                // Did not find an output or inverse
                all_forks_commute_to_output_or_inverse = false;
            }
            
        }

        if ((!can_commute and op != initial_op) or (not all_forks_commute_to_output_or_inverse))
            break;

        std::vector<graphlib::Edge> partial_datacopy_edges = graph->user_partial_datacopy_edges(users[0]);
        graphlib::OutputNode *output = dynamic_cast<graphlib::OutputNode *>(users[0]);

        // Usually there is no point in inserting an invers on top of an output if the initial op in question is 
        // Adjacent to the output. Unless, this node forks.
        if (output and (op != initial_op or graph->user_data_edges(op).size() > 1) and partial_datacopy_edges.empty())
        {   
            // We do not want to add an inverse reshaep on the output unless it cancels out on all forks coming to that output
            if (not all_producer_forks_have_equivalent(graph, initial_op, commute_shape, op))
                break;

            return std::make_pair<bool, std::unique_ptr<IOEdgeInfo>>(true, std::unique_ptr<IOEdgeInfo>(new IOEdgeInfo(graph->user_data_edges(op)[0], std::make_pair(commute_shape, clone_shape))));
        }

        iter = dynamic_cast<graphlib::OpNode *>(users[0]);
        previous_op = op;
        if (not iter)
            break;
    }
    return std::make_pair<bool, std::unique_ptr<IOEdgeInfo>>(false, nullptr);
}

std::pair<std::vector<IOEdgeInfo>, bool> find_incommutable_downsrtream_tm(graphlib::Graph *graph, graphlib::OpNode *initial_op, graphlib::Shape commute_shape, graphlib::OpNode *from = nullptr, graphlib::OpNode *previous_op = nullptr)
{
    graphlib::OpNode *iter = from ? from : initial_op;
    auto clone_shape = initial_op->shape();

    std::vector<IOEdgeInfo> edges_to_insert;
    bool found_output = false;
    while (not found_output)
    {   
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(iter);
        TT_ASSERT(op);
        
        if (previous_op) {
            handle_shape_change_through_bcast(graph, initial_op, previous_op, op, &commute_shape, &clone_shape);
        }

        bool can_commute = can_commute_past_op(op, initial_op, graph, &commute_shape, &clone_shape);

        bool found_incommutable_tm = initial_op->op_name() == op->op_name() and op->op_name() == "reshape" and not are_compatible_ops(graph, initial_op, op, &commute_shape);
        bool found_inverse = are_compatible_ops(graph, initial_op, op, &commute_shape);
        if (op != initial_op and found_incommutable_tm)
        {   
            // We do not want to add an inverse reshaep on the output unless it cancels out on all forks coming to that output
            if (not all_producer_forks_have_equivalent(graph, initial_op, commute_shape, previous_op))
                break;
            bool already_contains_edge = false;
            auto edge_to_add = graph->operand_data_edges(op)[0];
            for (auto &edge_info : edges_to_insert) {
                if (edge_info.first.edge_creation_id == edge_to_add.edge_creation_id) {
                    already_contains_edge = true;
                    break;
                }
            }
            if (not already_contains_edge)
                edges_to_insert.push_back(IOEdgeInfo(graph->operand_data_edges(op)[0], std::make_pair(commute_shape, clone_shape)));
            return {edges_to_insert, true};
        }
        else if (op != initial_op and found_inverse) {
            return {edges_to_insert, true};
        }
        else if (op != initial_op and not can_commute and not found_inverse) {
            edges_to_insert.clear();
            return {edges_to_insert, false};
        }

        std::vector<graphlib::Node *> users = graph->data_users(op);
        for (std::size_t i = 1; (i < users.size()); ++i)
        {
            graphlib::OpNode *user = dynamic_cast<graphlib::OpNode *>(users[i]);
            auto [forked_output_edges, should_continue] = find_incommutable_downsrtream_tm(graph, initial_op, commute_shape, user, op);
            if (not should_continue) {
                edges_to_insert.clear();
                return {edges_to_insert, false};
            }
            for (auto &new_edge_info : forked_output_edges) {
                bool already_contains_edge = false;
                auto edge_to_add = new_edge_info.first;
                for (auto &edge_info : edges_to_insert) {
                    if (edge_info.first.edge_creation_id == edge_to_add.edge_creation_id) {
                        already_contains_edge = true;
                        break;
                    }
                }
                if (not already_contains_edge)
                    edges_to_insert.push_back(new_edge_info);
            }
        }

        iter = dynamic_cast<graphlib::OpNode *>(users[0]);
        previous_op = op;
        if (not iter)
            break;
    }
    return {edges_to_insert, true};
}

bool insert_inverse_on_inputs(graphlib::Graph *graph) 
{   
    bool attempt_update = true;
    bool updated_anything = false;

    while (attempt_update) 
    {
        attempt_update = false;
        for (auto *node : graphlib::topological_sort(*graph))
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
            if (not op)
                continue;

            if (op->as<graphlib::TaggedNode>()->has_tag("dont_erase"))
                continue;

            if (op->op_name() != "reshape" and op->op_name() != "transpose")
                continue;

            auto input_edges = all_edges_to_input_nodes_commutable(graph, op, op->shape(), shape_of_only_operand(graph, op));
            if (input_edges.empty())
                continue;

            add_inverse_to_input_edges(graph, op, input_edges);
            attempt_update = true;
            updated_anything = true;
            break;
        }
    }
    return updated_anything;
}

bool insert_inverse_on_outputs(graphlib::Graph *graph)
{
    bool attempt_update = true;
    bool updated_anything = false;

    while (attempt_update)
    {
        attempt_update = false;
        for (auto *node : graphlib::topological_sort(*graph))
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
            if (not op)
                continue;

            if (op->as<graphlib::TaggedNode>()->has_tag("dont_erase"))
                continue;

            if (op->op_name() != "reshape" and op->op_name() != "transpose")
                continue;

            auto [_, output_edge_info] = find_commutable_output_edge(graph, op, shape_of_only_operand(graph, op));
            if (not output_edge_info)
                continue;

            add_inverse_to_output_edge(graph, op, *output_edge_info);
            attempt_update = true;
            updated_anything = true;
            break;
        }
    }
    return updated_anything;
}

bool insert_inverse_on_downstream_tms(graphlib::Graph *graph) {
    bool updated_anything = false;
    for (auto *node : graphlib::topological_sort(*graph))
    {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
        if (not op)
            continue;

        if (op->as<graphlib::TaggedNode>()->has_tag("dont_erase"))
            continue;

        if (op->op_name() != "reshape")
            continue;

        auto commute_shape = shape_of_only_operand(graph, op);
        if (are_different_ranked_shapes_equivalent(commute_shape, op->shape()))
            continue;
        auto [output_edges, _] = find_incommutable_downsrtream_tm(graph, op, shape_of_only_operand(graph, op));
        if (output_edges.empty())
            continue;
        
        add_inverse_to_tm_edges(graph, op, output_edges);
        updated_anything = true;
        break;
    }
    
    return updated_anything;
}

} // namespace tt::passes