// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/erase_inverse_ops.hpp"

#include <pybind11/pybind11.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"
#include "passes/passes_utils.hpp"
#include "passes/commute_utils.hpp"
#include "reportify/reportify.hpp"

namespace tt::passes
{

static std::vector<graphlib::Node *> find_path_to_inverse_op(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    graphlib::Shape commute_shape,
    graphlib::OpNode *from = nullptr,
    graphlib::OpNode *previous_op = nullptr)
{
    std::vector<graphlib::Node *> path;

    graphlib::OpNode *iter = from ? from : initial_op;
    auto clone_shape = initial_op->shape();

    bool found_inverse = false;
    while (not found_inverse)
    {   
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(iter);
        TT_ASSERT(op);
        bool can_commute = true;
        if (previous_op) {
            auto [can_commute_through_bcast, bcast_dims] = handle_shape_change_through_bcast(graph, initial_op, previous_op, op, &commute_shape, &clone_shape);
            can_commute = can_commute_through_bcast;
            if (not can_commute)
                break;
        }

        found_inverse = are_compatible_ops(graph, initial_op, op, &commute_shape);
        can_commute = can_commute_past_op(op, initial_op, graph, &commute_shape, &clone_shape, false);

        bool all_forks_have_inverse = true;
        std::vector<graphlib::Node *> users = graph->data_users(op);
        for (std::size_t i = 1; (i < users.size()) and all_forks_have_inverse; ++i)
        {
            graphlib::OpNode *user = dynamic_cast<graphlib::OpNode *>(users[i]);
            all_forks_have_inverse &= user and not find_path_to_inverse_op(graph, initial_op, commute_shape, user, op).empty();
        }

        if (found_inverse or can_commute or (op == initial_op))
        {
            path.push_back(op);
        }
        else
        {
            break;
        }

        if (not all_forks_have_inverse)
            break;

        TT_ASSERT(users.size() > 0);
        graphlib::OutputNode *output = dynamic_cast<graphlib::OutputNode *>(users[0]);
        if (output)
            break;
            
        previous_op = op;
        iter = dynamic_cast<graphlib::OpNode *>(users[0]);
        if (not iter)
            break;
    }

    if (not found_inverse)
        path.clear();

    return path;
}


bool has_broadcast_on_dim(graphlib::Graph *graph, graphlib::Edge edge, int needed_dim)
{
    auto tms = graph->get_edge_attributes(edge)->get_tms();
    for (graphlib::OpType &op_type : tms)
    {
        if (op_type.op == "broadcast")
        {
            int dim = std::get<int>(op_type.attr[0]);
            if (dim == needed_dim)
            {
                return true;
            }
        }
    }
    return false;
}

void set_bcast_dims(graphlib::Graph *graph, std::vector<int> &volumes, graphlib::Edge edge) {
    graph->get_edge_attributes(edge)->clear_broadcast_dims();

    for (std::size_t i = 0; i < volumes.size(); i++) {
        int volume = volumes[i];
        if (volume > 1) {
            graph->get_edge_attributes(edge)->set_broadcast_dim(i, volume, false); 
        }
    }
}

void commute_and_bypass(graphlib::Graph *graph, std::vector<graphlib::Node *> const &path)
{
    TT_ASSERT(path.size() >= 2);
    graphlib::OpNode *first = path.front()->as<graphlib::OpNode>();
    graphlib::OpNode *last = path.back()->as<graphlib::OpNode>();
    bool retain_operand_dim;
    std::pair<int, int> operand_dims;
    log_debug(LogGraphCompiler, "Commute and bypass inverse nodes: {} -> {}", first->name(), last->name());
    graphlib::OpType golden_transform = first->op_type();

    graphlib::Shape commute_shape = shape_of_only_operand(graph, first);
    graphlib::Shape clone_shape = first->shape();

    for (std::size_t i = 1; i < path.size(); ++i)
    {
        retain_operand_dim = false;

        graphlib::Node *producer = path[i - 1];
        graphlib::Node *consumer = path[i];
        auto consumer_df_before = consumer->output_df();

        // Handle forks (not on this `path`)
        for (graphlib::Edge user_edge : graph->user_data_edges(producer))
        {
            if (user_edge.consumer_node_id == consumer->id())
                continue;

            auto fork_commute_shape = commute_shape;
            auto fork_clone_shape = clone_shape;

            graphlib::OpNode *producer_as_op = dynamic_cast<graphlib::OpNode *>(producer);
            auto edge_consumer_as_op = dynamic_cast<graphlib::OpNode *>(graph->node_by_id(user_edge.consumer_node_id));
            if (producer_as_op and edge_consumer_as_op) {
                
                auto [commute_bcasts, clone_bcasts] = handle_shape_change_through_bcast(graph, first, producer_as_op, edge_consumer_as_op, &fork_commute_shape, &fork_clone_shape).second;
                graphlib::Edge between_edge = retrieve_between_edge(graph, producer, consumer);
                set_bcast_dims(graph, commute_bcasts, between_edge);  

            }
            auto name = first->name() + "_user_commute_clone" + std::to_string(user_edge.edge_creation_id);
            log_trace(LogGraphCompiler, "  User commute clone: {} -> between {} and {} ", name, producer->name(), graph->node_by_id(user_edge.consumer_node_id)->name());
            auto *clone = graph->add_node(first->clone(name), graph->get_subgraph_id_for_node(user_edge.consumer_node_id));
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(clone);
            update_reshape_attr(op, fork_clone_shape);
            clone->set_shape(fork_clone_shape);
            log_trace(LogGraphCompiler, "  User commute clone shape: {}", fork_clone_shape);
            auto [incoming_edge, outgoing_edge] = insert_node_on_edge(graph, user_edge, clone);
            convert_implicit_to_explicit_bcasts(graph, incoming_edge);
            handle_change_rank(graph, clone);
            clone->set_output_df(graph->node_by_id(incoming_edge.producer_node_id)->output_df());
        }
        
        std::vector<graphlib::OpType::Attr> original_op_attrs{};
        // Set the shape to the desired final shape for this whole path
        if (graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(consumer))
        {   
            original_op_attrs = op->op_attrs();

            graphlib::OpNode *producer_as_op = dynamic_cast<graphlib::OpNode *>(producer);
            if (producer_as_op) {
                // Must change commute shape, clone shape, and golden transform if there are broadcasts on the incoming edge
                auto [commute_bcasts, clone_bcasts] = handle_shape_change_through_bcast(graph, first, producer_as_op, op, &commute_shape, &clone_shape).second;
                if (golden_transform.op == "reshape") {
                    for (std::size_t i = 0; i < golden_transform.attr.size(); i++) {
                        int current_dim = std::get<int>(golden_transform.attr[i]);
                        golden_transform.attr[i] = clone_bcasts[i]*current_dim;
                    }
                }

                graphlib::Edge between_edge = retrieve_between_edge(graph, producer, consumer);
                set_bcast_dims(graph, commute_bcasts, between_edge);                
            }

            if (op->op_name() == "reduce_avg" or op->op_name() == "reduce_sum")
            {
                commute_through_reduce(
                    graph, 
                    op, 
                    first, 
                    producer_as_op, 
                    path[i+1],
                    &commute_shape, 
                    &clone_shape, 
                    false, // Not only checking for commutability
                    &retain_operand_dim, 
                    &operand_dims, 
                    &golden_transform
                );
            }
            else if (op->op_name() == "concatenate")
            {
                commute_through_concat(
                    graph, 
                    op, 
                    first, 
                    producer_as_op, 
                    &commute_shape, 
                    &clone_shape, 
                    false, // Not only checking for commutability
                    &retain_operand_dim, 
                    &operand_dims, 
                    &golden_transform
                );
            }
            else if (op->op_name() == "select")
            {
                commute_through_select(
                    graph, 
                    op, 
                    first, 
                    producer_as_op, 
                    &commute_shape, 
                    &clone_shape, 
                    false, // Not only checking for commutability
                    &retain_operand_dim, 
                    &operand_dims, 
                    &golden_transform
                );
            }
            else if (is_elementwise(op))
            {
                commute_through_eltwise(op, &commute_shape, &golden_transform);
            }
            else if (is_quantization_ops(op)) {
                commute_through_quantization(op, first, false, &commute_shape, &golden_transform);
            }
            else if (op->op_name() == "squeeze") {
                commute_through_squeeze(op, first, &commute_shape, &clone_shape, &golden_transform, false, false);
            }
            log_trace(LogGraphCompiler, "  Op node: {} -> shape set to {}", consumer->name(), commute_shape);
        }

        // Handle nary operands (not on this `path`)
        std::vector<graphlib::Edge> consumer_operands = graph->operand_data_edges(consumer);
        for (uint32_t operand_index = 0; operand_index < consumer_operands.size(); operand_index++)
        {
            graphlib::Edge operand_edge = consumer_operands[operand_index];
            if (operand_edge.producer_node_id == producer->id())
                continue;

            convert_implicit_to_explicit_bcasts(graph, operand_edge);
            auto name = last->name() + "_operand_commute_clone" + std::to_string(operand_edge.edge_creation_id);
            graphlib::Node *clone = graph->add_node(last->clone(name), graph->get_subgraph_id_for_node(operand_edge.producer_node_id));
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(clone);
            log_trace(LogGraphCompiler, "  Operand commute clone: {} -> between {} and {} ", name, consumer->name(), graph->node_by_id(operand_edge.producer_node_id)->name());

            // Special case for operand clones on a quantization scale
            auto *consumer_op = dynamic_cast<graphlib::OpNode *>(consumer);
            if (is_quantization_ops(consumer_op) and operand_index == 1) {
                
                // The shape should be all 1's except for (possiby) the quantization axis
                auto updated_commute_shape = commute_shape;
                int quant_axis = std::get<int>(consumer_op->op_attrs()[1]);
                updated_commute_shape[quant_axis] = consumer_op->shape()[quant_axis];
                update_reshape_attr(op, updated_commute_shape);
                clone->set_shape(updated_commute_shape);
                log_trace(LogGraphCompiler, "  Operand commute clone shape: {}", updated_commute_shape);
                
            }
            else if (retain_operand_dim)
            {
                auto updated_commute_shape = commute_shape;
                updated_commute_shape[operand_dims.second] = graph->node_by_id(operand_edge.producer_node_id)->shape()[operand_dims.first];
                update_reshape_attr(op, updated_commute_shape);
                clone->set_shape(updated_commute_shape);
                log_trace(LogGraphCompiler, "  Operand commute clone shape: {}", updated_commute_shape);
            }
            else
            {
                update_reshape_attr(op, commute_shape);
                clone->set_shape(commute_shape);
                log_trace(LogGraphCompiler, "  Operand commute clone shape: {}", commute_shape);
            }

            // Operand commute clones for squeeze/unsqueeze need to be swapped to the opposite op
            if (first->op_name() == "unsqueeze") {
                op->change_op_type("squeeze");
                op->overwrite_op_attrs({first->op_attrs()[0]});
            }
            else if (first->op_name() == "squeeze") {
                op->change_op_type("unsqueeze");
                op->overwrite_op_attrs({first->op_attrs()[0], (int)graph->node_by_id(operand_edge.producer_node_id)->shape().size()});
            }

            // Inputs can have mismatched number of dims and still function corretly to the consuming op
            // Thus if the op we are commuting through has shape (1, 128, 1024) and its operand is a param with shape (1024,)
            // Placing an unsqueeze clone will cause the input to be (1, 1024) which isn't wrong but also is not correct.
            // In this case we shall convert the clone to a reshape which contains the correct number of unsqueezes implicitly.
            if ((first->op_name() == "unsqueeze" or first->op_name() == "squeeze") and dynamic_cast<graphlib::InputNode *>(graph->node_by_id(operand_edge.producer_node_id))) {
                op->change_op_type("reshape");
                graphlib::Shape op_shape = graphlib::Shape::create(std::vector<uint32_t>(consumer->shape().size(), 1));
                auto input = dynamic_cast<graphlib::InputNode *>(graph->node_by_id(operand_edge.producer_node_id));

                for (int i = -1; i >= -(int)input->shape().size(); i--) {
                    if (i + (int)op_shape.size() >= 0)
                        op_shape[i] = input->shape()[i];
                    else {
                        TT_ASSERT(input->shape()[i] == 1, "After this point all dims should be 1 else the squeeze op is not valid.");
                    }
                }
                std::vector<graphlib::OpType> tms = graph->get_edge_attributes(operand_edge)->get_tms();
                for (graphlib::OpType& tm : tms) {
                    if (tm.op == "broadcast") {
                        int dim = std::get<int>(tm.attr[0]);
                        if (dim >= 0) {
                            dim -= input->shape().size();
                        }
                        int volume = std::get<int>(tm.attr[1]);
                        op_shape[dim] *= volume;
                    }
                }
                op->set_shape(op_shape);
                update_reshape_attr(op, op_shape);
            }

            auto [in_edge, out_edge] = insert_node_on_edge(graph, operand_edge, clone);
            // Set dataformat to match producer on operand edge
            clone->set_output_df(graph->node_by_id(in_edge.producer_node_id)->output_df());

            handle_change_rank(graph, clone);
            try_commute_bcast_through_clone(graph, op);
            if (graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(graph->data_operands(clone)[0]))
                try_consteval_input_no_operand_forks(graph, input, true);
        }

        // Maintain df from before commute
        consumer->set_output_df(consumer_df_before);
    }
    // if first and last are inverse due to broadcast, broadcast dims need to be updated
    size_t broadcast_volume = total_broadcast_volume(graph, graph->operand_data_edges(last)[0]);
    auto are_inverse = are_inverse_with_broadcast(shape_of_only_operand(graph, first), last->shape(), broadcast_volume);
    if (are_inverse.first)
    {
        auto edge = graph->operand_data_edges(last)[0];
        graph->get_edge_attributes(edge)->clear_broadcast_dims();
        graph->get_edge_attributes(edge)->set_broadcast_dim(are_inverse.second, broadcast_volume, false);
        
    }
    bool is_squeeze_unsqueeze = first->op_name() == "squeeze" or first->op_name() == "unsqueeze";
    int first_id = first->id();
    int last_id = last->id();
    // `first` has commuted to `last`, these nodes are now back to back and cancel each other out
    auto change_rank = [graph, path, first_id, last_id, is_squeeze_unsqueeze](graphlib::Edge new_edge, graphlib::Edge old_edge) {
        // If the inverse pair are squeeze/unsqueeze and ther are adjacent to eachother (i.e path length of 2)
        // We do not need to handle_change_rank. If we do, handle change_rank will end up adding back a pair
        // of adjacent squeeze/unsqueeze nodes unnecessarily.
        // We check the old edge because in the event 'first' forks we still want to handle_change_rank on those edges
        if (not is_squeeze_unsqueeze 
            or (first_id != old_edge.producer_node_id) 
            or (last_id != old_edge.consumer_node_id))
        {
            handle_change_rank(graph, new_edge);   
        }
    };
    bypass_node(graph, first, true, change_rank);
    bypass_node(graph, last, true, change_rank);
}

bool erase_inverse_ops(graphlib::Graph *graph)
{
    // Three step process:
    // 1. Find all inverse ops that can be commuted to each other or commuted to an ouptut
    //      -   If commuting to output add an inverse op, plus a reshape back to original the original shape 
    //          which will be lowered into reinterpret shape
    // 2. Find all ops that can be commuted to an input and add inverse op to the input
    // 3. Repeat step 1 to eliminate newly created ops and their inverse
    bool attempt_update = true;
    bool updated_anything = false;
    while (attempt_update)
    {   
        // Set to false here because we want to stop looping if no update occurs
        attempt_update = false;
        for (auto *node : graphlib::topological_sort(*graph))
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
            if (not op)
                continue;

            if (op->as<graphlib::TaggedNode>()->tag_value_or("dont_erase", false))
                continue;

            if (match_fns.find(op->op_name()) == match_fns.end())
                continue;


            std::vector<graphlib::Node *> path = find_path_to_inverse_op(graph, op, shape_of_only_operand(graph, op));
            if (path.empty())
                continue;

            commute_and_bypass(graph, path);
            attempt_update = true;
            updated_anything = true;
            break;
        }
    }
    return updated_anything;
}
}  // namespace tt::passes
