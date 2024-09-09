// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/consteval.hpp"

#include <pybind11/pybind11.h>

#include <algorithm>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "reportify/reportify.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{
static bool input_can_consteval(graphlib::Graph *graph, graphlib::InputNode *input)
{
    if (input->as<graphlib::TaggedNode>()->has_tag("dont_consteval"))
        return false;
    auto has_same_fork_destinations = [](std::vector<graphlib::Node *> const &nodes) -> bool {
        std::vector<graphlib::Node *> ids;
        for (auto const &node : nodes)
        {
            if (std::find(ids.begin(), ids.end(), node) != ids.end())
                return true;
            ids.push_back(node);
        }
        return false;
    };

    // Generally we don't want to consteval broadcast or repeat as it
    // causes the input to blow up in size with duplicated data
    auto is_broadcast_or_repeat = [](graphlib::Node *n) {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(n);
        if (not op)
            return false;
        return op->op_name() == "broadcast" or op->op_name() == "repeat" or op->op_name() == "repeat_dim";
    };

    // We want to go from weights->quantize->dequantize to quantized_weights->dequantize
    // without constevaling dequantize, as it would cancel the quantize op
    auto is_dequantize = [](graphlib::Node *node) {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
        if (not op)
            return false;

        return op->op_name() == "dequantize" or op->op_name() == "buda_dequantize";
    };

    TT_ASSERT(graphlib::is_consteval_capable_input_type(input));
    std::vector<graphlib::Node *> users = graph->data_users(input);
    auto user_can_consteval = [graph, is_broadcast_or_repeat, is_dequantize](graphlib::Node *n) {
        return graphlib::is_consteval_capable_op(graph, n, true /*allow_forks*/) and not is_broadcast_or_repeat(n) and not is_dequantize(n);
    };
    return not has_same_fork_destinations(users) and std::all_of(users.begin(), users.end(), user_can_consteval);
    // TODO: nsmith enable this
    // bool users_can_consteval = (not graph->enable_training() or input->is_constant())
    //                                ? std::any_of(users.begin(), users.end(), user_can_consteval)
    //                                : std::all_of(users.begin(), users.end(), user_can_consteval);
    // return users_can_consteval and not has_same_fork_destinations(users);
}

static std::vector<graphlib::Node *> split_consteval_input_forks(
    graphlib::Graph *graph, std::vector<graphlib::Node *> &inputs)
{
    std::vector<graphlib::Node *> new_inputs;
    std::unordered_map<graphlib::Node *, std::vector<graphlib::Node *>> removed_to_forked;

    for (graphlib::Node *node : inputs)
    {
        graphlib::InputNode *input = node->as<graphlib::InputNode>();
        if (not input_can_consteval(graph, input))
            continue;

        std::vector<graphlib::Edge> users = graph->user_data_edges(input);
        if (users.size() == 1)
            continue;

        input->get_consteval_graph(graph, true, true); // create graph before clone so input node name is correct
        for (int i = 0; i < (int)users.size(); ++i)
        {
            graphlib::Edge const &edge = users[i];
            log_trace(
                LogConstEval,
                "Split fork input {} -> {}",
                input->name(),
                graph->node_by_id(edge.consumer_node_id)->name());
 
            auto new_edge = clone_input_forking_edge(graph, edge, true /* allow_single_user */);
            auto new_input = graph->node_by_id(new_edge.producer_node_id);
            new_inputs.push_back(new_input);
            removed_to_forked[node].push_back(new_input);
        }
    }

    for (auto iter : removed_to_forked) {
        graphlib::Node *node = iter.first;
        graphlib::Node *first_forked = iter.second.front();

        auto remove_iter = std::find(inputs.begin(), inputs.end(), node);
        TT_ASSERT(remove_iter != inputs.end(), "Node not found in inputs, this should never happen");
        inputs.erase(std::find(inputs.begin(), inputs.end(), node));
        auto removed_node = graph->remove_node(node);

        // Need to maintain original name because user can access it by name
        first_forked->set_name(removed_node->name());
    }
    return new_inputs;
}

static std::vector<graphlib::Node *> consteval_input(graphlib::Graph *graph, graphlib::InputNode *input)
{
    std::vector<graphlib::Node *> users = graph->data_users(input);
    TT_ASSERT(users.size() == 1, "Input forks, this should have been handled by split_consteval_input_forks");
    graphlib::Node *user = users[0];
    log_debug(LogConstEval, "Promoting node - Graph: {} - Node: {}", input->name(), user->name());

    std::vector<graphlib::Node *> user_other_operands = graph->data_operands(user);

    auto iter = std::find(user_other_operands.begin(), user_other_operands.end(), input);
    TT_ASSERT(iter != user_other_operands.end());
    user_other_operands.erase(iter);  // every operand except for `input` is removed in `promote_node`

    // The other user operands is essentially cloned if they themselves have multiple users.
    // One clone will go in the consteval graph of the passed <*input>. And the other
    // will remain in the main graph. We do not want to include it in removed_operands
    // if this is the case as it wont have actually been removed from the graph.
    std::vector<graphlib::Node *> removed_operands;
    for (graphlib::Node * user_operand : user_other_operands) {
        if (graph->data_users(user_operand).size() == 1)
            removed_operands.push_back(user_operand);
    }


    graphlib::ConstEvalGraph *consteval_graph = input->get_consteval_graph(graph, true, true);
    consteval_graph->promote_node(graph, user);

    return removed_operands;
}

void run_consteval_graph_pass(graphlib::Graph *graph)
{
    auto set_difference = [](std::vector<graphlib::Node *> &a, std::vector<graphlib::Node *> const &b)  // a - b
    {
        for (auto *n : b)
        {
            auto iter = std::find(a.begin(), a.end(), n);
            if (iter != a.end())
                a.erase(iter);
        }
    };

    std::vector<graphlib::Node *> needs_visit =
        graphlib::topological_sort(*graph, graphlib::is_consteval_capable_input_type);
    while (not needs_visit.empty()) {
        graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(needs_visit.back());
        needs_visit.pop_back();

        // Will consteval the user(s) of an input as long as each user is identical and have no operands other than the input.
        if (try_consteval_input_no_operand_forks(graph, input, true)) {
            needs_visit.push_back(input);
            continue;
        } 
        // In the event the input has one user, but its user has multiple operands, we can still consteval.
        else if (input_can_consteval(graph, input) and graph->data_users(input).size() == 1) {
            std::vector<graphlib::Node *> removed_operands = consteval_input(graph, input);
            needs_visit.push_back(input);
            set_difference(needs_visit, removed_operands);
            continue;
        }

        // If multiple operands exist
        std::vector<graphlib::Node *> inputs_vec{input};
        std::vector<graphlib::Node *> new_inputs = split_consteval_input_forks(graph, inputs_vec);
        if (not new_inputs.empty()) {
            needs_visit.insert(needs_visit.end(), new_inputs.begin(), new_inputs.end());
        }
    }

    // Dump reportify consteval graphs
    for (graphlib::Node *node : graphlib::topological_sort(*graph, graphlib::is_consteval_capable_input_type))
    {
        graphlib::InputNode *input = node->as<graphlib::InputNode>();
        if (auto *consteval_graph = input->get_consteval_graph())
            reportify::dump_consteval_graph(graph->name(), input->name(), consteval_graph->get_graph());
    }
}
}  // namespace tt::passes
