// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placer/lower_to_placer.hpp"

#include <fmt/ranges.h>

#include <unordered_set>

#include "autograd/autograd.hpp"
#include "backend_api/device_config.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/utils.hpp"
#include "scheduler/utils.hpp"
#include "utils/assert.hpp"
#include "utils/logger.hpp"

using std::unordered_set;
using std::tuple;

// Aliases
using tt_node = tt::graphlib::Node;
using tt_graph = tt::graphlib::Graph;

namespace tt {
using namespace graphlib;
namespace placer {
namespace lowering {

bool are_dependencies_resolved(const tt_graph *graph, const string& node_name, const unordered_set<string>& visited, NodeEpochType epoch_type)
{
    tt_node* node = graph->get_node_by_name(node_name);
    for (tt_node* operand : graph->data_operands(node)) {
        if ((int)operand->get_epoch_type() < (int)epoch_type)
        {
            continue;
        }
        if (operand->node_type() == NodeType::kInput)
        {
            continue;
        }
        if (visited.find(operand->name()) == visited.end())
        {
            return false;
        }
    }

    return true;
}

vector<tuple<int, string, int>> get_opt_op_group_to_place(
    const Graph* graph, Node* fwd_node, const unordered_map<string, int>& op_to_schedule_index)
{
    vector<tuple<int, string, int>> ret;

    for (tt_node* operand : graph->data_operands(fwd_node))
    {
        if (operand->node_type() != NodeType::kInput) { continue; }

        for (const auto& [operand_index, input_opt_nodes] : graph->get_opt_nodes(operand))
        {
            for (tt_node* opt_node : input_opt_nodes)
            {
                if (op_to_schedule_index.find(opt_node->name()) != op_to_schedule_index.end()) {
                    ret.emplace_back(op_to_schedule_index.at(opt_node->name()), opt_node->name(), operand_index);
                }
            }
        }
    }

    for (const auto& [operand_index, opt_nodes] : graph->get_opt_nodes(fwd_node))
    {
        for (tt_node* opt_node : opt_nodes)
        {
            if (op_to_schedule_index.find(opt_node->name()) != op_to_schedule_index.end()) {
                ret.emplace_back(op_to_schedule_index.at(opt_node->name()), opt_node->name(), operand_index);
            }
        }
    }
    return ret;
}


unordered_map<string, vector<string>> get_fwd_to_bwd_nodes(graphlib::Graph const* graph)
{
    unordered_map<string, vector<string>> fwd_to_bwd_nodes;
    for (Node* node : tt::graphlib::topological_sort(*graph))
    {
        if (node->get_epoch_type() == NodeEpochType::Forward and node->node_type() == NodeType::kBudaOp)
        {
            Node* fwd_node = node;
            const string& fwd_node_name = fwd_node->name();
            fwd_to_bwd_nodes[fwd_node_name] = {};
            // Compute any recompute if exists
            for (const Node* fwd_input_node : scheduler::get_schedule_predecessors(graph, fwd_node))
            {
                for (Edge fwd_input_recompute_edge : graph->user_edges(
                         fwd_input_node,
                         [](Edge e) { return e.edge_type == graphlib::EdgeType::kAutogradFwdToRecompute; }))
                {
                    Node* fwd_input_recompute_node = graph->node_by_id(fwd_input_recompute_edge.consumer_node_id);
                    if (fwd_input_recompute_node->node_type() == NodeType::kBudaOp)
                    {
                        fwd_to_bwd_nodes[fwd_node_name].push_back(fwd_input_recompute_node->name());
                    }
                }
            }

            for (Edge bwd_edge : graph->user_edges(
                     fwd_node, [](Edge e) { return e.edge_type == graphlib::EdgeType::kAutogradFwdToBwd; }))
            {
                Node* bwd_node = graph->node_by_id(bwd_edge.consumer_node_id);
                if (bwd_node->node_type() == NodeType::kBudaOp)
                {
                    fwd_to_bwd_nodes[fwd_node_name].push_back(bwd_node->name());
                }
            }
            for (Node* fwd_input_node : graph->data_operands(fwd_node))
            {
                if (fwd_input_node->node_type() == NodeType::kInput)
                {
                    for (Edge bwd_edge : graph->user_edges(
                             fwd_input_node,
                             [](Edge e) { return e.edge_type == graphlib::EdgeType::kAutogradFwdToBwd; }))
                    {
                        Node* bwd_node = graph->node_by_id(bwd_edge.consumer_node_id);
                        if (bwd_node->node_type() == NodeType::kBudaOp)
                        {
                            fwd_to_bwd_nodes[fwd_node_name].push_back(bwd_node->name());
                        }
                    }
                }
            }
        }
    }
    return fwd_to_bwd_nodes;
}

// TODO(jchu): commonize with above
unordered_map<string, map<int, vector<string>>> get_fwd_to_opt_nodes(
    graphlib::Graph const *graph,
    const vector<string>& scheduled_ops)
{
    unordered_map<string, map<int, vector<string>>> fwd_to_opt_nodes;

    if (not graph->contains_opt_nodes())
    {
        return {};
    }


    unordered_map<string, int> op_to_schedule_index;

    for (int i = 0; i < (int)scheduled_ops.size(); ++i) {
        op_to_schedule_index[scheduled_ops[i]] = i;
    }

    unordered_set<string> visited_ops;
    vector<string> unprocessed;
    for (const string& fwd_node_name : scheduled_ops)
    {
        tt_node* fwd_node = graph->get_node_by_name(fwd_node_name);
        NodeEpochType epoch_type = fwd_node->get_epoch_type();

        if (epoch_type == NodeEpochType::Forward)
        {
            vector<tuple<int, string, int>> opt_ops_to_place = get_opt_op_group_to_place(graph, fwd_node, op_to_schedule_index);
            std::sort(opt_ops_to_place.begin(), opt_ops_to_place.end());

            for (const auto& [_, opt_node_name, operand_index] : opt_ops_to_place)
            {
                if (visited_ops.find(opt_node_name) == visited_ops.end())
                {
                    if (are_dependencies_resolved(graph, opt_node_name, visited_ops, NodeEpochType::Optimizer)) {
                        fwd_to_opt_nodes[fwd_node_name][operand_index].push_back(opt_node_name);
                        visited_ops.insert(opt_node_name);
                    } else {
                        unprocessed.push_back(opt_node_name);
                    }
                }
            }
        }
    }
    for (const string& name : unprocessed) {
        if (visited_ops.find(name) == visited_ops.end()) {
            log_fatal("{} was not included to be placed.", name);
        }
    }


    return fwd_to_opt_nodes;
}

unordered_map<string, NodeEpochType> get_op_to_epoch_type_mapping(
    tt_graph const* graph, const vector<string>& scheduled_ops) {
    unordered_map<string, NodeEpochType> op_to_epoch_type;
    for (const string& op_name : scheduled_ops)
    {
        tt_node* node = graph->get_node_by_name(op_name);
        op_to_epoch_type[op_name] = node->get_epoch_type();
    }
    return op_to_epoch_type;
}

unordered_map<string, bool> get_op_to_grad_op_mapping(
    tt_graph const* graph, const vector<string>& scheduled_ops) {
    unordered_map<string, bool> op_to_grad_op;
    for (const string& op_name : scheduled_ops)
    {
        tt_node* node = graph->get_node_by_name(op_name);
        if (node->node_type() == NodeType::kBudaOp) {
            op_to_grad_op[op_name] = node->as<BudaOpNode>()->is_gradient_op();
        } else {
            op_to_grad_op[op_name] = false;
        }
    }
    return op_to_grad_op;
}


unordered_map<string, bool>
get_op_to_recompute_mapping(graphlib::Graph const* graph, const vector<string>& scheduled_ops)
{
    unordered_map<string, bool> op_to_recompute;
    for (const string& op_name : scheduled_ops)
    {
        tt_node* node = graph->get_node_by_name(op_name);
        op_to_recompute[op_name] = graphlib::is_recompute(graph, node);
    }
    return op_to_recompute;
}

unordered_set<string> get_output_nodes(const graphlib::Graph *graph)
{
    unordered_set<string> output_ops;
    for (Node *n: graph->nodes_by_type(graphlib::NodeType::kOutput))
    {
        auto partial_datacopy_edges = graph->user_edges(n, [](Edge e) { return e.edge_type == graphlib::EdgeType::kPartialDataCopy; });
        if (not partial_datacopy_edges.empty())
            continue;

        std::vector<Edge> edges = graph->operand_data_edges(n);
        TT_ASSERT(edges.size() == 1);
        Node *source = graph->node_by_id(edges[0].producer_node_id);
        TT_ASSERT(source->node_type() == NodeType::kBudaOp);
        output_ops.insert(source->name());
    }

    return output_ops;
}

vector<string> generate_placer_schedule(tt_graph const* graph, PlacementScheduleOrder) {
    vector<string> scheduled_nodes;
    for (tt_node* node : tt::graphlib::topological_sort(*graph))
    {
        if (node->node_type() != NodeType::kInput and node->node_type() != NodeType::kOutput and node->node_type() != NodeType::kQueue)
        {
            scheduled_nodes.push_back(node->name());
        }
    }
    return scheduled_nodes;
}

static unordered_set<string> tag_ops_for_epoch_or_chip_break(
    const vector<vector<string>>& op_names_to_epoch_or_chip_break,
    const vector<string>& scheduled_ops,
    graphlib::Graph const* graph,
    bool /*is_epoch_break*/)
{
    check_user_defined_op_names_exist_in_schedule(op_names_to_epoch_or_chip_break, scheduled_ops);

    unordered_set<string> ops_tagged_for_epoch_break;
    for (const auto& op_names : op_names_to_epoch_or_chip_break)
    {
        if (op_names.size() == 1)
        {
            ops_tagged_for_epoch_break.insert(op_names[0]);
        }
        else
        {
            // select the op that comes earliest in the schedule. not really expecting a ton of these iterations
            // we'll just loop over scheduled_ops to simplify
            unordered_set<string> op_names_set(op_names.begin(), op_names.end());
            for (const string& scheduled_op : scheduled_ops)
            {
                if (op_names_set.find(scheduled_op) != op_names_set.end())
                {
                    ops_tagged_for_epoch_break.insert(scheduled_op);
                    break;
                }
            }
        }
    }

    // Add epoch breaks between subgraphs.
    // Add epoch breaks to split data parallel NOPs from other regular OPs.
    //
    unsigned int prev_subgraph_id = graph->get_subgraph_id_for_node(graph->get_node_by_name(scheduled_ops[0])->id());
    bool in_data_parallel_nop_group = false;
    for (const string& op : scheduled_ops)
    {
        Node* node = graph->get_node_by_name(op);
        unsigned int subgraph_id = graph->get_subgraph_id_for_node(node->id());
        TT_ASSERT(subgraph_id >= prev_subgraph_id);
        if (subgraph_id != prev_subgraph_id)
        {
            ops_tagged_for_epoch_break.insert(op);
            log_debug(LogPlacer, "Epoch break due to subgraph at: {}", op);
            prev_subgraph_id = subgraph_id;
        }

        if (node->node_type() == NodeType::kBudaOp)
        {
            BudaOpNode* buda_node = static_cast<BudaOpNode*>(node);

            if (buda_node->is_data_parallel_nop())
            {
                if (!in_data_parallel_nop_group)
                {
                    // Start of data parallel NOP group. Add an epoch break.
                    //
                    in_data_parallel_nop_group = true;
                    ops_tagged_for_epoch_break.insert(op);
                }
            }
            else if (in_data_parallel_nop_group)
            {
                // End of data parallel NOP group, add an epoch break.
                //
                in_data_parallel_nop_group = false;
                ops_tagged_for_epoch_break.insert(op);
            }
        }
    }
    return ops_tagged_for_epoch_break;
}

unordered_set<string> tag_ops_for_epoch_break(
    const DeviceConfig& device_config,
    const vector<vector<string>>& op_names_to_epoch_break,
    const vector<vector<string>>& op_names_to_chip_break,
    const vector<string>& scheduled_ops,
    graphlib::Graph const* graph,
    bool use_interactive_placer)
{
    if (env_as<bool>("PYBUDA_SINGLE_OP_EPOCHS"))
    {
        unordered_set<string> ops_tagged_for_epoch_break;
        for (const auto& op_name : scheduled_ops)
        {
            ops_tagged_for_epoch_break.insert(op_name);
        }
        return ops_tagged_for_epoch_break;
    }
    if ((use_interactive_placer == false || env_as<bool>("PYBUDA_WORMHOLE_PIPELINED_PLACER")) && device_config.is_wormhole_b0())
    {
        vector<vector<string>> updated_op_names_to_epoch_break = op_names_to_epoch_break;
        updated_op_names_to_epoch_break.insert(
            updated_op_names_to_epoch_break.end(),
            op_names_to_chip_break.begin(), op_names_to_chip_break.end());

        if (env_as<bool>("PYBUDA_NEBULA_GALAXY_PLACER"))
        {
            for (const auto& output_op : get_output_nodes(graph))
            {
                updated_op_names_to_epoch_break.insert(updated_op_names_to_epoch_break.end(), {output_op});
            }
        }
        return tag_ops_for_epoch_or_chip_break(updated_op_names_to_epoch_break, scheduled_ops, graph, true);
    }
    return tag_ops_for_epoch_or_chip_break(op_names_to_epoch_break, scheduled_ops, graph, true);
}

unordered_set<string> tag_ops_for_chip_break(
    const DeviceConfig& device_config,
    const vector<vector<string>>& op_names_to_chip_break,
    const vector<string>& scheduled_ops,
    graphlib::Graph const* graph,
    bool use_interactive_placer)
{
    if (use_interactive_placer == false && device_config.is_wormhole_b0())
    {
        return {};
    }
    return tag_ops_for_epoch_or_chip_break(op_names_to_chip_break, scheduled_ops, graph, false);
}

// only used by legacy placer, with interactive_placer epoch_break will act as a temporal_epoch break
unordered_set<string> tag_ops_for_temporal_epoch_break(
    graphlib::Graph const* graph,
    const vector<string>& scheduled_op_names,
    const std::unordered_map<std::string, placer::PlacerOpOverride>& op_name_to_placer_overrides)
{
    unordered_set<string> ops_tagged_for_temporal_epoch_break;
    unordered_map<std::string, std::uint32_t> op_to_schedule_index;
    unordered_set<std::uint32_t> visited_fracture_ids;

    for (std::uint32_t i = 0; i < scheduled_op_names.size(); ++i)
    {
        op_to_schedule_index[scheduled_op_names[i]] = i;
        graphlib::Node* node = graph->get_node_by_name(scheduled_op_names[i]);
        if (node->as<graphlib::TaggedNode>()->has_tag("fracture_group_id"))
        {
            graphlib::TagValue tag_value = node->as<graphlib::TaggedNode>()->tag_value("fracture_group_id");
            std::uint32_t fracture_group_id = std::get<std::uint32_t>(tag_value);
            if (visited_fracture_ids.find(fracture_group_id) == visited_fracture_ids.end())
            {
                ops_tagged_for_temporal_epoch_break.insert(scheduled_op_names[i]);
                visited_fracture_ids.insert(fracture_group_id);
            }
        }
    }
    for (const auto& op_name_to_placer_override : op_name_to_placer_overrides)
    {
        const auto& [op_name, placer_op_override] = op_name_to_placer_override;
        if (placer_op_override.temporal_epoch_break)
        {
            ops_tagged_for_temporal_epoch_break.insert(op_name);
        }
    }

    if (not ops_tagged_for_temporal_epoch_break.empty())
    {
        log_debug(LogPlacer, "ops_tagged_for_temporal_epoch_break: {}", ops_tagged_for_temporal_epoch_break);
    }
    return ops_tagged_for_temporal_epoch_break;
}


} // namespace lowering
} // namespace placer
} // namespace tt
