// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scheduler/utils.hpp"

#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "utils/logger.hpp"

using tt::LogScheduler;

namespace tt::scheduler
{

void log_schedule(const Schedule& schedule)
{
    for (std::uint32_t i = 0; i < schedule.size(); ++i)
    {
        log_debug(LogScheduler, "schedule index: {}, op: {}", i, schedule[i]);
    }
}

bool can_schedule_node(const graphlib::Node* node)
{
    return node->node_type() != graphlib::NodeType::kInput and node->node_type() != graphlib::NodeType::kOutput and
           node->node_type() != graphlib::NodeType::kQueue and node->node_type() != graphlib::NodeType::kBudaNaryTM;
}

// Returns operands of the node, skipping through queue nodes.
//
const std::vector<const graphlib::Node*> get_schedule_predecessors(
    const graphlib::Graph* graph, const graphlib::Node* node)
{
    std::vector<const graphlib::Node*> predecessors;
    for (const graphlib::Node* operand_node : graph->data_operands(node))
    {
        // Skip through queue nodes.
        //
        if (operand_node->node_type() == graphlib::NodeType::kQueue)
        {
            const graphlib::QueueNode* queue_node = operand_node->as<graphlib::QueueNode>();
            if (queue_node->queue_type() == graphlib::QueueNodeType::Buffering or
                queue_node->queue_type() == graphlib::QueueNodeType::EpochToEpoch)
            {
                predecessors.push_back(graph->data_operands(queue_node)[0]);
            }
        }
        else
        {
            predecessors.push_back(operand_node);
        }
    }

    return predecessors;
}

// Returns users of the node, skipping(expanding) through queue nodes.
//
const std::vector<const graphlib::Node*> get_schedule_successors(
    const graphlib::Graph* graph, const graphlib::Node* node)
{
    std::vector<const graphlib::Node*> successors;
    for (const graphlib::Node* user_node : graph->data_users(node))
    {
        // Skip through queue nodes.
        //
        if (user_node->node_type() == graphlib::NodeType::kQueue)
        {
            const graphlib::QueueNode* queue_node = user_node->as<graphlib::QueueNode>();
            if (queue_node->queue_type() == graphlib::QueueNodeType::Buffering or
                queue_node->queue_type() == graphlib::QueueNodeType::EpochToEpoch)
            {
                std::vector<tt::graphlib::Node*> data_users = graph->data_users(queue_node);
                successors.insert(successors.end(), data_users.begin(), data_users.end());
            }
        }
        else
        {
            successors.push_back(user_node);
        }
    }

    return successors;
}

std::unordered_map<std::string, int> get_op_to_schedule_index(const Schedule& scheduled_ops)
{
    std::unordered_map<std::string, int> op_to_schedule_index;
    op_to_schedule_index.reserve(scheduled_ops.size());
    for (int i = 0; i < (int)scheduled_ops.size(); ++i)
    {
        const std::string& node_name = scheduled_ops.at(i);
        op_to_schedule_index[node_name] = i;
    }
    return op_to_schedule_index;
}

Schedule get_filtered_schedule(const graphlib::Graph* graph, const Schedule& schedule, graphlib::NodeEpochType type)
{
    Schedule filtered_schedule;
    filtered_schedule.reserve(schedule.size());
    for (unsigned int subgraph_index = 0; subgraph_index < graph->num_subgraphs(); subgraph_index++)
    {
        for (const auto& node_name : schedule)
        {
            graphlib::Node* node = graph->get_node_by_name(node_name);
            if (node->get_epoch_type() == type)
            {
                if (graph->get_subgraph_id_for_node(node->id()) != subgraph_index)
                {
                    continue;
                }
                filtered_schedule.push_back(node_name);
            }
        }
    }
    return filtered_schedule;
}

bool are_schedule_dependencies_met(const graphlib::Graph* graph, const std::vector<std::string>& schedule)
{
    std::unordered_map<std::string, std::uint32_t> node_to_schedule_index;
    node_to_schedule_index.reserve(schedule.size());
    for (std::uint32_t i = 0; i < schedule.size(); ++i)
    {
        node_to_schedule_index[schedule[i]] = i;
    }

    for (const std::string& op : schedule)
    {
        graphlib::Node* node = graph->get_node_by_name(op);
        for (const graphlib::Edge& operand_edge : graph->operand_data_edges(node))
        {
            graphlib::Node* predecessor_node = graph->node_by_id(operand_edge.producer_node_id);
            if (node_to_schedule_index.find(predecessor_node->name()) != node_to_schedule_index.end())
            {
                if (node_to_schedule_index[predecessor_node->name()] > node_to_schedule_index[op])
                {
                    log_warning(
                        LogPlacer,
                        "Scheduler: dependency not met for node: {}: {} should come before",
                        op,
                        predecessor_node->name());
                    return false;
                }
            }
        }
    }
    return true;
}

}  // namespace tt::scheduler
