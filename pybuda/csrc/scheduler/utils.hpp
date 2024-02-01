// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Common utilities for schedulers
//
#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "graph_lib/defines.hpp"

namespace tt
{

namespace graphlib
{
class Graph;
class Node;
}

namespace scheduler
{

using Schedule = std::vector<std::string>;

void log_schedule(const Schedule& schedule);
bool can_schedule_node(const graphlib::Node* node);
const std::vector<const graphlib::Node*> get_schedule_predecessors(
    const graphlib::Graph* graph, const graphlib::Node* node);
const std::vector<const graphlib::Node*> get_schedule_successors(
    const graphlib::Graph* graph, const graphlib::Node* node);

std::unordered_map<std::string, int> get_op_to_schedule_index(const Schedule& scheduled_ops);
Schedule get_filtered_schedule(const graphlib::Graph* graph, const Schedule& schedule, graphlib::NodeEpochType type);
bool are_schedule_dependencies_met(const graphlib::Graph* graph, const Schedule& schedule);

}  // namespace scheduler

}  // namespace tt
