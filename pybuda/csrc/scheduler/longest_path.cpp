// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "scheduler/longest_path.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/utils.hpp"

#include "utils/logger.hpp"

using tt::graphlib::Graph;
using tt::graphlib::Node;
using tt::LogScheduler;

// 
// Give longest path throgh the graph the priority - schedule along it whenever possible, along with any additional outputs
// that are created along that path. The goal is to have outputs "sit" for the shortest amount of time.
//
// To achieve the goal:
//   - Schedule the next op in the longest path when possible.
//   - When an op on longest path can't be scheduled, schedule its operands as quickly as possible.
//   - Don't go deep down paths when other ops have their outputs sitting around.
//
// We'll do this by having two priority groups:
//
//   - Priority 1 - only schedule ops from this group. The next in the longest path gets priority.
//   - Priority 2 - when priority 1 group is empty, move priority 2 to priority 1
//
// Priority 1 starts with the first op of the longest path, priority 2 starts empty.
// Any time a new op is scheduled, its users are added to priority 2, if they aren't in priority 1 or have been scheduled already.
// If no op in P1 can be scheduled, then loop:
//   - all operands of P1 ops are added to P1, unless they have been scheduled already. If they are in P2, they are removed from there.
// ..until something from P1 can be scheduled.
namespace tt::scheduler {

Schedule run_longest_path_scheduler(const Graph* graph)
{

    auto less = [](Node *a, Node *b) { return a->id() < b->id(); };
    using SortedNodeSet = std::set<Node *, std::function<bool(Node *, Node*)>>; // using a sorted set to preserve determinism

    std::vector<Node *> longest_path = graphlib::get_longest_path(graph);
    SortedNodeSet P1(less);
    SortedNodeSet P2(less);
    P1.insert(longest_path[1]); // skipping the first node since it's an input.. it's already "scheduled"
    std::uint32_t current_longest_path_index = 1;

    // Set of ops currently scheduled for quick lookup
    SortedNodeSet scheduled(less);
    SortedNodeSet input_nodes(less);
    for (Node *node : graph->nodes())
        if (node->node_type() == graphlib::NodeType::kInput) {
            input_nodes.insert(node);
            scheduled.insert(node); // all inputs are "scheduled", so that ops that only depend on inputs can see their operands ready
        }

    // Actual schedule
    std::vector<Node *> schedule;

    // Check if op can be scheduled - i.e. all of its operands have been scheduled
    auto can_be_scheduled = [&scheduled, &graph](Node *node) -> bool { 
        std::vector<Node *> operands = graph->data_operands(node);
        TT_ASSERT(operands.size() > 0, "Input " + node->name() + " should've already been scheduled"); // all inputs should already be scheduled
        return std::all_of(operands.begin(), operands.end(), [&scheduled](Node *operand) { return scheduled.count(operand) > 0; });
    };

    // Schedule op - add to schedule, remove from P1, add outputs to P2
    auto schedule_op = [&scheduled, &P1, &P2, &can_be_scheduled, &schedule, &graph](Node *node) {
        TT_ASSERT(can_be_scheduled(node));
        TT_ASSERT(P1.count(node) > 0 && P2.count(node) == 0 && scheduled.count(node) == 0); 
        schedule.push_back(node);
        scheduled.insert(node);
        P1.erase(node);

        for (Node *user : graph->data_users(node))
        {
            if (P1.count(user) == 0 && scheduled.count(user) == 0)
                P2.insert(user);
        }

    };


    // Keep scheduling until nothing's left
    while (!P1.empty() || !P2.empty())
    {
        if (P1.empty()) {
            P1 = P2;
            P2.clear();
        }

        // First schedule longest path, if possible
        Node *next_on_longest_path = longest_path[current_longest_path_index];
        while (P1.count(next_on_longest_path) > 0 && can_be_scheduled(next_on_longest_path)) {
            schedule_op(next_on_longest_path);
            if (current_longest_path_index < longest_path.size() - 1)
                current_longest_path_index++;
            next_on_longest_path = longest_path[current_longest_path_index];
        }

        // Now schedule everything else in P1, if possible
        SortedNodeSet P1_copy(less);
        P1_copy = P1; // make a copy since we'll be removing items
        for (Node *node: P1_copy)
        {
            if (can_be_scheduled(node))
                schedule_op(node);
        }

        // If there's anything left in P1, it means it couldn't be scheduled, so we'll try to schedule its operands
        P1_copy = P1;
        for (Node *node: P1_copy)
        {
            for (Node *operand: graph->data_operands(node))
            {
                if (P1.count(operand) == 0 && scheduled.count(operand) == 0) {
                    if (P2.count(operand) > 0)
                        P2.erase(operand); // upgrade to P1
                    P1.insert(operand);
                }
            }
        }

        if (P1.empty() && P2.empty()) {
            // Check if any of the inputs have their users not scheduled - add them to P1 to get them queued up.
            for (Node *input_node : input_nodes)
                for (Node *user: graph->data_users(input_node))
                    if (scheduled.count(user) == 0)
                        P1.insert(user);
        }
    }

    // Debug
    //std::cout << "Schedule:" << std::endl;
    //for (Node *node: schedule)
    //    std::cout << " - " << node->name() << std::endl;

    // Verify
    auto visible_nodes = graphlib::visible_nodes(*graph);
    if (scheduled.size() != visible_nodes.size())
    {
        for (Node *node : visible_nodes)
            if (scheduled.count(node) == 0)
            {
                log_error(tt::LogScheduler, "{} hasn't been scheduled.", node->name());
            }
        TT_THROW("Some nodes haven't been scheduled");
    }

    std::unordered_set<Node *> visited;
    for (Node *node: schedule) {
        for (Node *operand: graph->data_operands(node))
            TT_ASSERT(visited.count(operand) > 0 || input_nodes.count(operand) > 0,
                    "Operand " + operand->name() + " of node " + node->name() + " hasn't been scheduled before the node.");
        visited.insert(node);
    }

    // Remove all unscheduleable nodes
    std::vector<Node *> final_schedule;
    for (Node *node : schedule)
    {
        if (can_schedule_node(node))
        {
            final_schedule.push_back(node);
        }
    }

    Schedule ret;
    for (Node *node : final_schedule) ret.push_back(node->name());
    return ret;
}

}

