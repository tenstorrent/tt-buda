// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "scheduler/interactive_scheduler.hpp"

#include "graph_lib/graph.hpp"
#include "graph_lib/node_types.hpp"
#include "scheduler/longest_path.hpp"
#include "utils/logger.hpp"

namespace tt::scheduler
{

bool InteractiveScheduler::op_is_ready(const graphlib::BudaOpNode *op) const
{
    if (op->get_epoch_type() != current_epoch_type)
        return false;

    for (graphlib::Node *node : graph->operands(op))
    {
        if (node->node_type() == graphlib::kBudaOp)
        {
            if (scheduled_ops.count(node->as<graphlib::BudaOpNode>()) == 0)
                return false;
        }
    }

    return true;
}

void InteractiveScheduler::try_schedule_users(const graphlib::Node *node)
{
    for (graphlib::Node *node : graph->users(node))
        if (node->node_type() == graphlib::kBudaOp) {
            auto it = std::find(ready_ops.begin(), ready_ops.end(), node->name());
            if ((it == ready_ops.end()) && op_is_ready(node->as<graphlib::BudaOpNode>()))
                ready_ops.push_back(node->name());
        }
}

InteractiveScheduler::InteractiveScheduler(
    const SchedulerConfig &config, const graphlib::Graph *graph, graphlib::NodeEpochType initial_epoch_type) :
    graph(graph), current_epoch_type(initial_epoch_type)
{
    // Generate preferred schedule. Scheduler will prioritize the next op in the scheduler when offering
    // choices.
    switch (config.policy)
    {
        case SchedulerPolicy::Topological: preferred_schedule = run_topological_scheduler(graph); break;
        case SchedulerPolicy::ModuleInputsBFS: preferred_schedule = run_module_by_module_scheduler(config, graph); break;
        case SchedulerPolicy::LongestPath: preferred_schedule = run_longest_path_scheduler(graph); break;
        default: log_fatal("providing unknown scheduler policy.");
    }

    if (preferred_schedule.size() == 0)
        return;

    // Create initial set of op candidates
    ready_ops.push_back(preferred_schedule.at(0));

    for (graphlib::Node *node : graph->nodes())
    {
        if (node->node_type() == graphlib::NodeType::kInput)
        {
            try_schedule_users(node);
        }
    }
}

std::vector<std::string> InteractiveScheduler::get_ops() const { return ready_ops; }

void InteractiveScheduler::accept_op(const std::string &op)
{
    // Mark op as scheduled, and update the 'ready_ops' list
    scheduled_ops.insert(graph->get_node_by_name(op));

    auto it = std::find(ready_ops.begin(), ready_ops.end(), op);
    TT_ASSERT(it != ready_ops.end());
    ready_ops.erase(it);

    try_schedule_users(graph->get_node_by_name(op));

    it = std::find(preferred_schedule.begin(), preferred_schedule.end(), op);
    if (it != preferred_schedule.end())
        preferred_schedule.erase(it);

    if (preferred_schedule.size() > 0)
    {
        it = std::find(ready_ops.begin(), ready_ops.end(), preferred_schedule.at(0));
        if (it != ready_ops.begin() && it != ready_ops.end()) // found it, and not at the top spot - move it
        {
            ready_ops.erase(it);
            ready_ops.insert(ready_ops.begin(), preferred_schedule.at(0));
        }
    }
}

bool InteractiveScheduler::done() const { return ready_ops.size() == 0; }
    
void InteractiveScheduler::set_epoch_type(graphlib::NodeEpochType epoch_type) 
{ 
    TT_ASSERT(done(), "Epoch type shouldn't be changed on the fly. At least not with current implementation");
    current_epoch_type = epoch_type; 

    for (const std::string &op : preferred_schedule)
    {
        if (op_is_ready(graph->get_node_by_name(op)->as<graphlib::BudaOpNode>()))
        {
            ready_ops.push_back(op);
        }
    }
}

InteractiveScheduler::Checkpoint InteractiveScheduler::save_checkpoint() const
{
    return Checkpoint{preferred_schedule, ready_ops, scheduled_ops, current_epoch_type};
}

void InteractiveScheduler::restore_checkpoint(const InteractiveScheduler::Checkpoint &checkpoint)
{
    preferred_schedule = checkpoint.preferred_schedule;
    ready_ops = checkpoint.ready_ops;
    scheduled_ops = checkpoint.scheduled_ops;
    current_epoch_type = checkpoint.current_epoch_type;
}

}  // namespace tt::scheduler
