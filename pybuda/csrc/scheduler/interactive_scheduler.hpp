// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <vector>
#include <unordered_set>

#include "scheduler/scheduler.hpp"
#include "scheduler/utils.hpp"
#include "graph_lib/defines.hpp"

namespace tt
{
namespace graphlib
{
class Graph;
class BudaOpNode;
}
namespace scheduler
{
// Interactive scheduler returns a list of ops that could be scheduled next, ordered by preference
// of the selected scheduling algorithm.
class InteractiveScheduler
{
   private:
    const graphlib::Graph *graph;
    graphlib::NodeEpochType current_epoch_type;

    Schedule preferred_schedule;
    std::vector<std::string> ready_ops; // ops ready to be executed
    std::unordered_set<graphlib::Node *> scheduled_ops; // ops that have been accepted

    bool op_is_ready(const graphlib::BudaOpNode *op) const;
    void try_schedule_users(const graphlib::Node *node);

   public:
    InteractiveScheduler(const SchedulerConfig &config, const graphlib::Graph *graph, graphlib::NodeEpochType initial_epoch_type);

    std::vector<std::string> get_ops() const;
    void accept_op(const std::string &op);
    bool done() const;

    void set_epoch_type(graphlib::NodeEpochType epoch_type);
    
    struct Checkpoint {
        Schedule preferred_schedule;
        std::vector<std::string> ready_ops; // ops ready to be executed
        std::unordered_set<graphlib::Node *> scheduled_ops; // ops that have been accepted
        graphlib::NodeEpochType current_epoch_type;
    };

    Checkpoint save_checkpoint() const;
    void restore_checkpoint(const Checkpoint &checkpoint);
};

}  // namespace scheduler
}  // namespace tt
