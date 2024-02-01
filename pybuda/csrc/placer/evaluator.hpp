// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>

#include "balancer/balancer.hpp"
#include "scheduler/interactive_scheduler.hpp"

namespace tt
{

namespace placer
{
// Copy of a valid placement solution, along with meta-data that allows selection between them, and
// restorating of the state.
class EpochPlacement
{
   private:
    std::shared_ptr<balancer::BalancerSolution> solution_;
    float score_;

    // state
    scheduler::InteractiveScheduler::Checkpoint scheduler_checkpoint_;
    std::unique_ptr<graphlib::Graph> graph_;

    float calculate_score(std::string const& arch_name);

   public:
    EpochPlacement() : solution_(nullptr), score_(0.0), graph_(nullptr) {}
    EpochPlacement(
        std::shared_ptr<balancer::BalancerSolution> solution,
        scheduler::InteractiveScheduler::Checkpoint scheduler_checkpoint,
        std::unique_ptr<graphlib::Graph> graph,
        std::string const& arch_name);

    bool valid() const { return solution_ != nullptr; }
    float score() const { return score_; }
    const scheduler::InteractiveScheduler::Checkpoint &scheduler_checkpoint() const { return scheduler_checkpoint_; }
    std::shared_ptr<balancer::BalancerSolution> solution() const { return solution_; }
    Graph *release_graph()
    {
        if (graph_ == nullptr)
            return nullptr;
        return graph_.release();
    }
    bool is_better_than(const EpochPlacement &other) const;
};

}  // namespace placer
}  // namespace tt
