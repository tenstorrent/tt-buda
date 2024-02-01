// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "placer/evaluator.hpp"

namespace tt::placer
{

EpochPlacement::EpochPlacement(
    std::shared_ptr<balancer::BalancerSolution> solution,
    scheduler::InteractiveScheduler::Checkpoint scheduler_checkpoint,
        std::unique_ptr<graphlib::Graph> graph,
        std::string const& arch_name) :
    solution_(solution), scheduler_checkpoint_(scheduler_checkpoint), graph_(std::move(graph))
{
    score_ = calculate_score(arch_name);
}

bool EpochPlacement::is_better_than(const EpochPlacement &other) const
{
    if (solution_ == nullptr)
        return other.solution_ == nullptr;

    return score_ > other.score_;
}

float f1_score(float a, float b) { return 2 * a * b / (a + b); }

float EpochPlacement::calculate_score(std::string const& arch_name)
{
    // Figure out how well we've balanced the ops
    if (solution_ == nullptr)
        return 0.0;

    // get_execution_cycles() is slow, so we'll only call it once
    // TODO: we need some kind of a global cache, since we're still going to call calculate_scores
    // many times
    std::unordered_map<std::string, std::uint32_t> execution_cycles;
    std::unordered_map<std::string, std::uint32_t> theoretical_cycles;

    const auto &placements = solution_->placer_solution.name_to_op_placement;
    for (auto it : placements)
    {
        const std::string &name = it.first;
        const auto &op_model = solution_->op_models.at(name);
        execution_cycles.insert(std::make_pair(name, op_model.get_execution_cycles(arch_name)));
        if (op_model.op_type() == "matmul")
            theoretical_cycles.insert(std::make_pair(name, op_model.get_execution_cycles(arch_name, true)));
    }

    std::uint32_t slowest_core = 0;
    for (auto it : placements)
    {
        const std::uint32_t cycles = execution_cycles.at(it.first);
        if (cycles > slowest_core)
            slowest_core = cycles;
    }

    // One goal is to have matmuls run as efficiently as possible.
    // Another goal is to keep as many cores as busy as we can.

    float matmul_utilization = 0.0;    // indicator of how close to theoretical matmuls are
    float balancer_utilization = 0.0;  // indicator of how busy all cores are
    std::uint32_t matmul_core_count = 0;
    for (auto it : placements)
    {
        const std::string &name = it.first;
        const auto &op_model = solution_->op_models.at(name);
        balancer_utilization += (1.0 * execution_cycles.at(name) / slowest_core) * op_model.grid_shape.volume();
        if (op_model.op_type() == "matmul")
        {
            matmul_utilization += (1.0 * theoretical_cycles.at(name) / slowest_core) * op_model.grid_shape.volume();
            matmul_core_count += op_model.grid_shape.volume();
        }
    }

    std::uint32_t core_count = solution_->placer_solution.epoch_id_to_device_grid.rows *
                               solution_->placer_solution.epoch_id_to_device_grid.columns;

    balancer_utilization /= core_count;
    if (matmul_core_count > 0)
    {
        matmul_utilization /= matmul_core_count;
    }

    if (matmul_core_count > 0)
        return f1_score(balancer_utilization, matmul_utilization);

    return balancer_utilization;
}

}  // namespace tt::placer
