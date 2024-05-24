// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "backend_api/device_config.hpp"
#include "balancer/legalizer/graph_solver_types.hpp"
#include "balancer/types.hpp"
#include "placer/chip_id_assignment.hpp"
#include "policies/policy_types.hpp"
#include "scheduler/scheduler.hpp"

namespace tt::balancer
{

struct OpOverride
{
    std::optional<std::pair<int, int>> grid_shape;
    std::optional<bool> force_dram_parameters;
    std::string t_stream_dir = "";
    std::optional<std::pair<int, int>> t_stream_shape;
    std::optional<int> u_kt;
    std::optional<std::map<uint32_t, std::uint32_t>> input_buffer_multiplier;
    std::optional<int> output_buffer_multiplier;

    void apply(
        FactorizedShape& grid_pars,
        bool& force_dram_parameters_out,
        std::vector<TStreamDir>& t_stream_dirs,
        FactorizedShape& overridden_streaming_pars,
        bool& enable_t_streaming,
        const std::string& op_name);

    std::optional<int> get_u_kt();
};

struct BalancerConfig
{
    DeviceConfig device_config;
    scheduler::SchedulerConfig scheduler_config;
    PolicyType policy_type;
    int random_policy_seed;
    std::vector<std::uint32_t> chip_ids;
    placer::ChipPlacementPolicy chip_placement_policy;
    bool default_dram_parameters;
    bool skip_l1_usage_validation;
    bool enable_t_streaming;
    bool manual_t_streaming;
    bool input_queues_on_host;
    bool output_queues_on_host;
    std::unordered_map<std::string, OpOverride> op_overrides;
    std::vector<std::vector<std::string>> op_names_to_epoch_break;
    std::vector<std::vector<std::string>> op_names_to_chip_break;
    placer::OpToChipIdAssignment op_to_chip_id_assignment;
    std::unordered_map<std::string, placer::PlacerOpOverride> op_name_to_placer_overrides;
    bool enable_auto_transposing_placement;
    legalizer::GraphSolverSelfCutType graph_solver_self_cut_type;
    bool use_interactive_placer;
    bool enable_enumerate_u_kt;
    bool enable_single_buffer_fallback;
    std::uint32_t target_cycles_offset = 0;

    BalancerConfig(
        const DeviceConfig& device_config,
        scheduler::SchedulerConfig scheduler_config,
        PolicyType policy_type,
        int random_policy_seed,
        const std::vector<std::uint32_t>& chip_ids,
        placer::ChipPlacementPolicy chip_placement_policy,
        bool default_dram_parameters,
        bool skip_l1_usage_validation,
        bool enable_t_streaming,
        bool manual_t_streaming,
        bool input_queues_on_host,
        bool output_queues_on_host,
        std::unordered_map<std::string, OpOverride> op_overrides,
        const std::vector<std::vector<std::string>>& op_names_to_epoch_break,
        const std::vector<std::vector<std::string>>& op_names_to_chip_break,
        const placer::OpToChipIdAssignment& op_to_chip_id_assignment,
        const std::unordered_map<std::string, placer::PlacerOpOverride>& op_name_to_placer_overrides,
        bool enable_auto_transposing_placement,
        legalizer::GraphSolverSelfCutType graph_solver_self_cut_type,
        bool use_interactive_placer,
        bool enable_enumerate_u_kt,
        bool enable_single_buffer_fallback) :
        device_config(device_config),
        scheduler_config(scheduler_config),
        policy_type(policy_type),
        random_policy_seed(random_policy_seed),
        chip_ids(chip_ids),
        chip_placement_policy(chip_placement_policy),
        default_dram_parameters(default_dram_parameters),
        skip_l1_usage_validation(skip_l1_usage_validation),
        enable_t_streaming(enable_t_streaming),
        manual_t_streaming(manual_t_streaming),
        input_queues_on_host(input_queues_on_host),
        output_queues_on_host(output_queues_on_host),
        op_overrides(op_overrides),
        op_names_to_epoch_break(op_names_to_epoch_break),
        op_names_to_chip_break(op_names_to_chip_break),
        op_to_chip_id_assignment(op_to_chip_id_assignment),
        op_name_to_placer_overrides(op_name_to_placer_overrides),
        enable_auto_transposing_placement(enable_auto_transposing_placement),
        graph_solver_self_cut_type(graph_solver_self_cut_type),
        use_interactive_placer(use_interactive_placer),
        enable_enumerate_u_kt(enable_enumerate_u_kt),
        enable_single_buffer_fallback(enable_single_buffer_fallback)
    {
    }

    // Constructor - used only by unittesting.
    //
    BalancerConfig(
        const DeviceConfig& device_config,
        PolicyType policy_type,
        placer::ChipPlacementPolicy chip_placement_policy = placer::ChipPlacementPolicy::MMIO_LAST) :
        device_config(device_config),
        policy_type(policy_type),
        chip_ids(device_config.chip_ids),
        chip_placement_policy(chip_placement_policy),
        graph_solver_self_cut_type(legalizer::GraphSolverSelfCutType::None)
    {
        // If unit tests specify policy which use IP, mark it as used.
        //
        use_interactive_placer = can_use_interactive_placer(policy_type);
    }

    inline std::optional<OpOverride> get_op_override(std::string const& op_name) const
    {
        auto op_override = op_overrides.find(op_name);
        if (op_override == op_overrides.end())
            return {};
        return op_override->second;
    }

    int get_total_cores() const
    {
        return device_config.grid_size.size();
    }
};

}  // namespace tt::balancer
