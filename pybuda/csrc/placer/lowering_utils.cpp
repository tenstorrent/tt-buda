// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placer/lowering_utils.hpp"

#include <math.h>

#include <unordered_set>
#include <utility>

#include "placer/chip_id_assignment.hpp"
#include "scheduler/scheduler.hpp"
#include "utils/assert.hpp"
#include "utils/logger.hpp"
#include "utils/ordered_associative_containers/ordered_map.hpp"

using tt::LogPlacer;
using std::unordered_set;

namespace tt {
namespace placer {
namespace lowering {

void validate_placer_config(const PlacerConfig& placer_config)
{
    bool is_config_valid = true;
    std::unordered_set<std::string> bwd_nodes_set;

    for (auto& [fwd, bwd_nodes] : placer_config.fwd_to_bwd_nodes)
    {
        for (auto bwd_node : bwd_nodes)
        {
            bwd_nodes_set.insert(bwd_node);
        }
    }
    for (auto& [fwd, index_to_opt_nodes_map] : placer_config.fwd_to_opt_nodes) {
        for (auto& [index, opt_nodes] : index_to_opt_nodes_map) {
            for (auto opt_node : opt_nodes) {
                if (bwd_nodes_set.find(opt_node) != bwd_nodes_set.end()) {
                    is_config_valid = false;
                    log_error("Invalid PlacerConfig: Found {} having both fwd->bwd AND fwd->opt edges", opt_node);
                }
            }
        }
    }
    if (not is_config_valid) {
        log_fatal("Invalid PlacerConfig. Cannot run placer module");
    }
}

std::map<ChipId, std::uint32_t> get_galaxy_snake_chip_order(const DeviceConfig& config)
{
    TT_ASSERT(config.galaxy_shelves.size() == 1, "SNAKE chip config is only supported for single-galaxy systems");
    // x-y galaxy chip coordinates for snake pattern
    std::vector<std::pair<int, int>> galaxy_snake_chip_order_in_logical_coordinates =
    {
        {3, 4}, {3, 3}, {3, 2}, {3, 1}, {3, 0}, {2, 0}, {1, 0}, {0, 0},
        {0, 1}, {1, 1}, {2, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 3}, {1, 3},
        {2, 3}, {2, 4}, {1, 4}, {0, 4}, {0, 5}, {1, 5}, {2, 5}, {2, 6},
        {1, 6}, {0, 6}, {0, 7}, {1, 7}, {2, 7}, {3, 7}, {3, 6}, {3, 5},
    };
    std::map<ChipId, std::uint32_t> chip_id_indices;
    std::uint32_t index = 0;
    for(auto& xy: galaxy_snake_chip_order_in_logical_coordinates)
    {
        int x = xy.first;
        int y = xy.second;
        chip_id_indices[config.chip_coord_to_chip_id.at(EthCoord(x, y, 0, config.galaxy_shelves.front()))] = index++;
    }
    return chip_id_indices;
}

vector<std::uint32_t> apply_chip_placement_policy(
    const DeviceConfig& config,
    ChipPlacementPolicy chip_placement_policy,
    const vector<ChipId>& chip_ids)
{
    std::vector<ChipId> sorted_chip_ids;

    // use given chip ids, sort non_mmio + mmio
    if(chip_placement_policy == ChipPlacementPolicy::MMIO_LAST)
    {
        for (ChipId chip_id : chip_ids)
        {
            if (std::find(config.chips_with_mmio.begin(), config.chips_with_mmio.end(), chip_id) == config.chips_with_mmio.end())
            {
                sorted_chip_ids.push_back(chip_id);
            }
        }
        for (std::uint32_t chip_id : chip_ids)
        {
            if (std::find(config.chips_with_mmio.begin(), config.chips_with_mmio.end(), chip_id) != config.chips_with_mmio.end())
            {
                sorted_chip_ids.push_back(chip_id);
            }
        }
        return sorted_chip_ids;
    }

    // get chip id order based on the ChipPlacementPolicy
    std::map<ChipId, std::uint32_t> galaxy_chip_id_indices =
        chip_placement_policy == ChipPlacementPolicy::SNAKE ? get_galaxy_snake_chip_order(config) :
        // add new policies here
        std::map<ChipId, std::uint32_t>();
    TT_ASSERT(galaxy_chip_id_indices.size());

    // split all available chip_ids into galaxy_chip_ids and non_galaxy_chip_ids
    std::vector<ChipId> galaxy_chip_ids;
    std::vector<ChipId> non_galaxy_chip_ids;
    for(auto& chip_id: chip_ids)
    {
        if(galaxy_chip_id_indices.find(chip_id) != galaxy_chip_id_indices.end())
        {
            galaxy_chip_ids.push_back(chip_id);
        }
        else {
            non_galaxy_chip_ids.push_back(chip_id);
        }
    }

    // order galaxy_chip_ids based on their order in the ChipPlacementPolicy
    std::sort(galaxy_chip_ids.begin(), galaxy_chip_ids.end(), [&galaxy_chip_id_indices](ChipId chip_a, ChipId chip_b)
    {
        return galaxy_chip_id_indices.at(chip_a) < galaxy_chip_id_indices.at(chip_b);
    });

    sorted_chip_ids = galaxy_chip_ids;
    sorted_chip_ids.insert(sorted_chip_ids.end(), non_galaxy_chip_ids.begin(), non_galaxy_chip_ids.end());

    return sorted_chip_ids;
}

unordered_map<string, GridShape> get_op_to_grid_shape(
    const vector<string>& scheduled_ops, uint32_t default_rows, uint32_t default_columns)
{
    unordered_map<string, GridShape> op_to_grid_shape;
    for (const string& node_name : scheduled_ops)
    {
        op_to_grid_shape[node_name] = GridShape(default_rows, default_columns);
    }
    return op_to_grid_shape;
}

void check_user_defined_op_names_exist_in_schedule(
    const PlacerConfig& config,
    const vector<string>& scheduled_ops)
{
    if (config.ops_tagged_for_chip_id_break.empty() and config.ops_tagged_for_epoch_break.empty())
    {
        return;
    }

    unordered_set<string> scheduled_ops_set;
    for (const string& name : scheduled_ops)
    {
        scheduled_ops_set.insert(name);
    }

    // Check all user-defined op_to_chip_id breaks exists
    for (const string& name_tagged_for_chip_id_break : config.ops_tagged_for_chip_id_break)
    {
        bool is_op_found_schedule =
            scheduled_ops_set.find(name_tagged_for_chip_id_break) != scheduled_ops_set.end();

        TT_ASSERT(
            is_op_found_schedule,
            "User provided an op tagged for chip break not in the schedule. (may have been consteval)",
            name_tagged_for_chip_id_break);
    }

    // Check all user-defined epoch breaks exists
    for (const string& name_tagged_for_epoch_break : config.ops_tagged_for_epoch_break)
    {
        bool is_op_found_schedule =
            scheduled_ops_set.find(name_tagged_for_epoch_break) != scheduled_ops_set.end();

        TT_ASSERT(
            is_op_found_schedule,
            "User provided an op tagged for epoch break not in the schedule. (may have been consteval)",
            name_tagged_for_epoch_break);
    }
}


void check_user_defined_op_names_exist_in_schedule(
    const vector<vector<string>>& op_names_to_chip_or_epoch_break,
    const vector<string>& scheduled_ops)
{
    if (op_names_to_chip_or_epoch_break.empty())
    {
        return;
    }

    unordered_set<string> scheduled_ops_set;
    for (const string& name : scheduled_ops)
    {
        scheduled_ops_set.insert(name);
    }

    // Check all user-defined op_to_chip_id breaks exists
    for (const vector<string>& op_names_for_epoch_or_chip_break : op_names_to_chip_or_epoch_break)
    {
        for (const string& op_name : op_names_for_epoch_or_chip_break)
        {
            bool is_op_found_schedule =
                scheduled_ops_set.find(op_name) != scheduled_ops_set.end();

            TT_ASSERT(is_op_found_schedule, "User provided an op tagged for epoch/chip break not in the schedule: {}", op_name);
        }
    }

}

vector<OpGroupToPlace> generate_simple_placer_workload(
    const PlacerConfig& config,
    const vector<string>& scheduled_ops)
{
    check_user_defined_op_names_exist_in_schedule(config, scheduled_ops);

    // For now, no actual groupings.. each group will just contain a single op
    uint32_t current_chip_index = 0;
    vector<OpGroupToPlace> placer_op_group_workload;
    for (const string& op_name : scheduled_ops)
    {
        bool increment_epoch = false;
        NodeEpochType epoch_type = config.op_to_epoch_type.at(op_name);

        if (not placer_op_group_workload.empty())
        {
            // Don't trigger increment_epoch/chip when placing first OpGroup
            if (config.ops_tagged_for_epoch_break.find(op_name) != config.ops_tagged_for_epoch_break.end())
            {
                increment_epoch = true;
            }
            if (config.ops_tagged_for_chip_id_break.find(op_name) != config.ops_tagged_for_chip_id_break.end())
            {
                current_chip_index = (current_chip_index + 1) % config.chip_ids.size();
            }
        }

        placer_op_group_workload.push_back(
            OpGroupToPlace{
                .op_group_id = OpGroupToPlace::get_next_op_group_id(),
                .op_names = {op_name},
                .op_name_to_relative_offset_from_first_op = {},
                .chip_id = config.chip_ids.at(current_chip_index),
                .increment_epoch = increment_epoch,
                .epoch_type=epoch_type,
            }
        );
    }
    return placer_op_group_workload;
}

vector<OpGroupToPlace> generate_forward_placer_workload(
    const PlacerConfig& config,
    const vector<string>& scheduled_ops)
{
    vector<OpGroupToPlace> op_groups;

    for (const string& op_name : scheduled_ops)
    {
        bool increment_epoch = false;

        if (not op_groups.empty())
        {
            // Don't trigger increment_epoch/chip when placing first OpGroup
            if (config.ops_tagged_for_epoch_break.find(op_name) != config.ops_tagged_for_epoch_break.end())
            {
                increment_epoch = true;
            }
        }

        NodeEpochType epoch_type = config.op_to_epoch_type.at(op_name);
        if (epoch_type == NodeEpochType::Forward)
        {
            uint32_t assigned_chip_id = config.get_chip_id(op_name);
            op_groups.push_back(
                OpGroupToPlace{
                    .op_group_id = OpGroupToPlace::get_next_op_group_id(),
                    .op_names={op_name},
                    .op_name_to_relative_offset_from_first_op = {},
                    .chip_id = assigned_chip_id,
                    .increment_epoch = increment_epoch,
                    .epoch_type=NodeEpochType::Forward,
                }
            );
        }
    }
    return op_groups;
}

vector<OpGroupToPlace> generate_backward_placer_workload(
    const PlacerConfig& config, const vector<string>& scheduled_ops)
{
    vector<OpGroupToPlace> op_groups;
    for (auto it = scheduled_ops.begin(); it != scheduled_ops.end(); ++it)
    {
        auto bwd_op = *it;
        if (config.op_to_epoch_type.at(bwd_op) == NodeEpochType::Backward)
        {
            bool is_grad_op =
                config.op_to_grad_op.find(bwd_op) != config.op_to_grad_op.end() and config.op_to_grad_op.at(bwd_op);
            bool is_recompute_op = config.op_to_recompute_op.find(bwd_op) != config.op_to_recompute_op.end() and
                                   config.op_to_recompute_op.at(bwd_op);
            std::string op_type = (is_grad_op ? "grad_op" : (is_recompute_op ? "recompute_op" : "bwd_op"));

            log_debug(tt::LogPlacer, "\tbwd_node: {} is type: {}", bwd_op, op_type);
            op_groups.push_back(OpGroupToPlace{
                .op_group_id = OpGroupToPlace::get_next_op_group_id(),
                .op_names = {bwd_op},
                .op_name_to_relative_offset_from_first_op = {},
                .chip_id = config.get_chip_id(bwd_op),
                .increment_epoch =
                    config.ops_tagged_for_epoch_break.find(bwd_op) != config.ops_tagged_for_epoch_break.end(),
                .epoch_type = NodeEpochType::Backward});
        }
    }
    return op_groups;
}

vector<OpGroupToPlace> generate_optimizer_placer_workload(
    const PlacerConfig& config,
    const vector<string>& scheduled_ops)
{
    vector<OpGroupToPlace> op_groups;
    for (const string& node_name : scheduled_ops)
    {
        NodeEpochType epoch_type = config.op_to_epoch_type.at(node_name);
        if (epoch_type == NodeEpochType::Optimizer)
        {
            uint32_t chip_id = config.get_chip_id(node_name);
            bool increment_epoch = false;
            if (config.ops_tagged_for_epoch_break.find(node_name) !=
                config.ops_tagged_for_epoch_break.end())
            {
                increment_epoch = true;
            }

            op_groups.push_back(
                    OpGroupToPlace{
                        .op_group_id = OpGroupToPlace::get_next_op_group_id(),
                        .op_names = {node_name},
                        .op_name_to_relative_offset_from_first_op = {},
                        .chip_id = chip_id,
                        .increment_epoch = increment_epoch,
                        .epoch_type = NodeEpochType::Optimizer,
                    });
        }
    }
    return op_groups;
}

ChipIdToPlacerWorkload generate_placer_workload(
    const PlacerConfig& config,
    const vector<string>& scheduled_ops)
{
    check_user_defined_op_names_exist_in_schedule(config, scheduled_ops);
    if (config.device_config.is_grayskull())
    {
        TT_ASSERT(not config.op_to_chip_id_assignment.empty(), "op to chip_id assignment not populated");
    }
    map<uint32_t, vector<OpGroupToPlace>> placer_workload;

    for (auto&& op_group : generate_forward_placer_workload(config, scheduled_ops))
    {
        placer_workload[op_group.chip_id].emplace_back(op_group);
    }
    for (auto&& op_group : generate_backward_placer_workload(config, scheduled_ops))
    {
        placer_workload[op_group.chip_id].emplace_back(op_group);
    }
    for (auto&& op_group : generate_optimizer_placer_workload(config, scheduled_ops))
    {
        placer_workload[op_group.chip_id].emplace_back(op_group);
    }
    return placer_workload;
}

vector<OpGroupToPlace> generate_wormhole_placer_workload(
    const PlacerConfig& config,
    const vector<string>& scheduled_ops)
{
    check_user_defined_op_names_exist_in_schedule(config, scheduled_ops);
    vector<OpGroupToPlace> placer_workload;

    for (auto&& op_group : generate_forward_placer_workload(config, scheduled_ops))
    {
        placer_workload.emplace_back(op_group);
    }
    for (auto&& op_group : generate_backward_placer_workload(config, scheduled_ops))
    {
        placer_workload.emplace_back(op_group);
    }
    for (auto&& op_group : generate_optimizer_placer_workload(config, scheduled_ops))
    {
        placer_workload.emplace_back(op_group);
    }
    return placer_workload;
}

} // namespace lowering
} // namespace placer
} // namespace tt
