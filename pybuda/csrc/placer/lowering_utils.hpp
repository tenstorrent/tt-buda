// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "placer/placer.hpp"

#include <vector>

using std::uint32_t;
using std::string;
using std::vector;
using std::unordered_map;

// Aliases
using NodeEpochType = tt::graphlib::NodeEpochType;

namespace tt {
namespace placer {
namespace lowering {

void validate_placer_config(const PlacerConfig& placer_config);

using ChipId = uint32_t;

vector<ChipId> apply_chip_placement_policy(const DeviceConfig& config, ChipPlacementPolicy chip_placement_policy, const vector<ChipId>& chip_ids);

unordered_map<string, GridShape> get_op_to_grid_shape(
        const vector<string>& scheduled_ops,
        uint32_t default_rows = 1,
        uint32_t default_columns = 1);

// Each returned OpGroupToPlace in the list only contains a single op in its grouping
vector<OpGroupToPlace> generate_simple_placer_workload(
        const PlacerConfig& config,
        const vector<string>& scheduled_ops);

ChipIdToPlacerWorkload generate_placer_workload(
        const PlacerConfig& config, const vector<string>& scheduled_ops);


vector<OpGroupToPlace> generate_wormhole_placer_workload(
    const PlacerConfig& config,
    const vector<string>& scheduled_ops);


void check_user_defined_op_names_exist_in_schedule(const PlacerConfig& config, const vector<string>& scheduled_ops);

void check_user_defined_op_names_exist_in_schedule(
    const vector<vector<string>>& op_names_to_chip_or_epoch_break,
    const vector<string>& scheduled_ops);

} // end namespace lowering
} // end namespace placer
} // end namespace tt
