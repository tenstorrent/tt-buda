// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "placer/placer.hpp"

namespace tt::placer {

struct ChipPlacerConfig
{
    // Arch config
    std::vector<std::uint32_t> chip_ids;
    string arch_name;    

    // Capture any user or op-specific config for placement
    // like chip-breaks or epoch-breaks
    unordered_map<string, NodeEpochType> op_to_epoch_type;

    // captures any user-configuration for chip-breaking
    unordered_set<string> ops_tagged_for_chip_id_break;
    unordered_set<string> ops_tagged_for_epoch_break;
    unordered_map<string, int> fracture_chip_id_assignments;

    unordered_map<string, vector<string>> fwd_to_bwd_nodes;
    unordered_map<string, map<int, vector<string>>> fwd_to_opt_nodes;
    unordered_set<string> output_ops = {};
    vector<int> chips_with_mmio; // for wormhole
};

using OpToChipIdAssignment = unordered_map<string, uint32_t>;

unordered_map<string, uint32_t> get_op_to_chip_id_assignment(
    const ChipPlacerConfig& config,
    const vector<string>& scheduled_ops);

enum class ChipPlacementPolicy
{
    MMIO_LAST = 0, // use chip id order as given by the user, use mmio chips last
    SNAKE = 1,     // sort chip ids in a snake pattern
};

inline ChipPlacementPolicy chip_placement_policy_from_string(std::string const& s)
{
    if (s == "MMIO_LAST") {
        return ChipPlacementPolicy::MMIO_LAST;
    } else if (s == "SNAKE") {
        return ChipPlacementPolicy::SNAKE;
    }
    TT_ASSERT(false);
    return ChipPlacementPolicy::MMIO_LAST;
}

} // namespace tt::placer
