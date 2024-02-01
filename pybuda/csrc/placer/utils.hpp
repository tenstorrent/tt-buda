// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "placer/placer.hpp"

#include <vector>
#include <map>
#include <unordered_map>
#include <string>

// Aliases
using NodeEpochType = tt::graphlib::NodeEpochType;

using std::uint32_t;
using std::string;
using std::vector;

namespace tt {
namespace placer {

bool is_backward_to_optimizer_epoch_transition(NodeEpochType prev_epoch_type, NodeEpochType current_epoch_type);
bool is_forward_to_backward_epoch_transition(NodeEpochType prev_epoch_type, NodeEpochType current_epoch_type);

void validate_placer_inputs(const PlacerConfig& config, vector<OpGroupToPlace>& placer_op_group_workload);
void validate_chip_mapping(const PlacerConfig& config, vector<OpGroupToPlace>& placer_workload);


} // end namespace placer
} // end namespace tt
