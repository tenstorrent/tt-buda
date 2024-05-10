// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "placer/placer.hpp"
#include "placer/lowering_utils.hpp"

#include <vector>

using std::uint32_t;
using std::string;
using std::vector;
using std::unordered_map;
using std::unordered_set;

// Aliases
using NodeEpochType = tt::graphlib::NodeEpochType;

namespace tt {

// Forward Declares
namespace graphlib
{
    class Graph;
}


namespace placer {
namespace lowering {

unordered_map<string, vector<string>> get_fwd_to_bwd_nodes(graphlib::Graph const* graph);
unordered_map<string, map<int, vector<string>>> get_fwd_to_opt_nodes(
    graphlib::Graph const* graph, const vector<string>& scheduled_ops);

unordered_map<string, NodeEpochType>
get_op_to_epoch_type_mapping(graphlib::Graph const* graph, const vector<string>& scheduled_ops);
unordered_map<string, bool>
get_op_to_grad_op_mapping(graphlib::Graph const* graph, const vector<string>& scheduled_ops);
unordered_map<string, bool>
get_op_to_recompute_mapping(graphlib::Graph const* graph, const vector<string>& scheduled_ops);
unordered_set<string> get_output_nodes(const graphlib::Graph *graph);

// Returns an ordered list of node names
vector<string> generate_placer_schedule(graphlib::Graph const* graph, PlacementScheduleOrder schedule_type);

unordered_set<string> tag_ops_for_epoch_break(
    const DeviceConfig& device_config,
    const vector<vector<string>>& op_names_to_epoch_break,
    const vector<vector<string>>& op_names_to_chip_break,
    const vector<string>& scheduled_ops,
    graphlib::Graph const* graph,
    bool use_interactive_placer);

unordered_set<string> tag_ops_for_chip_break(
    const DeviceConfig& device_config,
    const vector<vector<string>>& op_names_to_chip_break,
    const vector<string>& scheduled_ops,
    graphlib::Graph const* graph,
    bool use_interactive_placer);

unordered_set<string> tag_ops_for_temporal_epoch_break(
    graphlib::Graph const* graph,
    const vector<string>& scheduled_op_names,
    const std::unordered_map<std::string, placer::PlacerOpOverride>& op_name_to_placer_overrides);

} // end namespace lowering
} // end namespace placer
} // end namespace tt
