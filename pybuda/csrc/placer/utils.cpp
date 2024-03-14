// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placer/placer.hpp"

#include <algorithm>

#include "graph_lib/defines.hpp"
#include "utils/logger.hpp"
#include "utils/assert.hpp"

using NodeEpochType = tt::graphlib::NodeEpochType;
using std::ostream;
using std::to_string;

namespace tt {
namespace placer {

bool is_forward_to_backward_epoch_transition(NodeEpochType prev_epoch_type, NodeEpochType current_epoch_type)
{
    return (prev_epoch_type == NodeEpochType::Forward)
        and (current_epoch_type == NodeEpochType::Backward);
}

bool is_backward_to_optimizer_epoch_transition(NodeEpochType prev_epoch_type, NodeEpochType current_epoch_type)
{
    return prev_epoch_type == NodeEpochType::Backward and current_epoch_type == NodeEpochType::Optimizer;
}


void validate_placer_inputs(const PlacerConfig& config, vector<OpGroupToPlace>& placer_op_group_workload)
{
    std::unordered_set<std::string> visited_ops;

    for (const OpGroupToPlace& op_group : placer_op_group_workload)
    {
        for (std::size_t current_op_index = 0; current_op_index < op_group.op_names.size(); ++current_op_index)
        {
            const string& current_op_name = op_group.op_names.at(current_op_index);

            if (visited_ops.find(current_op_name) != visited_ops.end()) {
                log_fatal("{} belongs to more than one op_group_workload", current_op_name);
            }

            // verify all outputs are on MMIO capable devices for wormhole
            if (config.device_config.is_wormhole() and config.output_ops.find(current_op_name) != config.output_ops.end()) {
                // TODO(jchu): update this assert with MMIO chip ids
                TT_ASSERT(std::find(config.device_config.chips_with_mmio.begin(), config.device_config.chips_with_mmio.end(), op_group.chip_id) != config.device_config.chips_with_mmio.end(),
                    "Placer: For wormhole multichip, we expect all output ops to be placed on MMIO devices.");
            }

            // Validate that the op_grid sizes are able to fit within the device grid_shape
            try {
                const GridShape& op_grid_shape = config.op_to_grid_shape.at(current_op_name);
                TT_ASSERT(op_grid_shape.rows <= config.get_available_rows_on_device());
                if(op_grid_shape.columns > config.device_grid.columns) {
                    throw std::runtime_error("Error: op:" + current_op_name + " grid_shape.columns: " + to_string(op_grid_shape.columns) +
                    " but the device grid_shape.columns is: " + to_string(config.device_grid.columns));
                }

                if (current_op_index > 0)
                {
                    // Validate that all ops belonging to an op-group belong to the same epochType
                    const string& prev_op_name = op_group.op_names.at(current_op_index-1);
                    TT_ASSERT(config.op_to_epoch_type.at(prev_op_name) == config.op_to_epoch_type.at(current_op_name));
                }
                visited_ops.insert(current_op_name);

            } catch (std::out_of_range &e) {
                log_fatal("op_to_grid_shape missing for {}", current_op_name);
            }
        }
    }
}

void validate_chip_mapping(const PlacerConfig& config, vector<OpGroupToPlace>& placer_workload)
{
    for (const OpGroupToPlace& op_group : placer_workload)
    {
        TT_ASSERT(std::find(config.chip_ids.begin(), config.chip_ids.end(), op_group.chip_id) != config.chip_ids.end(), 
                "Placing an op group on chip that's not in the list of available devices: " + std::to_string(op_group.chip_id));
    }
}



} // namespace placer
} // namespace tt

