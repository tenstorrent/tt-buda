// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placer/chip_id_assignment.hpp"

#include "utils/assert.hpp"
#include "utils/logger.hpp"

namespace tt::placer {

static int get_num_fwd_ops(const ChipPlacerConfig& config, const vector<string>& scheduled_ops) {
    int num_fwd_ops = 0;
    for (const string& op_name : scheduled_ops)
    {
        NodeEpochType epoch_type = config.op_to_epoch_type.at(op_name);
        if (epoch_type == NodeEpochType::Forward)
        {
            num_fwd_ops += 1;
        }
    }
    return num_fwd_ops;
}

unordered_map<string, uint32_t> get_grayskull_fwd_op_to_chip_id_placement(
    const ChipPlacerConfig& config,
    const vector<string>& scheduled_ops)
{
    unordered_map<uint32_t, int> chip_id_to_num_assigned_ops;
    unordered_map<string, uint32_t> fwd_op_to_chip_id_placement;
    uint32_t current_chip_index = 0;
    uint32_t current_chip_id = config.chip_ids[current_chip_index];

    bool use_user_defined_scheme = not config.ops_tagged_for_chip_id_break.empty() or config.chip_ids.size() == 1;
    if (use_user_defined_scheme)
    {
        for (const string& op_name : scheduled_ops)
        {
            if (chip_id_to_num_assigned_ops.find(current_chip_id) != chip_id_to_num_assigned_ops.end())
            {
                if (config.ops_tagged_for_chip_id_break.find(op_name) != config.ops_tagged_for_chip_id_break.end())
                {
                    current_chip_index = (current_chip_index + 1) % config.chip_ids.size();
                }
            }

            NodeEpochType epoch_type = config.op_to_epoch_type.at(op_name);
            if (epoch_type == NodeEpochType::Forward)
            {
                current_chip_id = config.chip_ids[current_chip_index];
                fwd_op_to_chip_id_placement[op_name] = current_chip_id;
                chip_id_to_num_assigned_ops[current_chip_id] += 1;
            }
        }
    }
    else
    {
        log_info("Placer: Running Grayskull multichip auto-placement");
        int ops_per_device = ceil((float)get_num_fwd_ops(config, scheduled_ops) / config.chip_ids.size());
        int fwd_op_idx = 0;

        for (const string& op_name : scheduled_ops)
        {
            if (config.op_to_epoch_type.at(op_name) == NodeEpochType::Forward)
            {
                current_chip_id = config.chip_ids[current_chip_index];
                fwd_op_to_chip_id_placement[op_name] = current_chip_id;
                chip_id_to_num_assigned_ops[current_chip_id] += 1;
                fwd_op_idx += 1;

                // round-robin available chips
                if (fwd_op_idx % ops_per_device == 0)
                {
                    current_chip_index = (current_chip_index + 1) % config.chip_ids.size();
                }
            }
        }
    }

    return fwd_op_to_chip_id_placement;
}

unordered_map<string, uint32_t> get_op_to_chip_id_assignment(
    const ChipPlacerConfig& config,
    const vector<string>& scheduled_ops)
{
    if (config.arch_name != "grayskull")
    {
        // return empty map because we don't have a chip id assignment scheme for this arch
        // chip-id assignments will be generated after creating spatial epochs
        return {};
    }
    unordered_map<string, uint32_t> op_to_chip_id_assignment = get_grayskull_fwd_op_to_chip_id_placement(config, scheduled_ops);

    // chip-id assignment for BWD nodes
    for (int i = scheduled_ops.size() - 1; i >= 0; --i)
    {
        const string& fwd_node_name = scheduled_ops.at(i);
        NodeEpochType epoch_type = config.op_to_epoch_type.at(fwd_node_name);

        if (epoch_type == NodeEpochType::Forward)
        {
            uint32_t fwd_chip_id = op_to_chip_id_assignment.at(fwd_node_name);
            bool has_bwd_nodes = config.fwd_to_bwd_nodes.find(fwd_node_name) != config.fwd_to_bwd_nodes.end();

            if (has_bwd_nodes) {
                for (const auto& bwd_node_name : config.fwd_to_bwd_nodes.at(fwd_node_name))
                {
                    op_to_chip_id_assignment[bwd_node_name] = fwd_chip_id;
                }
            }
        }
    }

    // chip-id assignment for OPT nodes
    for (const string& fwd_node_name : scheduled_ops)
    {
        NodeEpochType epoch_type = config.op_to_epoch_type.at(fwd_node_name);
        if (epoch_type == NodeEpochType::Forward)
        {
            bool has_opt_nodes = config.fwd_to_opt_nodes.find(fwd_node_name) != config.fwd_to_opt_nodes.end();
            if (has_opt_nodes) {
                uint32_t fwd_chip_id = op_to_chip_id_assignment.at(fwd_node_name);
                for (const auto& [operand_index, opt_node_names] : config.fwd_to_opt_nodes.at(fwd_node_name))
                {
                    for (const string& opt_node_name : opt_node_names)
                    {
                        op_to_chip_id_assignment[opt_node_name] = fwd_chip_id;

                    }
                }
            }
        }
    }
    return op_to_chip_id_assignment;
}

} // namespace tt::placer
