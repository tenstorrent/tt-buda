// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "lower_to_buda/graph.hpp"

#include <ostream>

#include "lower_to_buda/comment.hpp"

namespace tt {

std::string get_subgraph_name(
    graphlib::NodeEpochType epoch_type,
    int epoch_number,
    const std::string& arch_name,
    std::uint32_t temporal_epoch_id,
    std::uint32_t subgraph_index)
{
    std::string ret;
    switch (epoch_type) {
        case graphlib::NodeEpochType::Forward: ret = "fwd_"; break;
        case graphlib::NodeEpochType::Backward: ret = "bwd_"; break;
        case graphlib::NodeEpochType::Optimizer: ret = "opt_"; break;
    }
    ret = ret + std::to_string(subgraph_index) + "_";
    if (arch_name == "grayskull")
    {
        return ret + std::to_string(epoch_number);
    } else {
        return ret + std::to_string(epoch_number) + "_" + "temporal_epoch_" + std::to_string(temporal_epoch_id);
    }
}

std::vector<std::uint32_t> BudaGraph::get_matching_epoch(graphlib::NodeEpochType type) const
{
    std::vector<std::uint32_t> ret;
    for (std::size_t i=0; i < epoch_types.size(); i++)
        if (epoch_types[i] == type)
            ret.push_back(i);
    return ret;
}

std::ostream &operator<<(std::ostream &os, BudaGraph const &g) {

    const std::string indent = "    ";
    for (std::size_t epoch = 0; epoch < g.ops.size(); epoch++) {
        if (g.arch_name == "grayskull" and g.ops[epoch].size() == 0)
        {
            continue;
        }
        os << "  " << get_subgraph_name(g.epoch_types[epoch], epoch, g.arch_name, g.epoch_to_temporal_epoch_id[epoch], g.epoch_to_subgraph_index[epoch]) << ":" << std::endl;
        os << indent << "target_device: " << g.epoch_target_devices[epoch] << std::endl;

        int input_count = (g.epoch_types[epoch] == graphlib::NodeEpochType::Optimizer) ? 1 : g.microbatch_size;
        os << indent << "input_count: " << input_count << std::endl;
        for (const BudaOp &op : g.ops[epoch]) {
            if (op.debug_info)
                os << std::endl << op.debug_info;
            os << indent << op << std::endl;
        }
        os << std::endl;
    }

    return os;
}

}


