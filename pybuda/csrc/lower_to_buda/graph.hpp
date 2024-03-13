// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "lower_to_buda/device.hpp"
#include "lower_to_buda/op.hpp"

namespace tt {

struct BudaGraph {

    std::string name;
    std::string arch_name;
    std::vector<graphlib::NodeEpochType> epoch_types;
    std::vector<std::vector<BudaDevice>> epoch_target_devices;
    std::vector<std::vector <BudaOp>> ops; // arrays of ops per epoch
    std::vector<std::uint32_t> epoch_to_temporal_epoch_id;
    std::vector<std::uint32_t> epoch_to_subgraph_index;
    std::uint32_t microbatch_size;

    BudaGraph(const std::string &name, const std::string& arch_name, std::uint32_t microbatch)
        : name(name), arch_name(arch_name), microbatch_size(microbatch) {}
    std::vector<std::uint32_t> get_matching_epoch(graphlib::NodeEpochType type) const;

};

std::string get_subgraph_name(graphlib::NodeEpochType epoch_type, int epoch_number, const std::string& arch_name, std::uint32_t temporal_epoch_id, std::uint32_t subgraph_index);
std::ostream &operator<<(std::ostream &os, BudaGraph const &g);

} // namespace tt


