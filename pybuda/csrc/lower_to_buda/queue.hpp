// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <optional>

#include "graph_lib/node_types.hpp"
#include "lower_to_buda/common.hpp"
#include "lower_to_buda/device.hpp"
#include "lower_to_buda/op.hpp"

namespace tt {

struct BudaQueueDimensions {
    int grid_r, grid_c;
};

enum BudaQueueLocation { DRAM, HOST };

struct BudaQueueDramLoc {
    std::uint32_t dram_channel;
    std::size_t dram_address;
};

struct BudaQueueHostLoc {
    std::uint32_t host_channel;
    std::size_t host_address;
};

struct BudaQueue {

    std::string name;
    std::string input_name;
    std::string type;
    std::string memory_access;
    std::string alias;
    int entries;
    int microbatch;
    BudaQueueDimensions dims;
    DataFormat data_format;
    std::vector<BudaDevice> target_devices;
    BudaQueueLocation loc;
    std::vector<BudaQueueDramLoc> dram_loc;
    std::vector<BudaQueueHostLoc> host_loc;
    BudaBlocks blocks;
    graphlib::UBlockOrder ublock_order = graphlib::UBlockOrder::R;
    BudaQueueLayout layout = BudaQueueLayout::Tilized;
    TileDim tile_dim_;

    BudaQueue(const std::string &name, const std::string &type, const std::string& memory_access, std::vector<uint32_t> device, TileDim tile_dim)
        : name(name), type(type), memory_access(memory_access), tile_dim_(tile_dim)
    {
        target_devices = std::vector<BudaDevice>();
        for (uint32_t dev: device)
        {
            target_devices.emplace_back(BudaDevice(dev));
        }
    }

    std::string as_string(int padded_name_length = 0) const;

};

std::ostream &operator<<(std::ostream &os, BudaQueue const &m);
std::ostream &operator<<(std::ostream &os, BudaQueueLocation const &l);
std::ostream &operator<<(std::ostream &os, BudaQueueDramLoc const &l);
std::ostream &operator<<(std::ostream &os, BudaQueueHostLoc const &l);
std::ostream &operator<<(std::ostream &os, BudaQueueLayout const &l);

} // namespace tt

