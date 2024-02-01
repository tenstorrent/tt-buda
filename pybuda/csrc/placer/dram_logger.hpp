// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Log DRAM allocations / deallocations to analyze later
//
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tt::graphlib {
    class Node;
}

namespace tt::placer {

class DramLogger {
    struct Allocation {
        std::string name;
        std::uint32_t dram_channel;
        std::uint32_t addr;
        std::uint32_t size;
        std::uint32_t allocate_epoch;
        std::uint32_t deallocate_epoch = 0;
    };

    std::vector<Allocation> allocations;

public:
    void log_allocate(
            const graphlib::Node *node, 
            std::uint32_t dram_channel, 
            std::uint32_t addr, 
            std::uint32_t size,
            std::uint32_t allocate_epoch);

    void log_deallocate(
            std::uint32_t dram_channel, 
            std::uint32_t addr, 
            std::uint32_t deallocate_epoch);

    void dump_to_reportify(const std::string &output_dir, const std::string &test_name) const;
};

}
