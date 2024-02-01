// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placer/dram_logger.hpp"
#include "graph_lib/node.hpp"

#include <fstream>
#include <experimental/filesystem>
#include "utils/logger.hpp"

namespace tt::placer {

void DramLogger::log_allocate(
            const graphlib::Node *node, 
            std::uint32_t dram_channel, 
            std::uint32_t addr, 
            std::uint32_t size,
            std::uint32_t allocate_epoch)
{
    allocations.push_back(Allocation{
            .name = node->name(),
            .dram_channel = dram_channel,
            .addr = addr,
            .size = size,
            .allocate_epoch = allocate_epoch,
            .deallocate_epoch = 0});
    log_trace(tt::LogPlacer, "Placing {} to channel: {} at addr: 0x{:x}", node->name(), dram_channel, addr);
}

void DramLogger::log_deallocate(std::uint32_t dram_channel, std::uint32_t addr, std::uint32_t deallocate_epoch)
{
    for (Allocation &alloc : allocations)
    {
        if ((alloc.deallocate_epoch == 0) && (alloc.dram_channel == dram_channel) && (alloc.addr == addr)) {
            alloc.deallocate_epoch = deallocate_epoch;
            return;
        }
    }
    // commenting out for now for wormhole, need to fix
    //TT_THROW("Logging a deallocation that can't be found in allocation list.");
}

void DramLogger::dump_to_reportify(const std::string &output_dir, const std::string &test_name) const
{
    if (env_as<bool>("PYBUDA_DISABLE_REPORTIFY_DUMP"))
        return;
    std::experimental::filesystem::create_directories(output_dir);
    std::string filename = output_dir + "/memory_dram_dynamic_analysis.json";
    std::ofstream out(filename);
    TT_ASSERT(out.is_open(), "Can't open " + filename + " for writing.");


    std::uint32_t max_dram_ch = 0;
    std::unordered_map<std::uint32_t, std::vector<Allocation>> alloc_table;
    for (const Allocation &alloc : allocations) {
        if (alloc.dram_channel > max_dram_ch)
            max_dram_ch = alloc.dram_channel;
        alloc_table[alloc.dram_channel].push_back(alloc);
    }

    out << "{" << std::endl
        << "  \"test_name\": \"" << test_name << "\", " 
        << "  \"dram\": 1," << std::endl;
    
    for (std::uint32_t dram_channel = 0; dram_channel <= max_dram_ch; dram_channel++)
    {

        out << "  \"Dynamic DRAM Map for Bank " << dram_channel << "\": [" << std::endl;
        bool first = true;
        for (const Allocation &alloc : alloc_table[dram_channel])
        {
            if (!first) {
                out << ", " << std::endl;
            }
            first = false;

            out << "{" 
                << "\"queue_name\": \"" << alloc.name << "\", "
                << "\"base\": " << alloc.addr << ", "
                << "\"size\": " << alloc.size << ", "
                << "\"allocation_cycle\": " << alloc.allocate_epoch;

            if (alloc.deallocate_epoch > 0) 
                out << ", \"deallocation_cycle\": " << alloc.deallocate_epoch;

            out << "}";
        }
        out << std::endl << "]";
        if (dram_channel < max_dram_ch)
            out << ", " << std::endl;

    }
    out << "}" << std::endl;
    out.close();
}

}

