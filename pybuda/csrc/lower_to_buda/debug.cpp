// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "lower_to_buda/debug.hpp"

#include "backend_api/device_config.hpp"
#include "balancer/balancer_utils.hpp"
#include "balancer/types.hpp"
#include "utils/env.hpp"

namespace tt
{
static bool debug_info_enabled() { return env_as<bool>("PYBUDA_NETLIST_DEBUG_INFO"); }

void to_debug_info(std::ostream& os, tt::DeviceConfig const& device_config)
{
    if (not debug_info_enabled())
        return;
    os << device_config << std::endl;
}

static void to_debug_info(std::ostream& os, balancer::BufferModel const& buffer, bool include_t = false)
{
    os << "l1_size_tiles: " << buffer.l1_size_tiles;
    os << ", ";
    os << "total_size_bytes: " << buffer.size_bytes(include_t);
}

static void to_debug_info(std::ostream& os, std::vector<balancer::BufferModel> const& buffers, bool include_t = false)
{
    for (std::size_t i = 0; i < buffers.size(); ++i)
    {
        if (not buffers[i])
            continue;
        os << "    [" << i << "] = ";
        to_debug_info(os, buffers[i], include_t);
        os << std::endl;
    }
}

static void to_debug_info(std::ostream& os, std::vector<std::size_t> const& input_dram_io_buf_size_tiles)
{
    for (std::size_t i = 0; i < input_dram_io_buf_size_tiles.size(); ++i)
    {
        os << "    [" << i << "] = " << input_dram_io_buf_size_tiles[i];
        os << std::endl;
    }
}

static void to_debug_info_tile_sizes(std::ostream& os, balancer::OpModel const& op_model)
{
    std::vector<DataFormat> dfs;
    auto collect_dfs = [&dfs](std::vector<balancer::BufferModel> const& buffers)
    {
        for (balancer::BufferModel const& buffer : buffers)
        {
            if (std::find(dfs.begin(), dfs.end(), buffer.data_format) == dfs.end())
                dfs.push_back(buffer.data_format);
        }
    };
    collect_dfs(op_model.input_buffers);
    collect_dfs(op_model.parameter_buffers);
    collect_dfs(op_model.intermediate_buffers);
    collect_dfs(op_model.output_buffers);
    std::sort(dfs.begin(), dfs.end());
    for (auto df : dfs)
    {
        os << "    " << df << ": " << balancer::tile_size_bytes(df) << " bytes";
    }
}

static std::size_t get_dram_io_size_bytes(
    std::vector<balancer::BufferModel> const& buffers, std::vector<std::size_t> const& input_dram_io_buf_size_tiles)
{
    TT_ASSERT(buffers.size() == input_dram_io_buf_size_tiles.size());
    std::size_t total_size = 0;
    for (std::size_t i = 0; i < input_dram_io_buf_size_tiles.size(); ++i)
      {
        total_size += balancer::tile_size_bytes(buffers[i].data_format) * input_dram_io_buf_size_tiles[i];
    }

    return total_size;
}

    void to_debug_info(
        std::ostream& os,
        std::string const& name,
        balancer::OpModel const& op_model,
        std::string const& arch_name,
        std::vector<std::size_t> const& input_dram_io_buf_size_tiles)
{
    if (not debug_info_enabled())
        return;

    os << std::endl << "Debug Info: " << name << std::endl;
    os << std::endl;
    os << op_model;
    os << std::endl;
    os << std::endl;
    os << "L1 Breakdown:" << std::endl;
    os << "  tile_sizes:" << std::endl;
    to_debug_info_tile_sizes(os, op_model);
    os << std::endl;
    os << "  input_buffers:" << std::endl;
    to_debug_info(os, op_model.input_buffers);
    os << "  parameter_buffers:" << std::endl;
    to_debug_info(os, op_model.parameter_buffers, true);
    os << "  intermediate_buffers:" << std::endl;
    to_debug_info(os, op_model.intermediate_buffers);
    os << "  output_buffers:" << std::endl;
    to_debug_info(os, op_model.output_buffers);
    os << std::endl;
    os << "  dram_io_buffers:" << std::endl;
    to_debug_info(os, input_dram_io_buf_size_tiles);
    os << std::endl;
    os << "  overlay_size: " << op_model.overlay_size << std::endl;
    os << std::endl;
    std::size_t dram_io_size_bytes = get_dram_io_size_bytes(op_model.input_buffers, input_dram_io_buf_size_tiles);
    os << "Total L1 buffer usage: " << (op_model.get_l1_memory_usage() + dram_io_size_bytes + op_model.overlay_size - 64 * 1024) << " bytes" << std::endl;
    os << "Estimated cycle count: " << op_model.get_execution_cycles(arch_name) << " cycles" << std::endl;
}
}
