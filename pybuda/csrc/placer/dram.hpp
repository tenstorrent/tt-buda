// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "backend_api/device_config.hpp"
#include "balancer/balancer.hpp"
#include "placer/placer.hpp"

namespace tt
{
struct DramQueueConfigOverride
{
    std::optional<std::uint32_t> chip_id;
    std::optional<std::uint32_t> channel;
    DramQueueConfigOverride(
        std::optional<std::uint32_t> chip_id = std::nullopt, std::optional<std::uint32_t> channel = std::nullopt) :
        chip_id(chip_id), channel(channel)
    {
    }
};
using DramQueueMap = std::unordered_map<std::string, DramQueueConfigOverride>;

namespace graphlib
{
class Graph;
}

namespace placer
{
class DramAllocator;
struct HostMemoryPlacerConfig;

struct DramConfig
{
    uint32_t channel;
    uint32_t sub_channel;
    size_t channel_size;
    Coord location;
    size_t initial_dram_offset;

    static std::vector<DramConfig> get_config(DeviceConfig const &device_config)
    {
        std::vector<DramConfig> ret;
        std::uint32_t num_channels = device_config.get_dram_num_channels();
        std::uint32_t num_subchannels = device_config.get_dram_num_subchannels();
        for (std::uint32_t channel = 0; channel < num_channels; channel++)
        {
            for (std::uint32_t sub_channel = 0; sub_channel < num_subchannels; sub_channel++)
            {
                ret.push_back(DramConfig{
                    channel,
                    sub_channel,
                    device_config.get_dram_channel_capacity(),
                    get_location(device_config, channel, sub_channel),
                    ((device_config.is_grayskull()) ? device_config.get_dram_backend_reserved_max() : device_config.get_dram_backend_reserved(channel)) +
                        0x100 /* tenstorrent/budabackend#461 */,
                });
            }
        }
        return ret;
    }

    static Coord get_location(const DeviceConfig &device_config, std::uint32_t channel, std::uint32_t subchannel = 0)
    {
        // NB: using NOC coordinates here. The coordinates aren't really used by FE.
        // it's also inconsistent because grayskull() config is just wrong.
        // TODO(jchu): this should all be cleaned up in favour of queries to the BE
        // that provide this info explicitly.
        const std::vector<Coord> grayskull_locs = {{0, 0}, {5, 0}, {0, 3}, {5, 3}, {0, 6}, {5, 6}, {0, 9}, {5, 9}};
        if (device_config.is_grayskull())
        {
            TT_ASSERT(channel < grayskull_locs.size());
            return grayskull_locs.at(channel);
        }

        if (device_config.is_wormhole_b0())
        {
            // TT_ASSERT(channel < wormhole_locs.size());
            auto c = device_config.get_dram_core_coord(channel, subchannel);
            return {(std::uint32_t)c.y, (std::uint32_t)c.x};
        }

        TT_THROW("Unknown arch: " + device_config.arch_name);
        return {0, 0};
    }
};

struct DramPlacerConfig
{
    const DeviceConfig &device_config;

    // vector of channels, assume same for each device
    std::vector<DramConfig> dram_config;

    // Allocate input queues in dram or on host
    bool input_queues_on_host;

    // Allocate output queues in dram or on host
    bool output_queues_on_host;

    // Disable dynamic dram support
    bool force_disable_dynamic_dram;

    std::uint32_t p2p_offset;
    std::uint32_t p2p_size;
    std::uint32_t host_mmio_range_offset;
    std::uint32_t host_mmio_range_size;

    // Manual dram queue placement
    DramQueueMap manual_dram_queue_placement;

    DramPlacerConfig(
        DeviceConfig const &device_config,
        bool input_queues_on_host,
        bool output_queues_on_host,
        const DramQueueMap &manual_dram_queue_placement) :
        device_config(device_config), manual_dram_queue_placement(manual_dram_queue_placement)
    {
        // host_mmio is part of the dram memory allocated for host->device acess. it is defined by offset and size.
        host_mmio_range_offset = device_config.get_host_mmio_range_offset();
        host_mmio_range_size = device_config.get_host_mmio_range_size();
        // p2p is part of the dram memory allocated for device->device acess. it is defined by offset and size.
        p2p_size = device_config.get_p2p_size();
        p2p_offset = device_config.get_p2p_offset();  // same for all channels
        TT_ASSERT(p2p_offset + p2p_size == 0x40000000);
        dram_config = DramConfig::get_config(device_config);
        this->input_queues_on_host = input_queues_on_host;
        this->output_queues_on_host = output_queues_on_host;
        force_disable_dynamic_dram = env_as<bool>("PYBUDA_DISABLE_DYNAMIC_DRAM");
    }
};

struct QueueDRAMPlacementParameters
{
    const DramPlacerConfig *config;

    const Node *node;

    // Producer op grid, which will drive the grid of the queue
    GridShape grid_shape;

    // For each grid buffer, list locations of the consumers (i.e. readers) and the producer (i.e. writer). The pair is
    // coordinate + epoch number
    using ConsumerMap = std::
        unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::vector<std::pair<Coord, std::uint32_t>>>>;
    using ProducerMap =
        std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::pair<Coord, std::uint32_t>>>;
    ConsumerMap consumer_loc;
    ProducerMap producer_loc;

    balancer::BlockShape block_shape;
    std::uint32_t producer_epoch;
    std::uint32_t last_consumer_epoch;
    bool in_p2p_region_soft;
    bool in_p2p_region_hard;
    bool is_input;
    bool is_prologue;
    std::size_t queue_size;
};

using DRAMScheduleData = std::pair<QueuePlacement, QueueDRAMPlacementParameters>;

// Figure out which cores are reading from which dram buffer (or writing to)
// dram_buffer is relative coordinate within the buffer grid
std::vector<Coord> get_reader_cores(
    const Node *node, const OpPlacement &placement, std::uint32_t operand, Coord dram_buffer, GridShape queue_grid);

bool disable_dynamic_dram_if_possible(
    const std::vector<DRAMScheduleData> &scheduled_queue_placements, const DramAllocator &allocator);

bool try_static_dram_placement(
    std::vector<DRAMScheduleData> &scheduled_queue_placements, DramAllocator &chip_dram_allocator, int microbatch_size);

// Place and allocate DRAM queues
void place_dram_queues(
    graphlib::Graph *graph,
    PlacerSolution &placer_solution,
    balancer::BalancerSolution &balancer_solution,
    const HostMemoryPlacerConfig &host_memory_placer_config,
    const DramPlacerConfig &config,
    std::vector<DramAllocator> &chip_dram_allocators);

}  // namespace placer
}  // namespace tt

