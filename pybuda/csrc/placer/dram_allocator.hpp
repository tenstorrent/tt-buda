// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Allocate queues in DRAM. Keep track of allocated space, try to distribute allocations in a good way.
//
#pragma once

#include "placer/dram.hpp"
#include "placer/dram_logger.hpp"

namespace tt
{
namespace placer
{
struct Block
{
    std::uint32_t addr, size;
};
struct Blocks
{
    std::map<std::uint32_t, Block> free_blocks_start;  // keyed on start addr
    std::unordered_map<std::uint32_t, Block> free_blocks_end;    // keyed on start+size
    std::unordered_map<std::uint32_t, Block> allocated_blocks;   // keyed on start
};
// Allocate buffers within one channel
class ChannelAllocator
{
   public:
    ChannelAllocator() {}
    virtual ~ChannelAllocator() = default;
    virtual bool allocate(std::uint32_t size, std::uint32_t &addr) = 0;  // return true if allocated, and update addr
    virtual void deallocate(std::uint32_t addr) = 0;
    virtual std::uint32_t get_capacity() = 0;
    virtual Blocks get_blocks() = 0;
};

class ChannelPicker
{
   public:
    virtual ~ChannelPicker() = default;
    virtual std::uint32_t pick_channel(
        const QueueDRAMPlacementParameters &parameters,
        Coord /*c*/,
        const std::vector<std::unique_ptr<ChannelAllocator>> &channel_allocators) = 0;
};

enum DRAMPlacementAlgorithm
{
    ROUND_ROBIN = 1,
    ROUND_ROBIN_FLIP_FLOP = 2,
    GREATEST_CAPACITY = 3,
    CLOSEST = 4
};

enum AllocationAlgorithm
{
    BEST_FIT = 1
};

// Allocate queues across all channels
class DramAllocator
{
   private:
    const DramPlacerConfig &dram_config;
    const std::string graph_name;
    std::uint32_t chip_id;
    std::unique_ptr<DramLogger> dram_logger;

    std::vector<std::unique_ptr<ChannelAllocator>> channel_allocators;
    std::unique_ptr<ChannelAllocator> p2p_allocator;
    std::unique_ptr<ChannelPicker> channel_picker;

    std::vector<QueueBufferPlacement> allocate_buffers(const QueueDRAMPlacementParameters &parameters);
    const std::unique_ptr<ChannelAllocator> &get_allocator(std::uint32_t channel_index, bool in_p2p_region) const;

   public:
    DramAllocator(
        const DramPlacerConfig &dram_config,
        const std::string &graph_name,
        std::uint32_t chip_id,
        std::vector<Blocks> &allocated_blocks,
        DRAMPlacementAlgorithm placement_algorithm = ROUND_ROBIN,
        AllocationAlgorithm allocator_algorithm = BEST_FIT);
    void allocate_queues(std::vector<DRAMScheduleData> &scheduled_queue_placements, bool disable_dynamic_dram, int microbatch_size);
    std::vector<Blocks> get_blocks();
    std::pair<uint32_t, uint32_t> get_dram_free_space();
};

}  // namespace placer
}  // namespace tt
