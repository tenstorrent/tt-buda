// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "placer/dram_allocator.hpp"
#include "placer/dram.hpp"

namespace tt::placer {

class BestFitAllocator : public ChannelAllocator
{
    Blocks blocks;
    void add_free_block(const Block &block);
    void remove_free_block(const Block &block);
public:
    virtual Blocks get_blocks() override { return blocks; }
    BestFitAllocator(std::uint32_t start_addr, std::uint32_t end_addr, Blocks pre_allocated_blocks = Blocks());
    virtual bool allocate(std::uint32_t size, std::uint32_t &addr) override; // return true if allocated, and update addr
    virtual void deallocate(std::uint32_t addr) override;
    virtual std::uint32_t get_capacity() override;
    virtual void clear_allocated_blocks();
};
}
