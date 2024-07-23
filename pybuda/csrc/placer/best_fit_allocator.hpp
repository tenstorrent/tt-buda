// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "placer/dram_allocator.hpp"

namespace tt::placer {

class BestFitAllocator : public ChannelAllocator
{
    Blocks blocks;
    void add_free_block(const Block &block);
    void remove_free_block(const Block &block);
public:
    virtual Blocks get_blocks() override { return blocks; }
    BestFitAllocator(std::size_t start_addr, std::size_t end_addr, Blocks pre_allocated_blocks = Blocks());
    virtual bool allocate(std::size_t size, std::size_t &addr) override; // return true if allocated, and update addr
    virtual void deallocate(std::size_t addr) override;
    virtual std::size_t get_capacity() override;
    virtual void clear_allocated_blocks() override;
};
}
