// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placer/best_fit_allocator.hpp"

namespace tt::placer {

BestFitAllocator::BestFitAllocator(std::uint32_t start_addr, std::uint32_t end_addr, Blocks pre_allocated_blocks) : ChannelAllocator()
{
    if (pre_allocated_blocks.free_blocks_start.size() > 0) {
        blocks = pre_allocated_blocks;
    } else if (start_addr < end_addr) {
        // end_addr is the last available address. block includes end_addr
        add_free_block(Block{start_addr, end_addr - start_addr + 1});
    }
}

void BestFitAllocator::add_free_block(const Block &block) 
{
    blocks.free_blocks_start[block.addr] = block;
    blocks.free_blocks_end[block.addr + block.size] = block;
}

std::uint32_t BestFitAllocator::get_capacity()
{
    std::uint32_t capacity = 0;
    for (auto free_block : blocks.free_blocks_start) {
        capacity += free_block.second.size;
    }
    return capacity;
}

void BestFitAllocator::remove_free_block(const Block &block) 
{
    std::uint32_t end = block.addr + block.size;
    blocks.free_blocks_start.erase(block.addr);
    blocks.free_blocks_end.erase(end);
}

bool BestFitAllocator::allocate(std::uint32_t size, std::uint32_t &addr)
{
    // Find the free block with the closest >= size
    Block closest_block;
    std::uint32_t diff = UINT32_MAX;
    for (auto it = blocks.free_blocks_start.rbegin(); it != blocks.free_blocks_start.rend(); it++) 
    {
        if (it->second.size >= size) 
        {
            auto my_diff = it->second.size - size;
            if (my_diff < diff) {
                diff = my_diff;
                closest_block = it->second;
                if (diff == 0)
                    break;
            }
        }
    }

    if (diff == UINT32_MAX)
        return false;

    addr = closest_block.addr;
    // Since we allocate new block from right to left, end of the free block will be the end of our new allocated block
    addr = closest_block.addr + closest_block.size - size;
    remove_free_block(closest_block);
    if (diff == 0) {
        blocks.allocated_blocks[addr] = closest_block;
    } else {
        blocks.allocated_blocks[addr] = Block{addr, size};
        add_free_block(Block{closest_block.addr, diff});
    }

    return true;
}

void BestFitAllocator::deallocate(std::uint32_t addr) 
{
    //return;
    auto it = blocks.allocated_blocks.find(addr);

    if (it == blocks.allocated_blocks.end())
        TT_THROW("Trying to deallocate addr that hasn't been allocated");

    // Find previous and next block to merge with, if any
    Block freed_block = it->second;
    auto next = blocks.free_blocks_start.find(addr + it->second.size);
    if (next != blocks.free_blocks_start.end())
    {
        freed_block.size += next->second.size;
        remove_free_block(next->second);
    }

    auto prev = blocks.free_blocks_end.find(addr);
    if (prev != blocks.free_blocks_end.end()) {
        freed_block.addr = prev->second.addr;
        freed_block.size += prev->second.size;
        remove_free_block(prev->second);
    }
            
    add_free_block(freed_block);
    blocks.allocated_blocks.erase(it);
}

// Deallocates all allocated blocks from allocator, and frees the space
void BestFitAllocator::clear_allocated_blocks()
{
    Blocks blocks = get_blocks();
    for (const auto& address_block_pair : blocks.allocated_blocks)
    {
        std::uint32_t start_address = address_block_pair.first;
        deallocate(start_address);
    }
}

}
