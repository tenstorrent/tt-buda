// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/dram_read_estimator_internal.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace tt
{
namespace balancer
{
namespace dram_read_estimator_internal
{

constexpr static int c_sequential_dram_io_threshold = 64 * 1024;
constexpr static int c_general_max_num_tiles_per_phase = 2048;

int compute_dram_pipe_scatter_chunk_size_tiles(
    const int scatter_gather_num_tiles, const int unpacker_buffer_size_bytes, const int tile_size)
{
    int dram_scatter_chunk_size = scatter_gather_num_tiles;
    const int unpacker_buffer_size_tiles = unpacker_buffer_size_bytes / tile_size;

    if (unpacker_buffer_size_tiles < dram_scatter_chunk_size)
    {
        dram_scatter_chunk_size = std::gcd(unpacker_buffer_size_tiles, scatter_gather_num_tiles);
    }

    int max_size_tiles = c_sequential_dram_io_threshold / tile_size;
    if (dram_scatter_chunk_size > max_size_tiles)
    {
        while (true)
        {
            if (dram_scatter_chunk_size % max_size_tiles == 0)
            {
                dram_scatter_chunk_size = max_size_tiles;
                break;
            }
            max_size_tiles--;
        }
    }

    return dram_scatter_chunk_size;
}

int compute_max_num_tiles_per_phase(
    const int start_divisor, const int root_tiles_per_input, const int tiles_to_send, const int subtree_common_divisor)
{
    int common_divisor = start_divisor;
    if (tiles_to_send > c_general_max_num_tiles_per_phase)
    {
        common_divisor = std::lcm(common_divisor, subtree_common_divisor);
    }

    if (root_tiles_per_input <= c_general_max_num_tiles_per_phase)
    {
        const int lcm_with_tiles_per_input = std::lcm(common_divisor, root_tiles_per_input);
        if (lcm_with_tiles_per_input <= c_general_max_num_tiles_per_phase)
        {
            common_divisor = lcm_with_tiles_per_input;
        }
    }

    return (c_general_max_num_tiles_per_phase / common_divisor) * common_divisor;
}

int compute_max_num_tiles_per_phase(const int start_divisor, const int root_tiles_per_input)
{
    return compute_max_num_tiles_per_phase(
        start_divisor, root_tiles_per_input, 1 /* tiles_to_send */, 1 /* subtree_common_divisor */);
}

int compute_dram_buf_read_chunk_size_tiles(
    const int scatter_chunk_size, const int kernel_clear_granularity, const int tiles_to_transfer, const int tile_size)
{
    const int phase_tiles = (c_general_max_num_tiles_per_phase / kernel_clear_granularity) * kernel_clear_granularity;

    int chunk_size = std::gcd(phase_tiles, tiles_to_transfer);
    chunk_size = std::lcm(chunk_size, scatter_chunk_size);

    return get_transfer_chunk_size_tiles(chunk_size, scatter_chunk_size, tile_size, 64 * 1024 - 1, 52 * 1024);
}

int get_transfer_chunk_size_tiles(
    int transfer_chunk_size_tiles,
    const int min_transfer_chunk_size_tiles,
    const int tile_size_bytes,
    const int max_transfer_size_tiles,
    const int max_transfer_size_bytes)
{
    if (!is_transfer_chunk_size_within_limits(
            transfer_chunk_size_tiles, tile_size_bytes, max_transfer_size_tiles, max_transfer_size_bytes))
    {
        for (int div = 2; div <= transfer_chunk_size_tiles; ++div)
        {
            if (transfer_chunk_size_tiles % div != 0)
            {
                continue;
            }
            int new_chunk_size = transfer_chunk_size_tiles / div;
            if (new_chunk_size % min_transfer_chunk_size_tiles == 0 &&
                is_transfer_chunk_size_within_limits(
                    new_chunk_size, tile_size_bytes, max_transfer_size_tiles, max_transfer_size_bytes))
            {
                transfer_chunk_size_tiles = new_chunk_size;
                break;
            }
        }
    }
    if (!is_transfer_chunk_size_within_limits(
            transfer_chunk_size_tiles, tile_size_bytes, max_transfer_size_tiles, max_transfer_size_bytes))
    {
        transfer_chunk_size_tiles = min_transfer_chunk_size_tiles;
    }

    return transfer_chunk_size_tiles;
}

bool is_transfer_chunk_size_within_limits(
    const int transfer_chunk_size_tiles,
    const int tile_size_bytes,
    const int max_transfer_size_tiles,
    const int max_transfer_size_bytes)
{
    return transfer_chunk_size_tiles <= max_transfer_size_tiles &&
           transfer_chunk_size_tiles * tile_size_bytes <= max_transfer_size_bytes;
}

int compute_unpacker_stream_buffer_size_bytes(
    int max_num_tiles_per_phase,
    const int dram_read_chunk_size_tiles,
    const int unpacker_buffer_size_bytes,
    const int tiles_to_transfer,
    const int root_tiles_per_input,
    const int tile_size)
{
    const int merged_stream_buffer_size_bytes = compute_merged_dram_unpacker_stream_buffer_size_bytes(
        dram_read_chunk_size_tiles, unpacker_buffer_size_bytes, tile_size);

    const bool can_merge_streams = merged_stream_buffer_size_bytes <= std::max(100 * 1024, unpacker_buffer_size_bytes);

    int result = unpacker_buffer_size_bytes;
    if (can_merge_streams)
    {
        if (tiles_to_transfer > c_general_max_num_tiles_per_phase)
        {
            max_num_tiles_per_phase = compute_max_num_tiles_per_phase(
                unpacker_buffer_size_bytes / tile_size,
                root_tiles_per_input,
                1 /* tiles_to_send */,
                1 /* subtree_common_divisor */);
        }

        return scale_up_dram_receiving_stream(merged_stream_buffer_size_bytes, max_num_tiles_per_phase, tile_size);
    }

    return result;
}

int compute_unpacker_stream_buffer_size_bytes(
    int max_num_tiles_per_phase, const int dram_read_chunk_size_tiles, const int tile_size)
{
    const int base_buffer_size = dram_read_chunk_size_tiles * tile_size;

    return scale_up_dram_receiving_stream(base_buffer_size, max_num_tiles_per_phase, tile_size);
}

int compute_merged_dram_unpacker_stream_buffer_size_bytes(
    const int dram_read_chunk_size_tiles, const int unpacker_buffer_size_bytes, const int tile_size)
{
    const int dram_read_chunk_size_bytes = dram_read_chunk_size_tiles * tile_size;

    return std::lcm(dram_read_chunk_size_bytes, unpacker_buffer_size_bytes);
}

int scale_up_dram_receiving_stream(
    const int base_buffer_size_bytes, const int max_num_tiles_per_phase, const int tile_size)
{
    const int max_num_tiles_per_phase_bytes = max_num_tiles_per_phase * tile_size;
    int scale_factor = 1;

    const int min_buffer_size = 52 * 1024;
    while ((scale_factor + 1) * base_buffer_size_bytes < min_buffer_size)
    {
        if (scale_factor * base_buffer_size_bytes >= min_buffer_size / 2 &&
            max_num_tiles_per_phase_bytes % (scale_factor * base_buffer_size_bytes) == 0)
        {
            break;
        }
        ++scale_factor;
    }

    if (scale_factor == 1 && (2 * base_buffer_size_bytes <= 100 * 1024) &&
        max_num_tiles_per_phase_bytes % (2 * base_buffer_size_bytes) == 0)
    {
        scale_factor = 2;
    }

    return scale_factor * base_buffer_size_bytes;
}

}  // namespace dram_read_estimator_internal
}  // namespace balancer
}  // namespace tt
