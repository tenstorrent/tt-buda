// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace tt
{
namespace balancer
{
namespace dram_read_estimator_internal
{

int compute_dram_pipe_scatter_chunk_size_tiles(
    const int scatter_gather_num_tiles, const int unpacker_buffer_size_bytes, const int tile_size);

int compute_max_num_tiles_per_phase(const int start_divisor, const int root_tiles_per_input);

int compute_max_num_tiles_per_phase(
    const int start_divisor, const int root_tiles_per_input, const int tiles_to_send, const int subtree_common_divisor);

int compute_dram_buf_read_chunk_size_tiles(
    const int scatter_chunk_size, const int kernel_clear_granularity, const int tiles_to_transfer, const int tile_size);

int get_transfer_chunk_size_tiles(
    int transfer_chunk_size_tiles,
    const int min_transfer_chunk_size_tiles,
    const int tile_size_bytes,
    const int max_transfer_size_tiles,
    const int max_transfer_size_bytes);

bool is_transfer_chunk_size_within_limits(
    const int transfer_chunk_size_tiles,
    const int tile_size_bytes,
    const int max_transfer_size_tiles,
    const int max_transfer_size_bytes);

int compute_unpacker_stream_buffer_size_bytes(
    int max_num_tiles_per_phase,
    const int dram_read_chunk_size_tiles,
    const int unpacker_buffer_size_bytes,
    const int tiles_to_transfer,
    const int root_tiles_per_input,
    const int tile_size);

int compute_unpacker_stream_buffer_size_bytes(
    int max_num_tiles_per_phase, const int dram_read_chunk_size_tiles, const int tile_size);

int compute_merged_dram_unpacker_stream_buffer_size_bytes(
    const int dram_read_chunk_size_tiles, const int unpacker_buffer_size_bytes, const int tile_size);

int scale_up_dram_receiving_stream(
    const int base_buffer_size_bytes, const int max_num_tiles_per_phase, const int tile_size);

}  // namespace dram_read_estimator_internal
}  // namespace balancer
}  // namespace tt
