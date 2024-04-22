// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest-param-test.h>

#include "balancer/dram_read_estimator_internal.hpp"

#include "gtest/gtest.h"

namespace tt::test
{

using namespace balancer;
using namespace dram_read_estimator_internal;

TEST(TestDramReadEstimatorInternal, compute_dram_pipe_scatter_chunk_size_tiles)
{
    int scatter_chunk_size;

    scatter_chunk_size = compute_dram_pipe_scatter_chunk_size_tiles(
        120 /* scatter_gather_num_tiles */, 
        10 * 2080 /* unpacker_buffer_size_bytes */, 
        2080 /* tile_size */
    );
    EXPECT_EQ(scatter_chunk_size, 10);

    scatter_chunk_size = compute_dram_pipe_scatter_chunk_size_tiles(
        1200 /* scatter_gather_num_tiles */, 
        100 * 2080 /* unpacker_buffer_size_bytes */, 
        2080 /* tile_size */
    );
    EXPECT_EQ(scatter_chunk_size, 25);

    scatter_chunk_size = compute_dram_pipe_scatter_chunk_size_tiles(
        2 /* scatter_gather_num_tiles */, 
        10 * 2080 /* unpacker_buffer_size_bytes */, 
        2080 /* tile_size */
    );
    EXPECT_EQ(scatter_chunk_size, 2);
}

TEST(TestDramReadEstimatorInternal, compute_max_num_tiles_per_phase)
{
    int max_tiles_per_phase;

    max_tiles_per_phase = compute_max_num_tiles_per_phase(1 /* start_divisor */, 3000 /* root_tiles_per_input */);
    EXPECT_EQ(max_tiles_per_phase, 2048);

    max_tiles_per_phase = compute_max_num_tiles_per_phase(10 /* start_divisor */, 3000 /* root_tiles_per_input */);
    EXPECT_EQ(max_tiles_per_phase, 2040);

    max_tiles_per_phase = compute_max_num_tiles_per_phase(10 /* start_divisor */, 300 /* root_tiles_per_input */);
    EXPECT_EQ(max_tiles_per_phase, 1800);
}

TEST(TestDramReadEstimatorInternal, compute_dram_buf_read_chunk_size_tiles)
{
    int dram_read_chunk_size;

    dram_read_chunk_size = compute_dram_buf_read_chunk_size_tiles(
        3 /* scatter_gather_num_tiles */,
        2 /* kernel_clear_granularity */,
        48 /* tiles_to_transfer */,
        2080 /* tile_size */);
    EXPECT_EQ(dram_read_chunk_size, 24);

    dram_read_chunk_size = compute_dram_buf_read_chunk_size_tiles(
        1 /* scatter_gather_num_tiles */,
        5 /* kernel_clear_granularity */,
        400 /* tiles_to_transfer */,
        1120 /* tile_size */);
    EXPECT_EQ(dram_read_chunk_size, 5);

    dram_read_chunk_size = compute_dram_buf_read_chunk_size_tiles(
        12 /* scatter_gather_num_tiles */,
        6 /* kernel_clear_granularity */,
        768 /* tiles_to_transfer */,
        1120 /* tile_size */);
    EXPECT_EQ(dram_read_chunk_size, 12);
}

TEST(TestDramReadEstimatorInternal, compute_unpacker_stream_buffer_size_bytes)
{
    int unpacker_buffer_size_bytes = compute_unpacker_stream_buffer_size_bytes(
        2048 /* max_num_tiles_per_phase */,
        10 /* dram_read_chunk_size_tiles */,
        20 * 2080 /* unpacker_buffer_size_bytes */,
        2280 /* tiles_to_transfer */,
        1140 /* root_tiles_per_input */,
        2080 /* tile_size  */);
    EXPECT_EQ(unpacker_buffer_size_bytes, 41600);
}


} // namespace tt::test