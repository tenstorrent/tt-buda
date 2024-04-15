// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "balancer/bandwidth_bucket.hpp"

namespace tt
{
namespace balancer
{

// Estimates the bandwidth for a direct connection based on given parameters by using decision tree technique.
BandwidthBucket estimate_direct_connection(const int unpacker_buffer_size_bytes,
                                           const int kernel_clear_granularity,
                                           const int buf_space_available_ack_thr,
                                           const int epoch_tiles,
                                           const int tile_size,
                                           const int packer_buffer_size_bytes,
                                           const int packer_scatter_gather_num_tiles,
                                           const int packer_num_phases,
                                           const bool scatter_pack);

// Estimates the bandwidth for a gather connection based on given parameters by using decision tree technique.
BandwidthBucket estimate_gather_connection(const int epoch_tiles,
                                           const int tile_size,
                                           const int packer_scatter_gather_num_tiles,
                                           const int consumer_fanin);

// Estimates the bandwidth for a forked connection based on given parameters by using decision tree technique.
BandwidthBucket estimate_forked_connection(const int epoch_tiles,
                                           const int tile_size,
                                           const int packer_buffer_size_bytes,
                                           const int packer_scatter_gather_num_tiles,
                                           const int producer_fanout);

} // namespace balancer
} // namespace tt