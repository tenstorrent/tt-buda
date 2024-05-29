
// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace tt
{
namespace balancer
{

constexpr static int c_general_max_num_tiles_per_phase = 2048;

constexpr static int c_sequential_dram_io_threshold = 64 * 1024;

constexpr static int c_max_dram_pending_read_bytes = 52 * 1024;

constexpr static int c_default_dram_io_available_space = 100 * 1024;

}  // namespace balancer
}  // namespace tt