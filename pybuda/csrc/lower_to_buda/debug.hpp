// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace tt
{

struct DeviceConfig;

namespace balancer
{
struct OpModel;
}

void to_debug_info(std::ostream& os, tt::DeviceConfig const& device_config);
void to_debug_info(
    std::ostream& os,
    std::string const& name,
    tt::balancer::OpModel const& op_model,
    std::string const& arch_name,
    std::vector<std::size_t> const& input_dram_io_buf_size_tiles);
}  // namespace tt
