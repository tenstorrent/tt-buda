// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace tt::balancer
{
struct OutputHostTM
{
    int hstack_factor = 1;
    int vstack_factor = 1;
    bool row_major = true;
};


using OutputHostTMMap = std::unordered_map<std::string, OutputHostTM>;

}  // namespace tt::balancer
