// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

namespace tt
{
// Ripped out of boost for std::size_t so as to not pull in
// bulky boost dependencies. Place this fcn in a more suitable place
inline void hash_combine(std::size_t& seed, std::size_t value)
{
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
}  // namespace tt
