// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#define _LIBCPP_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM

namespace tt {

namespace graphlib {

// Dimension Index
constexpr int X = -1;
constexpr int Y = -2;
constexpr int Z = -3;
constexpr int W = -4;

using GraphId = std::int64_t;
using NodeId = std::int64_t;
using PortId = std::uint32_t;

enum NodeType
{
    kInput,
    kOutput,
    kPyOp,
    kBudaOp,
    kBudaNaryTM,
    kQueue,
};

enum NodeEpochType {
  Forward = 1,
  Backward = 2,
  Optimizer = 3
};

enum class UBlockOrder {
    R,
    C,
};

inline UBlockOrder flip_ublock_order(UBlockOrder o) { return (o == UBlockOrder::R) ? UBlockOrder::C : UBlockOrder::R; }

}  // namespace graphlib
}  // namespace tt
