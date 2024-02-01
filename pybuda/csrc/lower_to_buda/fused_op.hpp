// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "graph_lib/node_types.hpp"
#include "lower_to_buda/common.hpp"

namespace tt
{

struct BudaFusedSubOp
{
    std::string name;
    std::string type;
    std::vector<std::string> inputs;
    BudaOpAttrs attrs;
    std::unordered_map<int, std::vector<graphlib::OpType>> tms;  // per operand
    std::string output;
    std::vector<std::uint32_t> popped_buffers;
    std::vector<std::uint32_t> popped_last_buffers;
    std::pair<std::uint32_t, std::uint32_t> block_shape;
    std::pair<std::uint32_t, std::uint32_t> ublock_shape;
    bool equivalent(const BudaFusedSubOp &other)
        const;  // return true if two sub ops are equivalent - i.e. same except for op names
};

struct BudaFusedOp
{
    std::uint32_t id;
    std::uint32_t input_count;
    std::vector<DataFormat> intermediate_buffer_df;
    std::vector<std::vector<BudaFusedSubOp>> schedule;

    bool equivalent(const BudaFusedOp &other)
        const;  // return true if two fused ops are equivalent - i.e. same except for sub op names
};

std::ostream &operator<<(std::ostream &os, BudaFusedOp const &op);

}  // namespace tt
