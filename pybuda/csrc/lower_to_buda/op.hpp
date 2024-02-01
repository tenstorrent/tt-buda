// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <variant>
#include <vector>

#include "graph_lib/node_types.hpp"
#include "lower_to_buda/comment.hpp"
#include "lower_to_buda/common.hpp"
#include "lower_to_buda/queue.hpp"

namespace tt {

struct BudaOpGrid {
    std::uint32_t grid_loc_r, grid_loc_c;
    std::uint32_t grid_size_r, grid_size_c;
};

struct BudaNaryTM;

using BudaOperand = std::variant<BudaName, BudaNaryTM>;

struct BudaOp {
    std::string name;
    std::string type;
    BudaOpGrid grid;
    std::vector<BudaOperand> inputs;
    std::vector<std::pair<std::string, std::string>> forked_dram_inputs;
    std::vector<DataFormat> input_data_formats;
    std::vector<std::uint32_t> input_buf_min_size_tiles;
    std::vector<std::size_t> input_dram_io_buf_size_tiles;
    DataFormat output_data_format;
    DataFormat intermediate_data_format;
    DataFormat accumulate_data_format;
    MathFidelity fidelity;
    bool untilize_output;
    BudaBlocks blocks;
    graphlib::UBlockOrder ublock_order = graphlib::UBlockOrder::R;
    int buf_size_mb;
    int overlay_size = 0;  // Op-level override for overlay blob size in Bytes, default is 65536 (64 kB)
    std::unordered_map<int, std::vector<graphlib::OpType>> tms; // per operand
    BudaOpAttrs attrs;
    bool gradient_op;
    bool grid_transpose;
    Comment debug_info;
    TileDim tile_dim;

};

struct BudaNaryTM
{
    std::string name;
    std::string type;
    std::vector<BudaOperand> inputs;
};

std::ostream &operator<<(std::ostream &os, BudaOp const &op);
std::ostream &operator<<(std::ostream &os, BudaNaryTM const &tm);
std::ostream &operator<<(std::ostream &os, BudaOperand const &operand);
std::ostream &operator<<(std::ostream &os, std::vector<BudaOperand> const &operands);
std::ostream &operator<<(std::ostream &os, BudaOpGrid const &g);
std::ostream &operator<<(std::ostream &os, std::vector<DataFormat> const &dfs);
std::ostream &operator<<(std::ostream &os, TileDim const tile_dim);
std::ostream &operator<<(std::ostream &os, std::vector<std::pair<std::string, std::string>> const &forked_dram_inputs);
} // namespace tt



