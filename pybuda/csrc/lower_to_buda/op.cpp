// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "lower_to_buda/op.hpp"
#include "lower_to_buda/common.hpp"

namespace tt {

static bool op_uses_grid_location(BudaOp const &op) {
    return op.type != "ethernet_datacopy";
}

std::ostream &operator<<(std::ostream &os, BudaOp const &op) {
    os << op.name << ": {";

    os << "type: " << op.type << ", ";
    if (op_uses_grid_location(op))
    {
        os << op.grid << ", ";
    } else {
        os << "grid_size: [" << op.grid.grid_size_r << ", " << op.grid.grid_size_c << "],";
    }
    os << "inputs: "<< op.inputs;

    os <<op.forked_dram_inputs;

    if (op.gradient_op)
        os << ", gradient_op: true";

    if (op.untilize_output)
        os << ", untilize_output: true";
    
    if (op.grid_transpose)
        os << ", grid_transpose: true";

    if (op.overlay_size > 0)
    {
        os << ", overlay_size: " << op.overlay_size;
    }

    os << ","
       << std::endl << "         "
       << op.blocks;

    // Next, write out tile dim
    os << ", tile_dim: " << op.tile_dim;
    os << ", buf_size_mb: " << op.buf_size_mb;

    if (op.input_buf_min_size_tiles.size() > 0)
    {
        bool first = true;
        os << ", input_buf_min_size_tiles: [";
        for (std::uint32_t tiles : op.input_buf_min_size_tiles)
        {
            if (!first) os << ", ";
            first = false;
            os << tiles;
        }
        os << "]";
    }

    if (not op.input_dram_io_buf_size_tiles.empty())
    {
        bool first = true;
        os << ", input_dram_io_buf_size_tiles: [";
        for (std::size_t size_tiles : op.input_dram_io_buf_size_tiles)
        {
            if (!first)
                os << ", ";
            first = false;
            os << size_tiles;
        }
        os << "]";
    }

    switch (op.ublock_order) {
        case graphlib::UBlockOrder::R: os << ", ublock_order: r"; break;
        case graphlib::UBlockOrder::C: os << ", ublock_order: c"; break;
    }

    os << ", "
       << "in_df: " << op.input_data_formats << ", "
       << "out_df: " << op.output_data_format << ", "
       << "intermed_df: " << op.intermediate_data_format << ", "
       << "acc_df: " << op.accumulate_data_format << ", " 
       << "math_fidelity: " << op.fidelity;

    if (op.tms.size() > 0) {

        os << ","
           << std::endl << "         ";

        // Next, we write out the TMs
        bool first = true;
        for (auto t : op.tms) {

            if (!first) os << ", ";
            first = false;

            std::vector<graphlib::OpType> unpaddings;

            for (auto it = t.second.begin(); it != t.second.end(); ++it) {

                if ((*it).op == "buda_unpad") {
                    unpaddings.push_back(*it);
                    continue;
                }

            }

            // First, we write out the padding unpads
            if (unpaddings.size() == 1) {

                // Create unpadding pad TM in the netlist
                os << "input_" << t.first << "_unpad: " 
                    // Write out number of tiles for R dimension
                   << "{rt: " << unpaddings[0].buda_attrs["rt"]
                    // Write out number of tiles for C dimension
                   << ", ct: " << unpaddings[0].buda_attrs["ct"] << "}";

                if (t.second.size() - unpaddings.size() > 0)
                    os << ", ";

            }

            if (t.second.size() - unpaddings.size() > 0) {
                
                os << "input_" << t.first << "_tms: [";

                bool first_tm = true;
                for (auto &tm : t.second) {

                    // Unpadding operations/atrributes are in rank of TMs, 
                    // they shouldn't be listed inside TMs' attributes
                    if (tm.op == "buda_unpad")
                        continue;

                    // If the operation was pad or unpad, this flag should be reset after that check
                    // otherwise we can have hanging comma in the netlist, in the list of tms
                    if (!first_tm) os << ", ";
                    first_tm = false;

                    // Unpadding operations/atrributes are in rank of TMs,
                    // they shouldn't be listed inside TMs' attributes
                    if (tm.op == "buda_pad") {
                        os << "pad: [" << tm.buda_attrs["rt"] << ", " << tm.buda_attrs["ct"] << ", " << tm.buda_attrs["pad_value"] << "]";
                        continue;
                    }

                    if (tm.op == "broadcast") {
                        // User-friendly dims
                        os << "broadcast: {";
                        assert(tm.attr.size() <= 3);
                        switch(std::get<int>(tm.attr[0])) {
                            case 0: throw std::runtime_error("Broadcast of W not supported");
                            case 1: os << "z";
                                os << ": " << std::get<int>(tm.attr[1]) << "}";
                                break;
                            case 2: os << "r";
                                os << ": " << std::get<int>(tm.attr[1]) << "}";
                                break;
                            case 3: os << "c";
                                os << ": " << std::get<int>(tm.attr[1]) << "}";
                                break;
                        }
                        continue;
                    }

                    if (tm.op == "transpose") {
                        os << "transpose";
                        continue;
                    }

                    if (tm.op == "select") {
                        // User-friendly dims
                        os << "select: {";
                        assert(tm.attr.size() == 4);
                        os << "range: [" << tm.buda_attrs["index"] << ", " << tm.buda_attrs["length"]
                        << "], stride: " << tm.buda_attrs["stride"] << "}";
                        continue;
                    }

                    if (tm.op == "gather") {
                        // User-friendly dims
                        os << "gather: {";
                        assert(tm.attr.size() == 5);
                        os << "range: [" << tm.buda_attrs["index"] << ", " << tm.buda_attrs["length"]
                        << "], stride: " << tm.buda_attrs["stride"] << ", size: " << tm.buda_attrs["size"] << "}";
                        continue;
                    }

                    if (tm.op == "hslice" or tm.op == "vslice" or tm.op == "hstack" or tm.op == "vstack")
                    {
                        TT_ASSERT(tm.attr.size() == 1);
                        os << tm.op << ": " << std::get<int>(tm.attr[0]);
                        continue;
                    }

                    os << tm.op;
                    if (tm.attr.size() > 0) {
                        os << ": {";
                        bool first_param = true;
                        for (graphlib::OpType::Attr param : tm.attr) {
                            if (!first_param) os << ", ";
                            first_param = false;
                            // os << param;
                            os << std::get<int>(param);
                        }
                        os << "}";
                    }
                }

                os << "]";

            }

        }

    }

    if (!op.attrs.empty()) {
        os << "," << std::endl << "         attributes: {";
        bool first = true;
        for (auto [key, value] : op.attrs) {
            if (!first) {
                os << ", ";
            }
            first = false;
            os << key << ": " << value;
        }
        os << "}";
    }

    os << "}";


    return os;

}

std::ostream &operator<<(std::ostream &os, std::vector<std::pair<std::string, std::string>> const &forked_dram_inputs)
{
    if (!forked_dram_inputs.empty())
    {
        bool first = true;
        os << ", forked_dram_inputs: [";
        for (auto &forked_input : forked_dram_inputs)
        {
            if (!first)
                os << ", ";
            first = false;
            os << forked_input.first << ": " << forked_input.second;
        }
        os << "]";
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, BudaNaryTM const &tm)
{
    os << tm.type << ": " << tm.inputs;
    return os;
}

std::ostream &operator<<(std::ostream &os, BudaOperand const &operand)
{
    if (const BudaName *n = std::get_if<BudaName>(&operand))
    {
        os << *n;
    }
    else if (const BudaNaryTM *c = std::get_if<BudaNaryTM>(&operand))
    {
        os << *c;
    }
    else
    {
        TT_ASSERT(false, "Unhandled variant type for BudaOperand");
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, std::vector<BudaOperand> const &operands)
{
    bool first = true;
    os << "[";
    for (const BudaOperand &operand : operands)
    {
        if (!first)
            os << ", ";
        os << operand;
        first = false;
    }
    return os << "]";
}

std::ostream &operator<<(std::ostream &os, BudaOpGrid const &g) {
    os << "grid_loc: [" << g.grid_loc_r << ", " << g.grid_loc_c << "], ";
    os << "grid_size: [" << g.grid_size_r << ", " << g.grid_size_c << "]";
    return os;

}

std::ostream &operator<<(std::ostream &os, std::vector<DataFormat> const &dfs)
{
    if (dfs.size() == 0) {
        return os << "[]";
    }

    os << "[" << dfs[0];
    for (std::size_t i=1; i < dfs.size(); i++)
        os << ", " << dfs[i];
    return os << "]";
}

std::ostream &operator<<(std::ostream &os, TileDim const tile_dim) {

    switch(tile_dim)
    {
        case TileDim::Dim32x32:
            os << "[32, 32]";
            return os;
        case TileDim::Dim16x32:
            os << "[16, 32]";
            return os;
        case TileDim::Dim32x16:
            os << "[32, 16]";
            return os;
        case TileDim::Dim8x32:
            os << "[8, 32]";
            return os;
        case TileDim::Dim4x32:
            os << "[4, 32]";
            return os;
        case TileDim::Dim2x32:
            os << "[2, 32]";
            return os;
        case TileDim::Dim1x32:
            os << "[1, 32]";
            return os;  
        default:
            TT_ASSERT(false, "Invalid tile dim");
    }
    return os;
}

}



