// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "lower_to_buda/fused_op.hpp"

#include <iostream>

namespace tt
{

std::ostream &operator<<(std::ostream &os, BudaFusedOp const &op)
{
    os << op.id << ": " << std::endl;

    const std::string indent = "    ";
    os << indent << "inputs: " << op.input_count << std::endl;

    // Dropped support for intermed formats at the moment. We'll provide a count only.
    /*os << indent << "intermed_df: [";

    bool first = true;
    for (DataFormat df : op.intermediate_buffer_df)
    {
        if (!first) os << ", ";
        first = false;
        os << df;
    }
    os << "]" << std::endl;*/
    os << indent << "intermediates: " << op.intermediate_buffer_df.size() << std::endl;

    os << indent << "schedules: " << std::endl;

    auto print_op = [&os](auto op, auto indent, int sch_id)
    {
        os << indent << "    - " << op.name << "." << sch_id << ": { type: " << op.type << ", inputs: [";

        bool first = true;
        for (auto i : op.inputs)
        {
            if (!first)
                os << ", ";
            first = false;
            os << i;
        }

        os << "]";

        for (auto t : op.tms)
        {
            if (t.second.size() == 0)
                continue;

            os << ", input_" << t.first << "_tms: [";

            bool first_tm = true;
            for (auto &tm : t.second)
            {
                if (!first_tm)
                    os << ", ";
                first_tm = false;

                TT_ASSERT(tm.op == "broadcast" || tm.op == "tile_broadcast");  // only supported kind here

                if (tm.op == "broadcast")
                {
                    // User-friendly dims
                    os << "broadcast: {";
                    assert(tm.attr.size() == 2);
                    switch (std::get<int>(tm.attr[0]))
                    {
                        case 0: throw std::runtime_error("Broadcast of W not supported");
                        case 1:
                            os << "z";
                            os << ": " << std::get<int>(tm.attr[1]) << "}";
                            break;
                        case 2:
                            os << "r";
                            os << ": " << std::get<int>(tm.attr[1]) << "}";
                            break;
                        case 3:
                            os << "c";
                            os << ": " << std::get<int>(tm.attr[1]) << "}";
                            break;
                    }
                    continue;
                }
                else if (tm.op == "tile_broadcast")
                {
                    // User-friendly dims
                    os << "tile_broadcast: ";
                    assert(tm.attr.size() == 1);
                    switch (std::get<int>(tm.attr[0]))
                    {
                        case 0:
                        case 1: throw std::runtime_error("Tile broadcast of W/Z not supported");
                        case 2: os << "r"; break;
                        case 3: os << "c"; break;
                    }
                    continue;
                }
            }
            os << "]";
        }

        if (!op.attrs.empty())
        {
            os << ", attributes: {";
            bool first = true;
            for (auto [key, value] : op.attrs)
            {
                if (!first)
                {
                    os << ", ";
                }
                first = false;
                os << key << ": " << value;
            }
            os << "}";
        }

        if (op.popped_buffers.size() > 0)
        {
            os << ", pop: [";
            bool first = true;
            for (std::uint32_t buf : op.popped_buffers)
            {
                if (!first)
                {
                    os << ", ";
                }
                first = false;
                os << "intermed" << buf;
            }
            os << "]";
        }

        if (op.popped_last_buffers.size() > 0)
        {
            os << ", pop_last: [";
            bool first = true;
            for (std::uint32_t buf : op.popped_last_buffers)
            {
                if (!first)
                {
                    os << ", ";
                }
                first = false;
                os << "intermed" << buf;
            }
            os << "]";
        }

        os << ", mblock: [" << op.block_shape.first << ", " << op.block_shape.second << "]";
        os << ", ublock: [" << op.ublock_shape.first << ", " << op.ublock_shape.second << "]";

        os << ", output: " << op.output << "}" << std::endl;
    };

    int sch_id = 0;
    for (auto sch : op.schedule)
    {
        os << indent << "  -" << std::endl;
        for (auto op : sch)
        {
            print_op(op, indent, sch_id);
        }
        sch_id++;
    }
    return os;
}

bool BudaFusedSubOp::equivalent(const BudaFusedSubOp &other) const
{
    if (type != other.type)
        return false;
    if (output != other.output)
        return false;

    if (inputs.size() != other.inputs.size())
        return false;
    for (std::size_t i = 0; i < inputs.size(); i++)
        if (inputs[i] != other.inputs[i])
            return false;

    if (attrs != other.attrs)
        return false;

    if (popped_buffers != other.popped_buffers)
        return false;

    for (auto &[index, input_tms] : tms)
    {
        if (other.tms.count(index) == 0)
            return false;
        if (input_tms.size() != other.tms.at(index).size())
            return false;

        for (std::size_t i = 0; i < input_tms.size(); i++)
        {
            const graphlib::OpType &us = input_tms[i];
            const graphlib::OpType &them = other.tms.at(index).at(i);
            if (us.op != them.op)
                return false;
            if (us.attr != them.attr)
                return false;
        }
    }

    if (block_shape != other.block_shape)
        return false;

    if (ublock_shape != other.ublock_shape)
        return false;

    return true;
}

// return true if two fused ops are equivalent - i.e. same except for sub op name
bool BudaFusedOp::equivalent(const BudaFusedOp &other) const
{
    if (input_count != other.input_count)
        return false;
    if (intermediate_buffer_df.size() != other.intermediate_buffer_df.size())
        return false;
    for (std::size_t i = 0; i < intermediate_buffer_df.size(); i++)
        if (intermediate_buffer_df[i] != other.intermediate_buffer_df[i])
            return false;

    if (schedule.size() != other.schedule.size())
        return false;
    for (std::size_t schedule_index = 0; schedule_index < schedule.size(); schedule_index++)
    {
        if (schedule[schedule_index].size() != other.schedule[schedule_index].size())
            return false;

        for (std::size_t i = 0; i < schedule[schedule_index].size(); i++)
            if (!schedule[schedule_index][i].equivalent(other.schedule[schedule_index][i]))
                return false;
    }
    return true;
}

}  // namespace tt

