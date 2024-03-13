// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <iostream>

#include "lower_to_buda/queue.hpp"
#include "lower_to_buda/op.hpp"

namespace tt {

std::string BudaQueue::as_string(int padded_name_length) const
{
    std::stringstream ss;
    ss << name << ": ";
    for (int i = name.length(); i < padded_name_length + 1; i++) ss << " ";

    ss << "{";
    ss << "input: " << input_name << ", ";
    if (alias != "") ss << "alias: " << alias << ", ";
    ss << "type: " << (memory_access == "RAM" ? "ram" : "queue") << ", ";
    ss << "entries: " << entries << ", ";
    ss << "grid_size: [" << dims.grid_r << ", " << dims.grid_c << "], ";
    ss << blocks << ", ";
    switch (ublock_order)
    {
        case graphlib::UBlockOrder::R: ss << "ublock_order: r, "; break;
        case graphlib::UBlockOrder::C: ss << "ublock_order: c, "; break;
    }

    ss << "tile_dim: " << tile_dim_ << ", ";
    ss << "df: " << data_format << ", ";

    if (layout != BudaQueueLayout::Tilized)
        ss << "layout: " << layout << ", ";

    ss << "target_device: ";
    if (target_devices.size() == 1)
    {
        ss << target_devices[0] << ", ";
    }
    else
    {
        for (size_t i = 0; i < target_devices.size(); i++)
        {
            ss << ((i == 0) ? "[" : ", ") << target_devices[i];
        }
        ss << "], ";
    }
    ss << "loc: " << loc;

    if (loc == BudaQueueLocation::DRAM) {
        TT_ASSERT(dram_loc.size() > 0);
        ss << ", dram: [" << dram_loc[0];
        for (std::size_t i=1; i < dram_loc.size(); i++)
            ss << ", " << dram_loc[i];
        ss << "]";
    } else if (loc == BudaQueueLocation::HOST) {
        TT_ASSERT(host_loc.size() > 0);
        ss << ", host: [" << host_loc[0];
        for (std::size_t i = 1; i < host_loc.size(); i++) ss << ", " << host_loc[i];
        ss << "]";
    }

    ss << "}";

    return ss.str();
}

std::ostream &operator<<(std::ostream &os, BudaQueue const &q) 
{
    return os << q.as_string();
}

std::ostream &operator<<(std::ostream &os, BudaQueueDramLoc const &l)
{
    return os << "[" << l.dram_channel << ", 0x" << std::hex << l.dram_address << std::dec << "]";
}

std::ostream &operator<<(std::ostream &os, BudaQueueHostLoc const &l)
{
    return os << "[" << l.host_channel << ", 0x" << std::hex << l.host_address << std::dec << "]";
}

std::ostream &operator<<(std::ostream &os, BudaQueueLocation const &l)
{
    return os << ((l == BudaQueueLocation::DRAM) ? "dram" : "host");
}

std::ostream &operator<<(std::ostream &os, BudaQueueLayout const &l)
{
    switch (l)
    {
        case BudaQueueLayout::Tilized: os << "tilized"; break;
        case BudaQueueLayout::Flat: os << "flat"; break;
    }

    return os;
}

} // namespace tt
