// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <utility>

#include "graph_lib/defines.hpp"

namespace tt::graphlib
{
class Graph;
struct Edge;
class Node;

}  // namespace tt::graphlib

namespace tt
{
std::tuple<graphlib::Edge, graphlib::Node*, graphlib::Edge> insert_serialized_dram_queue_between_ops(
    graphlib::Graph* graph,
    std::string const& producer_name,
    std::string const& consumer_name,
    graphlib::PortId consumer_input_port_id,
    int num_entries = 2);
}
