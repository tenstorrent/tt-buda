// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "graph_lib/shape.hpp"
namespace tt::graphlib
{
class Graph;
class OpNode;
class Node;
class Shape;
}
namespace tt::passes 
{
void set_tile_dim_for_nodes(graphlib::Graph *graph);
} // namespace tt:passes

