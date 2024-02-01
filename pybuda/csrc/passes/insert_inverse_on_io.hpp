// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <vector>
#pragma once

namespace tt::graphlib
{
class Graph;
class OpNode;
class Shape;
struct Edge;
}

namespace tt::passes 
{
// Returns true if any ops were commuted to input
bool insert_inverse_on_inputs(graphlib::Graph *graph);
bool insert_inverse_on_outputs(graphlib::Graph *graph);
bool insert_inverse_on_downstream_tms(graphlib::Graph *graph);

std::vector<std::pair<graphlib::Edge, std::pair<graphlib::Shape, graphlib::Shape>>> all_edges_to_input_nodes_commutable(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    graphlib::Shape commute_shape,
    graphlib::Shape clone_shape,
    graphlib::OpNode *from = nullptr,
    graphlib::OpNode *previous_op = nullptr);

void add_inverse_to_input_edges(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    std::vector<std::pair<graphlib::Edge, std::pair<graphlib::Shape, graphlib::Shape>>> input_edges_and_shapes);
}
