// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "graph_lib/node_types.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/graph.hpp"

namespace tt {

using Graph = graphlib::Graph;
using Node = graphlib::Node;

void lower_bwd_gather_ops(Graph *graph);

}
