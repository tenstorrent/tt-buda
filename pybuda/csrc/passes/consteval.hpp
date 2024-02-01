// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace tt::graphlib
{
class Graph;
class Node;
}

namespace tt::passes
{
void run_consteval_graph_pass(graphlib::Graph *graph);
}
