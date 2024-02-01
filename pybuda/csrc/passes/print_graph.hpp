// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
namespace tt::graphlib
{
class Graph;
}

namespace tt::passes
{
void print_graph(graphlib::Graph *graph, std::string stage);
}
