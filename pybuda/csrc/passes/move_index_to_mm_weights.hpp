// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <map>
namespace tt::graphlib
{
class Graph;
}

namespace tt::passes
{
void move_index_to_mm_weights(graphlib::Graph *graph);
}
