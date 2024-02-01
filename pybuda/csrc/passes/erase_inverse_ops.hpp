// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace tt::graphlib
{
class Graph;
class OpNode;
class Shape;
}

namespace tt::passes
{
// Returns true if any inverse ops were erased
bool erase_inverse_ops(graphlib::Graph *graph);
}
