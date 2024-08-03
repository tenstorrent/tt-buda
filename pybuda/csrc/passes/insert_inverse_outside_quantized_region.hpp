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
// Returns true if any transposes were moved outside quantized regions
bool insert_inverse_outside_quantized_region(graphlib::Graph *graph);
}