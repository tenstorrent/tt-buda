// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace tt::graphlib
{
class Graph;
}

namespace tt::passes
{
// Returns true if any inverse ops were erased
void move_select_after_matmul_optional(graphlib::Graph *graph);
}
