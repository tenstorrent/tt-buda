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
bool erase_consecutive_reshape(graphlib::Graph *graph, bool commute_eltwise);
void bypass_nop_tms(graphlib::Graph *graph);
}
