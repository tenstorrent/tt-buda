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
void bind_reshape_to_io(graphlib::Graph *graph);
}
