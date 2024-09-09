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
void limit_to_4d_reshape(graphlib::Graph *graph);
}
