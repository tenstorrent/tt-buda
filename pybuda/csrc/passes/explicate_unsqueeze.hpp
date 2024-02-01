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
void explicate_unsqueeze(graphlib::Graph *graph);
void hoist_unsqueeze_squeeze_to_reshape(graphlib::Graph *graph);
}
