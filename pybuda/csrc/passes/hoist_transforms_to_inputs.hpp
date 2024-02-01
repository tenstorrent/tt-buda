// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "graph_lib/graph.hpp"

namespace tt::passes
{
    void hoist_transforms_to_inputs(tt::graphlib::Graph *graph);
}
