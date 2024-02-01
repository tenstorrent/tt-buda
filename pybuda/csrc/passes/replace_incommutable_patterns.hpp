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
// Returns true if any patterns were replaced with something commutable
bool replace_incommutable_patterns(graphlib::Graph *graph);
}