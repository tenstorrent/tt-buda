// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace tt::graphlib
{
class Graph;
class OpNode;
class Node;
class Shape;
}
namespace tt::passes 
{
bool squeeze_to_reshape(graphlib::Graph *graph);
} // namespace tt:passes