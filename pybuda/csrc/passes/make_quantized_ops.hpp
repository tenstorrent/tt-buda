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
bool make_quantized_ops(graphlib::Graph *graph);
void separate_conv2d_bias(graphlib::Graph *graph);
}