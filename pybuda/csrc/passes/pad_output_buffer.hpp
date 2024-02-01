// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "backend_api/device_config.hpp"

namespace tt::graphlib
{
class Graph;
class OpNode;
class Node;
class Shape;
}
namespace tt::passes 
{
void pad_output_buffer(graphlib::Graph *graph, const DeviceConfig &device_config);
} // namespace tt:passes