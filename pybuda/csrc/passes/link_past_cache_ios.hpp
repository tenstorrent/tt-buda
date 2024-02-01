// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <map>
#include <string>

namespace tt::graphlib
{
class Graph;
}

namespace tt::passes
{
std::map<std::string, std::size_t> link_past_cache_ios(graphlib::Graph *graph);
}
