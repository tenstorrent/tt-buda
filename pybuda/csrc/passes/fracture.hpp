// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace tt::graphlib
{
class Graph;
}

namespace tt::passes
{
inline constexpr int k_dim = std::numeric_limits<int>::min();
using FractureGroup = std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>>>;
using FractureChipIds = std::unordered_map<std::string, std::vector<int>>;
using FractureGroups = std::vector<std::tuple<FractureGroup, FractureChipIds>>;
using FractureChipIdAssignments = std::unordered_map<std::string, int>;

FractureChipIdAssignments fracture(graphlib::Graph* graph, FractureGroups const& fracture_groups);
}  // namespace tt::passes
