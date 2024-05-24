// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "balancer/types.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "lower_to_buda/common.hpp"

using Graph = tt::graphlib::Graph;
using Node = tt::graphlib::Node;
using OpNode = tt::graphlib::OpNode;

namespace tt::balancer
{

std::pair<int, int> get_parallelization(Graph const* graph, OpNode const* node);
int get_execution_cycles(
    std::string const& arch_name,
    OpModel const& op_model,
    bool theoretical = false,
    std::vector<FusedSubOpModel> const& sub_op_models = {});

}  // namespace tt::balancer
