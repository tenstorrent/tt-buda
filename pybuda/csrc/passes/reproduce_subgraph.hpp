// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <pybind11/pybind11.h>

#include "balancer/balancer.hpp"

namespace py = pybind11;

namespace tt::graphlib
{
class Graph;
}

namespace tt::passes
{
// 
void reproduce_subgraph(
    graphlib::Graph *graph,
    std::string input_name,
    std::string output_name,
    std::unordered_map<std::string, py::object> intermediates,
    std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    placer::PlacerSolution *placer_solution);
}
