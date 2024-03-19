// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <utility>

#include "graph_lib/defines.hpp"

struct PostPlacerConfig;

namespace tt::graphlib
{
class Graph;
struct Edge;
class Node;

}  // namespace tt::graphlib

namespace tt::balancer
{
    struct BalancerSolution;
}  // namespace tt::balancer

namespace tt::placer
{
    struct PlacerSolution;
}  // namespace tt::placer

namespace tt
{
struct DeviceConfig;

std::tuple<graphlib::Edge, graphlib::Node*, graphlib::Edge> insert_serialized_dram_queue_between_ops(
    graphlib::Graph* graph,
    std::string const& producer_name,
    std::string const& consumer_name,
    graphlib::PortId consumer_input_port_id,
    int num_entries = -1);

void reduce_ethernet_stream_usage(
    PostPlacerConfig& config,
    graphlib::Graph* graph,
    balancer::BalancerSolution& balancer_solution,
    placer::PlacerSolution& placer_solution,
    tt::DeviceConfig const& device_config);
}
