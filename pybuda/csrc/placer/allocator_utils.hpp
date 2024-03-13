// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// fwd declare
namespace tt
{
namespace graphlib
{

class Node;
class Graph;
class QueueNode;
}  // namespace graphlib
namespace balancer
{
struct BlockShape;
struct BalancerSolution;
}  // namespace balancer

}  // namespace tt::graphlib

namespace tt::placer
{

struct HostMemoryPlacerConfig;

int get_queue_size(const graphlib::QueueNode* node, balancer::BlockShape const& block_shape, bool untilized);
graphlib::Node* get_reference_node(const graphlib::Graph* graph, const graphlib::Node* node);
bool is_queue_already_placed(const PlacerSolution& placer_solution, const graphlib::Node* node);
bool is_queue_already_allocated(const PlacerSolution& placer_solution, const graphlib::Node* node);

bool is_input_host_queue(
    const HostMemoryPlacerConfig& config, const graphlib::Graph* graph, const graphlib::Node* node);
bool is_output_host_queue(
    const HostMemoryPlacerConfig& config, const graphlib::Graph* graph, const graphlib::Node* node);

bool is_host_queue(
    const HostMemoryPlacerConfig& host_memory_config, const graphlib::Graph* graph, const graphlib::Node* node);

}  // namespace tt::placer
