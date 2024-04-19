// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// post placer buda passes
// these functions are called from run_post_placer_buda_passes

#pragma once

#include "balancer/balancer.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "placer/dram.hpp"
#include "placer/dram_allocator.hpp"
#include "placer/host_memory.hpp"
#include "placer/placer.hpp"

namespace tt {

using Graph = graphlib::Graph;
using Node = graphlib::Node;
using NodeContext = graphlib::NodeContext;
using NodeToNodeMap = std::unordered_map<Node *, Node *>;

struct PostPlacerConfig {
    PostPlacerConfig(
        DeviceConfig const &device_config,
        std::uint32_t microbatch_size,
        std::uint32_t microbatch_count,
        bool enable_t_streaming,
        bool input_queues_on_host,
        bool output_queues_on_host,
        DramQueueMap manual_dram_queue_placement,
        std::uint32_t output_queue_multiplier = 2,
        std::uint32_t input_queue_multiplier = 2,
        bool enable_cross_chip_buffering = true,
        placer::DRAMPlacementAlgorithm placement_algorithm = placer::ROUND_ROBIN) :
        device_config(device_config),
        dram_placer_config(device_config, input_queues_on_host, output_queues_on_host, manual_dram_queue_placement),
        host_memory_placer_config(device_config, input_queues_on_host, output_queues_on_host),
        output_queue_multiplier(output_queue_multiplier),
        input_queue_multiplier(input_queue_multiplier),
        microbatch_size(microbatch_size),
        microbatch_count(microbatch_count),
        enable_t_streaming(enable_t_streaming),
        enable_cross_chip_buffering(enable_cross_chip_buffering),
        placement_algorithm(placement_algorithm)
    {
    }

    DeviceConfig const &device_config;
    placer::DramPlacerConfig dram_placer_config;
    placer::HostMemoryPlacerConfig host_memory_placer_config;

    std::uint32_t output_queue_multiplier;
    std::uint32_t input_queue_multiplier;
    std::uint32_t microbatch_size;
    std::uint32_t microbatch_count;
    bool enable_t_streaming;
    bool enable_cross_chip_buffering;
    placer::DRAMPlacementAlgorithm placement_algorithm;
};

void set_prologue_queues(Graph *graph, balancer::OpModelMap const &op_model_map);

void post_placer_lower_buda_attrs(
    Graph *graph, DeviceConfig const &device_config, balancer::OpModelMap const &op_model_map);

void replace_recompute_with_checkpoint(graphlib::Graph *graph, const placer::PlacerSolution &placer_solution);

void validate_subgraph_placement(Graph *graph, placer::PlacerSolution &placer_solution);

// Remove buffering queues connecting cross epoch nodes so that E2E queues can be inserted instead.
void  remove_buffering_queues_from_cross_epoch_edges(
    graphlib::Graph *graph,
    const placer::PlacerSolution &placer_solution
);

// Insert a queue between every two ops that are not in the same epoch
// or if we have an edge that was cut by a GraphSolver.
//
void insert_epoch_to_epoch_queues(
    graphlib::Graph *graph,
    const placer::PlacerSolution &placer_solution,
    const std::unordered_set<graphlib::NodeEpochType> &epoch_types,
    const balancer::CutEdges &graph_solver_cut_edges);

// Insert queues between ops on different epochs
void insert_epoch_to_epoch_queue(
    graphlib::Graph *graph,
    const std::string &name,
    graphlib::Edge edge,
    graphlib::UBlockOrder op_ublock_order,
    bool cross_epoch_type);

void connect_gradient_accum_queue(graphlib::Graph *graph, Node* node, const graphlib::Edge& edge);

// Set queue entry sizes based on the configuration for different types of queues
void set_queue_sizes(graphlib::Graph *graph, PostPlacerConfig &config, const placer::PlacerSolution &placer_solution);

std::vector<std::uint32_t> get_consumer_epoch_ids(const graphlib::Graph *graph, const graphlib::Node *node, const placer::PlacerSolution &placer_solution);
std::uint32_t get_last_epoch_use(const graphlib::Graph *graph, const graphlib::Node *node, const placer::PlacerSolution &placer_solution);
std::uint32_t get_first_epoch_producer(const graphlib::Graph *graph, const graphlib::Node *node, const placer::PlacerSolution &placer_solution);
void validate_multichip_queue_placements(const PostPlacerConfig& config, const graphlib::Graph *graph, const placer::PlacerSolution &placer_solution);
bool any_consumers_cross_epoch(graphlib::Graph *graph, graphlib::Node* producer);
}
