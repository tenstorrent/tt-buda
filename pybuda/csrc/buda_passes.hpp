// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "balancer/balancer.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "passes/fork_join.hpp"
#include "passes/fracture.hpp"
#include "passes/dataformat.hpp"
#include "passes/post_placer_buda_passes.hpp"
#include "placer/chip_id_assignment.hpp"
#include "placer/dram.hpp"
#include "placer/dram_allocator.hpp"
#include "placer/placer.hpp"
#include "scheduler/scheduler.hpp"
#include "utils/ordered_associative_containers/ordered_map.hpp"

namespace tt {

void reduce_ethernet_stream_usage(
    PostPlacerConfig& config,
    graphlib::Graph* graph,
    balancer::BalancerSolution& balancer_solution,
    placer::PlacerSolution& placer_solution,
    tt::DeviceConfig const& device_config);

using NodeId = graphlib::NodeId;
using PortId = graphlib::PortId;
void lower_reshape(Graph *, graphlib::OpNode *node);    // unused

// Run post initial graph passes
std::tuple<std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>>, passes::FractureChipIdAssignments>
run_post_initial_graph_passes(
    graphlib::Graph *graph, py::object compiler_cfg_object, passes::FractureGroups const &fracture_groups);
void run_optimization_graph_passes(graphlib::Graph *graph, const DeviceConfig &device_config);
std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>> run_post_optimize_decompose_graph_passes(
    graphlib::Graph *graph, py::object compiler_cfg_object);
std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>> run_post_autograd_graph_passes(
    graphlib::Graph *graph, py::object compiler_cfg_object);

// Run lowering passes
std::pair<std::unique_ptr<graphlib::Graph>, placer::PlacerConfigUpdate> run_pre_placer_buda_passes(
    graphlib::Graph *graph,
    scheduler::SchedulerConfig scheduler_config,
    const DeviceConfig &device_config,
    std::vector<std::uint32_t> chip_ids = {0},
    const placer::PredicatesToBreaks &predicates_to_chip_break = {},
    const placer::PredicatesToBreaks &predicates_to_epoch_break = {},
    const std::vector<std::string> &op_names_dont_fuse = {},
    const std::vector<std::string> &op_names_manual_fuse = {},
    const passes::FractureChipIdAssignments &fracture_chip_id_assignments = {},
    const std::optional<DataFormat> default_df_override = {},
    const std::optional<DataFormat> default_accumulate_df = {},
    const bool enable_broadcast_splitting = false,
    const DataFormat fp32_fallback = DataFormat::Float16_b,
    const MathFidelity default_math_fidelity = MathFidelity::HiFi3,
    const bool enable_auto_fusing = false,
    const int amp_level = 0,
    const bool enable_recompute = false,
    const bool output_queues_on_host = true,
    const bool input_queues_on_host = true,
    const tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        &ins_instructions = {},
    const std::vector<std::tuple<std::string, std::string, int>> &insert_queues = {},
    std::vector<AMPNodeProperties> amp_properties = {},
    const std::vector<std::string> &op_intermediates_to_save = {},
    bool use_interactive_placer = true,
    bool enable_device_tilize = false);
struct PostPlacerResults
{
    std::unordered_map<std::string, float> perf_model_results;
    tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        ins_instructions;
    std::vector<std::vector<placer::Blocks>> allocated_blocks;
    std::uint32_t current_host_address;
};

// Run post-placer passes, like queue and buffer insertion. Return perf model results, if applicable.
PostPlacerResults run_post_placer_buda_passes(
    graphlib::Graph *graph,
    const std::string &graph_name,
    const DeviceConfig &device_config,
    placer::PlacerSolution &placer_solution,
    PostPlacerConfig &config,
    std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    const tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        &previous_ins_instructions,
    std::vector<std::vector<placer::Blocks>> &pre_allocated_blocks,
    std::uint32_t last_host_address);

// Last minute changes before netlist generation
void run_pre_netlist_generation_buda_passes(
    graphlib::Graph *graph,
    const std::string &graph_name,
    const DeviceConfig &device_config,
    std::unordered_map<std::string, py::object> intermediates,
    placer::PlacerSolution &placer_solution,
    PostPlacerConfig &config,
    std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    std::vector<std::vector<placer::Blocks>> &pre_allocated_blocks,
    std::uint32_t last_host_address);

// Pre-lowering passes, last-minute changes before going to buda ops
void run_pre_lowering_passes(graphlib::Graph *graph);

}
