// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placer/host_memory.hpp"

#include "backend_api/device_config.hpp"
#include "balancer/balancer.hpp"
#include "balancer/types.hpp"
#include "graph_lib/node_types.hpp"
#include "placer/allocator_utils.hpp"
#include "utils/logger.hpp"

namespace tt::placer
{

HostMemoryPlacerConfig::HostMemoryPlacerConfig(
    const DeviceConfig& device_config, bool input_queues_on_host, bool output_queues_on_host)
{
    if (input_queues_on_host)
    {
        if (device_config.is_grayskull())
        {
            log_warning(
                LogPlacer,
                "Compilation Option with input queue placed on host, but Grayskull does not support fast device reads "
                "from host. Placer opting to allocate the queue on device instead.");
            input_queues_on_host = false;
        }
        if (device_config.is_wormhole_b0() && device_config.chip_ids.size() > 1)
        {
            log_warning(
                LogPlacer,
                "Compilation Option with input queue placed on host, but Wormhole does not support fast device reads "
                "from host in multi-chip systems. Placer opting to allocate the queue on device instead.");
            input_queues_on_host = false;
        }
    }

    this->input_queues_on_host = input_queues_on_host;
    this->output_queues_on_host = output_queues_on_host;

    for (std::uint32_t i = 0; i < device_config.get_host_memory_num_channels(); i++)
    {
        this->host_memory_regions.emplace_back(
            i, device_config.get_host_memory_channel_start_address(), device_config.get_host_memory_channel_size(i));
    }
}

bool HostMemoryPlacerConfig::place_input_queues_on_host() const { return this->input_queues_on_host; }
bool HostMemoryPlacerConfig::place_output_queues_on_host() const { return this->output_queues_on_host; }

std::pair<CoordRange, GridShape> get_host_queue_grid(
    const HostMemoryPlacerConfig &config,
    const PlacerSolution &placer_solution,
    const OpPlacement &placement,
    const graphlib::Graph *graph,
    const graphlib::Node *node,
    CoordRange queue_coord_range)
{
    GridShape queue_grid;
    if (node->node_type() == graphlib::NodeType::kInput)
    {
        queue_grid = placer_solution.input_queue_to_grid_shape.at(node->name());
        // Adjust the range to queue grid
        queue_coord_range.end.row = queue_coord_range.start.row + queue_grid.rows;
        queue_coord_range.end.col = queue_coord_range.start.col + queue_grid.columns;
    }
    else if (is_output_host_queue(config, graph, node))
    {
        queue_grid = GridShape(1, 1);
        queue_coord_range.end.row = queue_coord_range.start.row + 1;
        queue_coord_range.end.col = queue_coord_range.start.col + 1;
    }
    else
    {
        bool grid_transpose = placement.grid_transpose;
        queue_grid = GridShape(
            (grid_transpose) ? queue_coord_range.size_c() : queue_coord_range.size_r(),
            (grid_transpose) ? queue_coord_range.size_r() : queue_coord_range.size_c());
    }
    return {queue_coord_range, queue_grid};
}

std::string get_host_input_name(
    const graphlib::Graph *graph, const graphlib::Node *ref_node, const graphlib::Node *node)
{
    // Loopback queue (i.e. queue that optmizer writes to) should not have 'HOST' as their input even
    // though the host will be initializing them.
    std::string input_name = ref_node->name();
    if (node->node_type() == graphlib::NodeType::kInput)
    {
        std::vector<graphlib::Edge> loopback_edges = graph->operand_edges(
            node, [](graphlib::Edge e) { return e.edge_type == graphlib::EdgeType::kDataLoopback; });
        if (loopback_edges.size() > 0)
        {
            input_name = graph->node_by_id(loopback_edges[0].producer_node_id)->name();
        }
        else
        {
            input_name = "HOST";
        }
    }
    return input_name;
}
QueuePlacement get_queue_placement(
    const HostMemoryPlacerConfig &config,
    HostMemoryAllocator &allocator,
    const graphlib::Graph *graph,
    const graphlib::Node *node,
    const graphlib::Node *ref_node,
    const PlacerSolution &placer_solution,
    const balancer::BalancerSolution &balancer_solution)
{
    GridShape queue_grid;
    OpPlacement placement;
    CoordRange queue_coord_range;
    balancer::BlockShape block_shape;

    try
    {
        placement = placer_solution.name_to_op_placement.at(ref_node->name());
        queue_coord_range = placement.placed_cores;
        if (ref_node->get_type() == "BudaOp::ethernet_datacopy")
        {
            auto const &grid_shape = balancer_solution.op_models.at(ref_node->name()).grid_shape;
            queue_coord_range = CoordRange{
                .start = Coord{.row = 0, .col = 0},
                .end = Coord{.row = (uint32_t)grid_shape.r, .col = (uint32_t)grid_shape.c}};
        }
        block_shape = balancer_solution.block_shapes.at(
            (node->node_type() == graphlib::NodeType::kQueue) ? ref_node->name() : node->name());
        if (node->node_type() == graphlib::NodeType::kQueue and
            balancer_solution.op_models.at(ref_node->name()).has_sparse_buffer())
        {
            TT_ASSERT((queue_coord_range.size_c() % 2) == 0);
            queue_coord_range.end.col = queue_coord_range.start.col + (queue_coord_range.size_c() / 2);
        }
    }
    catch (std::out_of_range &e)
    {
        throw std::runtime_error(
            "Placement for node " + ref_node->name() + " from queue " + node->name() + " is missing.");
    }

    bool output_host_queue = is_output_host_queue(config, graph, node);  // only output tensors to host are untilized
    bool untilize = output_host_queue;
    if (output_host_queue)
    {
        block_shape = balancer_solution.block_shapes.at(node->name());
    }
    std::tie(queue_coord_range, queue_grid) =
        get_host_queue_grid(config, placer_solution, placement, graph, node, queue_coord_range);
    std::uint32_t queue_size = get_queue_size(node->as<graphlib::QueueNode>(), block_shape, untilize);

    return QueuePlacement{
        .name = node->name(),
        .input_name = get_host_input_name(graph, ref_node, node),
        .grid_shape = queue_grid,
        .on_host = true,
        .chip_id = placement.chip_id,
        .dram_buffers = {},
        .host_buffers = allocator.allocate_queue(node, queue_coord_range, queue_size)};
}

void place_host_queues(
    const HostMemoryPlacerConfig &host_memory_config,
    HostMemoryAllocator &host_memory_allocator,
    const graphlib::Graph *graph,
    PlacerSolution &placer_solution,
    balancer::BalancerSolution &balancer_solution)
{
    for (Node *node : graphlib::topological_sort(*graph))
    {
        if (Node *ref_node = get_reference_node(graph, node);
            ref_node != nullptr and is_host_queue(host_memory_config, graph, node))
        {
            // Output and intermediate queues depend on the producer grid. There can only be one writer to the queue.
            // If the output queue is to be placed on host, then no allocation is needed
            bool already_placed = is_queue_already_placed(placer_solution, node);
            bool already_allocated = is_queue_already_allocated(placer_solution, node);
            if (already_allocated || already_placed)
            {
                log_trace(LogPlacer, "Skipping queue {} since it is already allocated.", node->name());
                continue;
            }

            placer_solution.name_to_queue_placement.insert(std::make_pair(
                node->name(),
                get_queue_placement(
                    host_memory_config,
                    host_memory_allocator,
                    graph,
                    node,
                    ref_node,
                    placer_solution,
                    balancer_solution)));
        }
    }
}

}  // namespace tt::placer
