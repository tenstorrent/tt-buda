// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placer/dram.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_map>

#include "buda_passes.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node_types.hpp"
#include "placer/allocator_utils.hpp"
#include "placer/dram_allocator.hpp"
#include "utils/logger.hpp"

namespace tt
{
// from backend
namespace placer
{
using Graph = graphlib::Graph;
using Node = graphlib::Node;

static void log_epoch_to_epoch_queue_info(Graph *graph, const PlacerSolution &placer_solution)
{
    log_debug(LogPlacer, "*** Logging Epoch-To-Epoch Queues ***");
    std::uint32_t total_e2e_queue_size = 0;
    std::map<std::uint32_t, std::uint32_t> epoch_id_to_total_e2e_input_buffer_size;
    std::map<graphlib::NodeEpochType, std::uint32_t> epoch_type_to_total_e2e_input_buffer_size;

    for (Node *node : graph->nodes())
    {
        if (node->node_type() == graphlib::NodeType::kQueue and
            node->as<graphlib::QueueNode>()->queue_type() == graphlib::QueueNodeType::EpochToEpoch)
        {
            std::uint32_t producer_epoch_id = get_first_epoch_producer(graph, node, placer_solution);
            graphlib::NodeEpochType producer_epoch_type = placer_solution.epoch_type(producer_epoch_id);
            const auto &queue_placement = placer_solution.name_to_queue_placement.at(node->name());

            std::uint32_t total_buffer_size = std::accumulate(
                queue_placement.dram_buffers.begin(),
                queue_placement.dram_buffers.end(),
                0,
                [](std::uint32_t accumulator, const QueueBufferPlacement &buffer_placement)
                { return accumulator + buffer_placement.buffer_size; });

            total_e2e_queue_size += total_buffer_size;
            for (std::uint32_t consumer_epoch_id : get_consumer_epoch_ids(graph, node, placer_solution))
            {
                graphlib::NodeEpochType consumer_epoch_type = placer_solution.epoch_type(consumer_epoch_id);
                log_trace(
                    LogPlacer,
                    "{}->{}, {}->{}, e2e-queue: {}, size: {}",
                    graphlib::node_epoch_type_to_string(producer_epoch_type),
                    graphlib::node_epoch_type_to_string(consumer_epoch_type),
                    producer_epoch_id,
                    consumer_epoch_id,
                    node->name(),
                    total_buffer_size);
                epoch_id_to_total_e2e_input_buffer_size[consumer_epoch_id] += total_buffer_size;
                epoch_type_to_total_e2e_input_buffer_size[consumer_epoch_type] += total_buffer_size;
            }
        }
    }
    for (const auto &[epoch_id, total_input_buffer_size] : epoch_id_to_total_e2e_input_buffer_size)
    {
        log_debug(
            LogPlacer,
            "EpochId: {}, EpochType: {}: total input e2e buffer size: {} KB",
            epoch_id,
            graphlib::node_epoch_type_to_string(placer_solution.epoch_type(epoch_id)),
            ((float)total_input_buffer_size / 1024));
    }
    for (const auto &[epoch_type, total_input_buffer_size] : epoch_type_to_total_e2e_input_buffer_size)
    {
        log_debug(
            LogPlacer,
            "EpochType={}: total input e2e buffer size: {} KB",
            graphlib::node_epoch_type_to_string(epoch_type),
            ((float)total_input_buffer_size / 1024));
    }
    log_debug(LogPlacer, "Total e2e transfer: {} KB", ((float)total_e2e_queue_size / 1024));
}

// TODO - get this from backend
Coord logical_to_physical_coord(
    const Coord &logical_coord, const DeviceConfig &config, const std::vector<std::uint32_t> &harvested_rows)
{
    if (config.is_grayskull())
        return logical_coord;  // TODO

    auto col = (logical_coord.col <= 3) ? logical_coord.col + 1 : logical_coord.col + 2;
    auto row = (logical_coord.row <= 4) ? logical_coord.row + 1 : logical_coord.row + 2;

    for (auto harvested : harvested_rows)
    {
        if (row >= harvested)
            row++;
    }

    return Coord{.row = row, .col = col};
}

// Figure out which cores are reading from which dram buffer (or writing to)
// dram_buffer is relative coordinate within the buffer grid
std::vector<Coord> get_reader_cores(
    const Node *node, const OpPlacement &placement, std::uint32_t operand, Coord dram_buffer, GridShape queue_grid)
{
    if (node->node_type() == graphlib::NodeType::kBudaOp)
    {
        const graphlib::OpNode *op = node->as<graphlib::OpNode>();

        const bool t = placement.grid_transpose;

        // Figure out the scale ratio of the op vs. queue grid, and scale start/end linearly
        std::uint32_t op_size_c = t ? placement.placed_cores.size_r() : placement.placed_cores.size_c();
        std::uint32_t op_size_r = t ? placement.placed_cores.size_c() : placement.placed_cores.size_r();
        float queue_relative_size_c = (float)op_size_c / (float)queue_grid.columns;
        float queue_relative_size_r = (float)op_size_r / (float)queue_grid.rows;

        std::uint32_t op_start_c = (float)dram_buffer.col * queue_relative_size_c;
        std::uint32_t op_start_r = (float)dram_buffer.row * queue_relative_size_r;

        float op_end_c_fl = (float)(dram_buffer.col + 1) * queue_relative_size_c;
        float op_end_r_fl = (float)(dram_buffer.row + 1) * queue_relative_size_r;

        // If we moved a bit into the next core, include it, otherwise don't
        std::uint32_t op_end_c = ((int)op_end_c_fl == op_end_c_fl) ? op_end_c_fl - 1 : op_end_c_fl;
        std::uint32_t op_end_r = ((int)op_end_r_fl == op_end_r_fl) ? op_end_r_fl - 1 : op_end_r_fl;

        if (op->is_matmul())
        {
            std::vector<Coord> cores;
            switch (operand)
            {
                case 0:  // Activations are only read by the first column.
                    for (std::uint32_t row = op_start_r; row <= op_end_r; row++)
                    {
                        Coord c =
                            placement.grid_transpose
                                ? Coord{.row = placement.placed_cores.start.row, .col = placement.placed_cores.start.col + row}
                                : Coord{
                                      .row = placement.placed_cores.start.row + row,
                                      .col = placement.placed_cores.start.col};
                        cores.push_back(c);
                    }
                    break;

                case 1:
                case 2:  // Weights are read by the last row and broadcast up
                    for (std::uint32_t col = op_start_c; col <= op_end_c; col++)
                    {
                        Coord c =
                            placement.grid_transpose
                                ? Coord{.row = placement.placed_cores.start.row + col, .col = placement.placed_cores.end.col - 1}
                                : Coord{
                                      .row = placement.placed_cores.end.row - 1,
                                      .col = placement.placed_cores.start.col + col};
                        cores.push_back(c);
                    }
                    break;
            }
            return cores;
        }
        else
        {
            // Even distribution for other ops
            std::vector<Coord> cores;
            for (std::uint32_t row = op_start_r; row <= op_end_r; row++)
            {
                for (std::uint32_t col = op_start_c; col <= op_end_c; col++)
                {
                    Coord c =
                        placement.grid_transpose
                            ? Coord{.row = placement.placed_cores.start.row + col, .col = placement.placed_cores.start.col + row}
                            : Coord{
                                  .row = placement.placed_cores.start.row + row,
                                  .col = placement.placed_cores.start.col + col};
                    cores.push_back(c);
                }
            }
            return cores;
        }
    }

    // Not an op
    return {placement.placed_cores.start};
}

// Writing is always 1-1 for now
Coord get_writer_core(const OpPlacement &placement, Coord dram_buffer)
{
    return Coord{
        .row = placement.placed_cores.start.row + dram_buffer.row,
        .col = placement.placed_cores.start.col + dram_buffer.col};
}

// Generate consumer locations for a queue
std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::vector<std::pair<Coord, std::uint32_t>>>>
get_consumer_locations(
    const PlacerSolution &placer_solution,
    const Graph *graph,
    const Node *node,
    GridShape queue_grid,
    const DramPlacerConfig &config,
    const std::vector<std::uint32_t> &harvested_rows)
{
    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::vector<std::pair<Coord, std::uint32_t>>>>
        consumer_loc;

    for (std::uint32_t row = 0; row < queue_grid.rows; row++)
    {
        for (std::uint32_t col = 0; col < queue_grid.columns; col++)
        {
            for (Edge user_edge : graph->user_data_edges(node))
            {
                const Node *user = graph->node_by_id(user_edge.consumer_node_id);
                auto it = placer_solution.name_to_op_placement.find(user->name());
                TT_LOG_ASSERT(
                    it != placer_solution.name_to_op_placement.end(),
                    "Consumer {} of queue {} not placed",
                    user->name(),
                    node->name());
                std::vector<Coord> readers = get_reader_cores(
                    user, it->second, user_edge.consumer_input_port_id, Coord{.row = row, .col = col}, queue_grid);
                for (Coord reader : readers)
                {
                    consumer_loc[row][col].push_back(
                        {logical_to_physical_coord(reader, config.device_config, harvested_rows),
                         it->second.epoch_id()});
                }
            }
        }
    }

    return consumer_loc;
}

// Generate producer locations for a queue
std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::pair<Coord, std::uint32_t>>>
get_producer_locations(
    const PlacerSolution &placer_solution,
    const Graph *graph,
    const Node *node,
    GridShape queue_grid,
    const DramPlacerConfig &config,
    const std::vector<std::uint32_t> &harvested_rows)
{
    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::pair<Coord, std::uint32_t>>> producer_loc;

    if (node->as<graphlib::QueueNode>()->is_input())
    {
        if (node->as<graphlib::InputNode>()->input_type() == graphlib::InputNodeType::Activation)
        {
            // Producer is PCIe
            for (std::uint32_t row = 0; row < queue_grid.rows; row++)
            {
                for (std::uint32_t col = 0; col < queue_grid.columns; col++)
                {
                    // TODO - get PCIe from backend
                    producer_loc[row][col] = {Coord{.row = 3, .col = 0}, 0};
                }
            }
        }
        // Return empty for other inputs, as they are filled in previous epochs or constants
        return producer_loc;
    }

    auto operands = graph->data_operands(node);
    TT_ASSERT(operands.size() == 1);
    auto it = placer_solution.name_to_op_placement.find(operands[0]->name());
    TT_LOG_ASSERT(
        it != placer_solution.name_to_op_placement.end(),
        "Producer {} of queue {} not placed",
        operands[0]->name(),
        node->name());
    for (std::uint32_t row = 0; row < queue_grid.rows; row++)
    {
        for (std::uint32_t col = 0; col < queue_grid.columns; col++)
        {
            Coord writer = get_writer_core(it->second, Coord{.row = row, .col = col});
            producer_loc[row][col] = {
                logical_to_physical_coord(writer, config.device_config, harvested_rows), it->second.epoch_id()};
        }
    }

    return producer_loc;
}

//
// The DRAM queues are split into buffers, one for each of the cores that is reading from a queue. These buffers can be
// freely allocated to any DRAM channel.
//
// The placer's job is to pick optimal dram channels for these buffers to maximize bandwidth and minimize latency, while
// ensuring that all dram queues fit in dram.
//
// If total queue size is simply too big, placer will fail to allocate successfully and this function will return false.
//
// This function assumes that ops are already placed and placer_solution is populated with the current solution.
void place_dram_queues(
    Graph *graph,
    PlacerSolution &placer_solution,
    balancer::BalancerSolution &balancer_solution,
    const HostMemoryPlacerConfig &host_memory_placer_config,
    const DramPlacerConfig &config,
    std::vector<DramAllocator> &chip_dram_allocators)
{
    std::map<string, string> linked_queues;
    std::map<string, string> subgraph_link_queues;
    std::unordered_map<std::uint32_t, std::vector<DRAMScheduleData>> scheduled_queue_placements;
    std::uint32_t max_chip_id = 0;
    std::uint32_t mmio_chip_index = 0;

    // Get harvested rows once, since it's an expensive query
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> harvested_rows;  // per chip_id
    auto get_harvested_rows = [&config, &harvested_rows](std::uint32_t chip_id)
    {
        if (harvested_rows.count(chip_id) == 0)
        {
            harvested_rows[chip_id] =
                config.device_config.get_harvested_rows(config.device_config.get_harvested_cfg()[chip_id]);
        }
        return harvested_rows.at(chip_id);
    };

    for (Node *node : graphlib::topological_sort(*graph))
    {
        Node *ref_node = get_reference_node(graph, node);  // Node from which we'll get the placement information
        if (ref_node == nullptr)
            continue;

        bool already_placed = is_queue_already_placed(placer_solution, node);
        bool already_allocated = is_queue_already_allocated(placer_solution, node);

        if (already_allocated)
        {
            log_trace(LogPlacer, "Skipping queue {} since it is already allocated.", node->name());
            continue;
        }
        if (is_host_queue(host_memory_placer_config, graph, node))
        {
            continue;
        }

        OpPlacement placement;
        CoordRange queue_coord_range;
        GridShape queue_grid;
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
        bool linked_queue =
            node->node_type() == graphlib::NodeType::kOutput and
            not graph->user_edges(node, [](Edge e) { return e.edge_type == graphlib::EdgeType::kPartialDataCopy; })
                    .empty();

        bool subgraph_link_queue =
            node->node_type() == graphlib::NodeType::kOutput and
            not graph->user_edges(node, [](Edge e) { return e.edge_type == graphlib::EdgeType::kSubgraphLink; })
                    .empty();

        if (node->node_type() == graphlib::NodeType::kInput)
        {
            queue_grid = placer_solution.input_queue_to_grid_shape.at(node->name());
            // Adjust the range to queue grid
            queue_coord_range.end.row = queue_coord_range.start.row + queue_grid.rows;
            queue_coord_range.end.col = queue_coord_range.start.col + queue_grid.columns;
        }
        else
        {
            bool grid_transpose = placement.grid_transpose;
            queue_grid = GridShape(
                (grid_transpose) ? queue_coord_range.size_c() : queue_coord_range.size_r(),
                (grid_transpose) ? queue_coord_range.size_r() : queue_coord_range.size_c());
        }
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

        // Output and intermediate queues depend on the producer grid. There can only be one writer to the queue.
        // If the output queue is to be placed on host, then no allocation is needed
        if (linked_queue)
        {
            node->as<graphlib::QueueNode>()->set_memory_access_type(graphlib::MemoryAccessType::RAM);
            auto partial_loopback_edges =
                graph->user_edges(node, [](Edge e) { return e.edge_type == graphlib::EdgeType::kPartialDataCopy; });
            TT_ASSERT(partial_loopback_edges.size() == 1);
            auto linked_node = graph->node_by_id(partial_loopback_edges[0].consumer_node_id);
            log_debug(tt::LogPlacer, "Linked node {}", linked_node->name());
            auto writer_block_shape = balancer_solution.block_shapes.at(linked_node->name());
            auto reader_block_shape = balancer_solution.block_shapes.at(node->name());
            TT_ASSERT(writer_block_shape.t % reader_block_shape.t == 0);
            int multiplier = writer_block_shape.t / reader_block_shape.t;
            node->as<graphlib::QueueNode>()->set_num_entries(multiplier * graph->get_microbatch());
            node->as<graphlib::QueueNode>()->set_alias(linked_node->name());
            linked_node->as<graphlib::QueueNode>()->set_memory_access_type(graphlib::MemoryAccessType::RAM);
            linked_node->as<graphlib::QueueNode>()->set_num_entries(graph->get_microbatch());
            linked_queues.emplace(node->name(), linked_node->name());
            placer_solution.name_to_queue_placement.insert(std::make_pair(
                node->name(),
                QueuePlacement{
                    .name = node->name(),
                    .input_name = ref_node->name(),
                    .grid_shape = queue_grid,
                    .on_host = false,
                    .chip_id = placement.chip_id,
                    .dram_buffers = {},
                    .host_buffers = {},
                    .write_stride = multiplier,
                }));
            log_debug(tt::LogPlacer, "Adding linked queue {}", node->name());
        }
        else if (subgraph_link_queue) {
            node->as<graphlib::QueueNode>()->set_memory_access_type(graphlib::MemoryAccessType::RAM);
            auto subgraph_link_edges =
                graph->user_edges(node, [](Edge e) { return e.edge_type == graphlib::EdgeType::kSubgraphLink; });
            TT_ASSERT(subgraph_link_edges.size() == 1);
            auto linked_node = graph->node_by_id(subgraph_link_edges[0].consumer_node_id);
            log_debug(tt::LogPlacer, "Subgraph link node {}", linked_node->name());
            auto writer_block_shape = balancer_solution.block_shapes.at(linked_node->name());
            auto reader_block_shape = balancer_solution.block_shapes.at(node->name());
            TT_ASSERT(writer_block_shape.t % reader_block_shape.t == 0);
            int multiplier = writer_block_shape.t / reader_block_shape.t;
            node->as<graphlib::QueueNode>()->set_num_entries(graph->get_microbatch());
            node->as<graphlib::QueueNode>()->set_alias(linked_node->name());
            linked_node->as<graphlib::QueueNode>()->set_memory_access_type(graphlib::MemoryAccessType::RAM);
            linked_node->as<graphlib::QueueNode>()->set_num_entries(graph->get_microbatch());
            subgraph_link_queues.emplace(node->name(), linked_node->name());
            placer_solution.name_to_queue_placement.insert(std::make_pair(
                node->name(),
                QueuePlacement{
                    .name = node->name(),
                    .input_name = ref_node->name(),
                    .grid_shape = queue_grid,
                    .on_host = false,
                    .chip_id = placement.chip_id,
                    .dram_buffers = {},
                    .host_buffers = {},
                    .write_stride = multiplier,
                }));
            log_debug(tt::LogPlacer, "Adding subgraph link queue {}", node->name());
        }
        else
        {
            // Currently assume that all consumers to the queue belong to the same chip
            int producer_chip_id = placement.chip_id;
            int consumer_chip_id = -1;
            for (Node *user : graph->data_users(node))
            {
                if (placer_solution.name_to_op_placement.find(user->name()) !=
                    placer_solution.name_to_op_placement.end())
                {
                    int current_consumer_chip_id = placer_solution.name_to_op_placement.at(user->name()).chip_id;
                    if (consumer_chip_id != -1 and consumer_chip_id != current_consumer_chip_id)
                    {
                        if (config.device_config.is_grayskull())
                        {
                            throw std::runtime_error(
                                "Placement for queue " + node->name() + " contains multiple remote consumers.");
                        }
                        else
                        {
                            // Wormhole allows this, but we need to turn off the prologue bit
                            if (node->node_type() == graphlib::NodeType::kInput)
                            {
                                node->as<graphlib::InputNode>()->set_prologue(false);
                            }
                        }
                    }
                    consumer_chip_id = current_consumer_chip_id;
                }
            }

            // the target device for the queue placement is set based on the consumer. For both GS and WH,
            // we want the consumer to read its input data from an input queue on the same chip.
            uint32_t chip_id = consumer_chip_id >= 0 ? consumer_chip_id : producer_chip_id;

            // If this is an e2e epoch, figure out which epoch is writing, and the last epoch that is reading, after
            // which we can deallocate the queue
            std::uint32_t producer_epoch = 0;
            std::uint32_t last_consumer_epoch = UINT_MAX;
            if (node->as<graphlib::QueueNode>()->queue_type() == graphlib::QueueNodeType::EpochToEpoch)
            {
                producer_epoch = get_first_epoch_producer(graph, node, placer_solution);
                last_consumer_epoch = get_last_epoch_use(graph, node, placer_solution);
            }

            // relevant only for grayskull p2p access
            bool in_p2p_region_soft = false;
            bool in_p2p_region_hard = false;
            bool is_input = false;
            bool is_prologue = false;
            if (node->as<graphlib::QueueNode>()->is_input())
            {
                auto input_node = node->as<graphlib::InputNode>();
                if ((input_node->is_activation() || input_node->is_loss()))
                    in_p2p_region_soft = ((queue_grid.rows == 1) && (queue_grid.columns == 1));  // try, if possible
                is_input = true;
                is_prologue = input_node->is_prologue();
            }
            if (config.device_config.is_grayskull() and node->as<graphlib::QueueNode>()->is_epoch_to_epoch())
            {
                if (producer_chip_id != consumer_chip_id)
                    in_p2p_region_hard = true;  // won't work if we don't fit
            }
            if (config.device_config.is_wormhole() and in_p2p_region_soft)
            {
                // Try to put the activations into mmio-capable p2p region to enable the fast-tilize region
                // simply round-robin across available mmio capable chips if the consumer chip-id is not mmio capable.
                auto mmio_chip_id_it = std::find(
                    std::begin(config.device_config.chips_with_mmio),
                    std::end(config.device_config.chips_with_mmio),
                    chip_id);
                if (mmio_chip_id_it == config.device_config.chips_with_mmio.end())
                {
                    mmio_chip_index = (mmio_chip_index % config.device_config.chips_with_mmio.size());
                    chip_id = config.device_config.chips_with_mmio.at(mmio_chip_index++);
                }
            }

            // Override all dram queue placements with user specified assignments
            if (config.manual_dram_queue_placement.find(node->name()) != config.manual_dram_queue_placement.end())
            {
                const auto &dram_placement = config.manual_dram_queue_placement.at(node->name());
                if (dram_placement.chip_id.has_value())
                {
                    log_debug(
                        tt::LogPlacer,
                        "Manually placing dram queue {} to chip_id: {}",
                        node->name(),
                        dram_placement.chip_id.value());
                    chip_id = dram_placement.chip_id.value();
                }
            }

            // Save placement information, the actual placement will have in second pass-through
            if (chip_id > max_chip_id)
                max_chip_id = chip_id;
            log_debug(tt::LogPlacer, "\tScheduling queue {} for placement", node->name());

            scheduled_queue_placements[chip_id].push_back(std::make_pair(
                already_placed
                    ? placer_solution.name_to_queue_placement.at(node->name())
                    : QueuePlacement{.name = node->name(), .input_name = input_name, .grid_shape = queue_grid, .on_host = false, .chip_id = chip_id, .dram_buffers = {}, .host_buffers = {}, .epoch_allocate = -1, .epoch_deallocate = -1},
                QueueDRAMPlacementParameters{
                    .config = &config,
                    .node = node,
                    .grid_shape = queue_grid,
                    .consumer_loc = get_consumer_locations(
                        placer_solution, graph, node, queue_grid, config, get_harvested_rows(chip_id)),
                    .producer_loc = get_producer_locations(
                        placer_solution, graph, node, queue_grid, config, get_harvested_rows(chip_id)),
                    .block_shape = block_shape,
                    .producer_epoch = producer_epoch,
                    .last_consumer_epoch = last_consumer_epoch,
                    .in_p2p_region_soft = in_p2p_region_soft,
                    .in_p2p_region_hard = in_p2p_region_hard,
                    .is_input = is_input,
                    .is_prologue = is_prologue}));
        }
    }

    for (std::uint32_t chip_id = 0; chip_id <= max_chip_id; chip_id++)
    {
        if (scheduled_queue_placements.count(chip_id) == 0)
            continue;
        log_info(tt::LogPlacer, "Running DRAM allocator for device {}", chip_id);
        chip_dram_allocators.at(chip_id).allocate_queues(
            scheduled_queue_placements.at(chip_id), config.disable_dynamic_dram);
        for (auto &[queue_placement, parameters] : scheduled_queue_placements[chip_id])
        {
            log_debug(tt::LogPlacer, "\tAllocating/placing queue {} on chip {}", queue_placement.name, chip_id);
            if (placer_solution.name_to_queue_placement.find(parameters.node->name()) ==
                placer_solution.name_to_queue_placement.end())
            {
                placer_solution.name_to_queue_placement.emplace(parameters.node->name(), queue_placement);
            }
            else
            {
                placer_solution.name_to_queue_placement[parameters.node->name()] = queue_placement;
            }
        }
    }
    for (auto &[key, val] : linked_queues)
    {
        placer_solution.name_to_queue_placement[key].dram_buffers =
            placer_solution.name_to_queue_placement[val].dram_buffers;
        placer_solution.name_to_queue_placement[key].write_only = true;
        placer_solution.name_to_queue_placement[val].read_only = true;
        placer_solution.name_to_queue_placement[key].chip_id = placer_solution.name_to_queue_placement[val].chip_id;
    }

    for (auto &[key, val] : subgraph_link_queues)
    {
        placer_solution.name_to_queue_placement[key].dram_buffers =
            placer_solution.name_to_queue_placement[val].dram_buffers;
        placer_solution.name_to_queue_placement[key].write_only = true;
        placer_solution.name_to_queue_placement[val].read_only = true;
        placer_solution.name_to_queue_placement[key].chip_id = placer_solution.name_to_queue_placement[val].chip_id;
    }

    log_epoch_to_epoch_queue_info(graph, placer_solution);
}

}  // namespace placer
}  // namespace tt
