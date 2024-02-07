// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/eth_stream_reduction.hpp"

#include "backend_api/device_config.hpp"
#include "balancer/balancer_cache_collection.hpp"
#include "balancer/legalizer/legalizer.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"

#include "placer/placer.hpp"
#include "post_placer_buda_passes.hpp"
#include "t_stream.hpp"

#include "graph_lib/defines.hpp"
#include "third_party/budabackend/device/tt_cluster_descriptor.h"

#include "lower_to_buda/common.hpp"
#include "utils/logger.hpp"

#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tt {
using chip_boundary_id_t = std::pair<uint32_t,uint32_t>;
using producer_consumer_pair_t = std::tuple<std::string, std::string, graphlib::PortId>;
};

namespace std {
template <>
struct hash<tt::chip_boundary_id_t> {
  std::size_t operator()(tt::chip_boundary_id_t const &o) const {
    std::size_t seed = 0;
    seed = std::hash<std::size_t>()(o.first) ^ std::hash<std::size_t>()(o.second) << 1;
    return seed;
  }
};

template <>
struct hash<tt::producer_consumer_pair_t> {
  std::size_t operator()(tt::producer_consumer_pair_t const &o) const {
    std::size_t seed = 0;
    seed = std::hash<std::string>()(std::get<0>(o)) ^ (std::hash<std::string>()(std::get<1>(o)) << 1) ^ (std::hash<std::size_t>()(std::get<2>(o)) << 2);
    return seed;
  }
};

}; // namespace std


namespace tt {

struct chip_to_chip_data_edge_t 
{
    uint32_t producer_chip;
    uint32_t consumer_chip;
    graphlib::PortId operand_index;
    int streams_needed_per_hop;
    int streams_needed_total; // in case of multiple hops this may be different from above
};

// producer-consumer_pair, data_edge, chip_to_insert_serializing op on
using data_edge_serialization_spec_t = std::tuple<producer_consumer_pair_t, chip_to_chip_data_edge_t, placer::PlacerSolution::EpochId>;

struct temporal_epoch_chip_to_chip_data_edges_t 
{
    std::unordered_map<chip_boundary_id_t, std::unordered_set<producer_consumer_pair_t>> chip_boundary_producer_consumer_pairs;
    std::unordered_map<producer_consumer_pair_t, chip_to_chip_data_edge_t> chip_to_chip_data_edges;
    std::unordered_map<chip_boundary_id_t, int> chip_boundary_needed_streams;
};


static tt_xy_pair get_op_or_queue_placed_grid_size(placer::PlacerSolution const& placer_solution, const graphlib::Node &op_info) {
    bool has_op_placement = placer_solution.name_to_op_placement.find(op_info.name()) != placer_solution.name_to_op_placement.end();
    if (has_op_placement) {
        placer::OpPlacement const& placement = placer_solution.name_to_op_placement.at(op_info.name());
        return tt_xy_pair(placement.placed_cores.size_c(), placement.placed_cores.size_r());
    } else {
        placer::QueuePlacement const& placement = placer_solution.name_to_queue_placement.at(op_info.name());
        return tt_xy_pair(placement.grid_shape.columns, placement.grid_shape.rows);
    }
}

static int get_op_num_input_streams(const graphlib::Node &op_info, placer::PlacerSolution const& placer_solution, graphlib::PortId operand_index) 
{
    tt_xy_pair placed_grid_size = get_op_or_queue_placed_grid_size(placer_solution, op_info);

    if (op_info.get_type() == "BudaOp::matmul") 
    {
        return operand_index == 0 ? placed_grid_size.y : placed_grid_size.x;
    } 
    else if (op_info.get_type() == "fused_op") 
    {
        TT_ASSERT(false, "Don't know how to support yet");
        return placed_grid_size.y * placed_grid_size.x;
    } 
    else 
    {
        return placed_grid_size.y * placed_grid_size.x;
    }
}


static uint32_t get_producer_consumer_pair_temporal_epoch(placer::PlacerSolution const& placer_solution, std::string const& producer, std::string const& consumer) {
    bool producer_has_op_placement = placer_solution.name_to_op_placement.find(producer) != placer_solution.name_to_op_placement.end();
    if (producer_has_op_placement) {
        return placer_solution.temporal_epoch_id(placer_solution.name_to_op_placement.at(producer).global_epoch_id);
    } else {
        return placer_solution.temporal_epoch_id(placer_solution.name_to_op_placement.at(consumer).global_epoch_id);
    }
}

static std::unordered_map<int, temporal_epoch_chip_to_chip_data_edges_t> collect_chip_to_chip_data_edges_per_temporal_epoch(
    graphlib::Graph *graph, 
    placer::PlacerSolution &placer_solution) 
{
    auto chip_to_chip_data_edges_per_temporal_epoch = std::unordered_map<int, temporal_epoch_chip_to_chip_data_edges_t>{};
    for (auto const& [node_id, edges] : graph->operands_map())
    {
        graphlib::Node* consumer = graph->node_by_id(node_id);

        auto consumer_chip = placer_solution.chip_id(consumer->name());
        for (graphlib::Edge const& edge : edges)
        {
            if (edge.edge_type != graphlib::EdgeType::kData and edge.edge_type != graphlib::EdgeType::kDataLoopback)
            {
                continue;
            }
            TT_ASSERT(node_id == edge.consumer_node_id);

            graphlib::Node* producer = graph->node_by_id(edge.producer_node_id);

            uint32_t producer_chip = placer_solution.chip_id(producer->name());
            bool is_chip_to_chip_edge = consumer_chip != producer_chip;

            if (is_chip_to_chip_edge)
            {
                int temporal_epoch = (consumer->node_type() == graphlib::NodeType::kBudaOp) ? placer_solution.temporal_epoch_id(consumer->name()) : placer_solution.temporal_epoch_id(producer->name());
                graphlib::Node* consumer_node = graph->node_by_id(edge.consumer_node_id);
                auto &chip_to_chip_edges = chip_to_chip_data_edges_per_temporal_epoch[temporal_epoch];
                int streams_needed_per_hop = get_op_num_input_streams(*consumer_node, placer_solution, edge.consumer_input_port_id); 
                int num_hops = std::abs(static_cast<long>(producer_chip) - static_cast<long>(consumer_chip));
                int streams_needed_total = num_hops * streams_needed_per_hop;
                auto const& chip_boundary = chip_boundary_id_t{std::min(producer_chip, consumer_chip), std::max(producer_chip, consumer_chip)};
                auto const& producer_consumer_pair = producer_consumer_pair_t{producer->name(), consumer->name(), edge.consumer_input_port_id};

                log_debug("\tChip-to-chip edge between {} (chip {}) and {} (chip {}). {} streams needed per hop", producer->name(), producer_chip, consumer->name(), consumer_chip, streams_needed_per_hop);
                for (auto c = chip_boundary.first; c != chip_boundary.second; c++) 
                {
                    auto const& one_hop_chip_boundary = chip_boundary_id_t{c, c+1};
                    chip_to_chip_edges.chip_boundary_producer_consumer_pairs[one_hop_chip_boundary].insert(producer_consumer_pair);
                    chip_to_chip_edges.chip_boundary_needed_streams[one_hop_chip_boundary] += streams_needed_per_hop;
                    log_debug("\t\t chip {} -> chip {}: {} required streams added, {} needed in total", one_hop_chip_boundary.first, one_hop_chip_boundary.second, streams_needed_per_hop, chip_to_chip_edges.chip_boundary_needed_streams.at(one_hop_chip_boundary));
                }
                chip_to_chip_edges.chip_to_chip_data_edges[producer_consumer_pair] = chip_to_chip_data_edge_t{
                        .producer_chip=producer_chip, 
                        .consumer_chip=consumer_chip, 
                        .operand_index=edge.consumer_input_port_id,
                        .streams_needed_per_hop=streams_needed_per_hop,
                        .streams_needed_total=streams_needed_total
                    };
                TT_ASSERT(edge.producer_node_id == producer->id());
            }
        }
    }

    return chip_to_chip_data_edges_per_temporal_epoch;
}

static std::unordered_map<placer::PlacerSolution::EpochId, std::unordered_map<uint32_t, int>> collect_available_cores_per_temporal_epoch_per_chip(
    placer::PlacerSolution const& placer_solution,
    DeviceConfig const& device_config
    ) 
{   
    std::unordered_map<int, std::unordered_map<uint32_t, placer::PlacerSolution::EpochId>> temporal_epoch_chip_id_to_global_epoch_id_map;
    for (uint32_t e = 0; e < placer_solution.num_epochs; e++) {
        temporal_epoch_chip_id_to_global_epoch_id_map[placer_solution.temporal_epoch_id(e)][placer_solution.epoch_id_to_chip.at(e)] = e;
    }

    auto available_cores = std::unordered_map<placer::PlacerSolution::EpochId, std::unordered_map<uint32_t, int>>{};
    int num_worker_cores = device_config.grid_size.r * device_config.grid_size.c;
    for (uint32_t i = 0; i < placer_solution.num_epochs; i++) 
    {
        int temporal_epoch = placer_solution.temporal_epoch_id(i);
        for (auto chip_id : device_config.chip_ids) {
            available_cores[temporal_epoch][chip_id] = num_worker_cores;
        }
    }

    for (auto const& [epoch_id, op_placements] : placer_solution.epoch_id_to_op_placement)
    {
        for (placer::OpPlacement const& placement : op_placements)
        {
            TT_ASSERT(static_cast<int>(placement.global_epoch_id) == static_cast<int>(epoch_id));
            std::uint32_t temporal_epoch = placer_solution.temporal_epoch_id(epoch_id);
            int chip_id = placer_solution.epoch_id_to_chip.at(placement.global_epoch_id);
            TT_ASSERT(static_cast<long>(temporal_epoch) == static_cast<long>(placer_solution.temporal_epoch_id(placement.name)));
            available_cores.at(temporal_epoch).at(chip_id) -= (placement.placed_cores.size_r() * placement.placed_cores.size_c());
            TT_ASSERT(available_cores.at(temporal_epoch).at(chip_id) >= 0, "More tensix cores used than are available");
        }
    }

    return available_cores;
}

void try_serialize(
    std::vector<producer_consumer_pair_t>::iterator& edge_iter,
    std::unordered_map<producer_consumer_pair_t, chip_to_chip_data_edge_t> const& chip_to_chip_data_edges,
    std::vector<data_edge_serialization_spec_t>& edges_to_serialize,
    std::unordered_map<chip_boundary_id_t, std::unordered_set<producer_consumer_pair_t>>&
        chip_boundary_producer_consumer_pairs,
    placer::PlacerSolution::EpochId target_epoch_id)
{
    auto const& data_edge = chip_to_chip_data_edges.at(*edge_iter);
    edges_to_serialize.push_back({*edge_iter, data_edge, target_epoch_id});

    // remove the edge from all chip-to-chip-boundaries from producer to consumer
    auto start = std::min(data_edge.producer_chip, data_edge.consumer_chip);
    auto end = std::max(data_edge.producer_chip, data_edge.consumer_chip);
    TT_ASSERT(end > start);
    for (auto c = start; c != end; c++)
    {
        auto const& chip_boundary = chip_boundary_id_t{c, c + 1};
        chip_boundary_producer_consumer_pairs.at(chip_boundary).erase(*edge_iter);
    }
}

void try_serialize_with_tensix_datacopy(
    std::vector<producer_consumer_pair_t>::iterator& edge_iter,
    int& stream_overage,
    int& chip_available_cores,
    std::unordered_map<producer_consumer_pair_t, chip_to_chip_data_edge_t> const& chip_to_chip_data_edges,
    std::vector<data_edge_serialization_spec_t>& edges_to_serialize,
    std::unordered_map<chip_boundary_id_t, std::unordered_set<producer_consumer_pair_t>>&
        chip_boundary_producer_consumer_pairs,
    placer::PlacerSolution::EpochId target_epoch_id,
    std::unordered_map<chip_boundary_id_t, int>& chip_boundary_needed_streams)
{
    auto const& data_edge = chip_to_chip_data_edges.at(*edge_iter);
    int streams_saved = data_edge.streams_needed_per_hop - 1;  // we still need a stream after serialization
    auto start = std::min(data_edge.producer_chip, data_edge.consumer_chip);
    auto end = std::max(data_edge.producer_chip, data_edge.consumer_chip);
    TT_ASSERT(end > start);
    for (auto c = start; c != end; c++)
    {
        auto const& chip_boundary = chip_boundary_id_t{c, c + 1};
        chip_boundary_needed_streams.at(chip_boundary) -= streams_saved;
    }

    try_serialize(
        edge_iter, chip_to_chip_data_edges, edges_to_serialize, chip_boundary_producer_consumer_pairs, target_epoch_id);

    stream_overage -= streams_saved;
    log_debug(
        "\tSaved {} streams. New overage: {}. For producer {} -> consumer {} @ port {} ",
        streams_saved,
        stream_overage,
        std::get<0>(*edge_iter),
        std::get<1>(*edge_iter),
        std::get<2>(*edge_iter));
    edge_iter++;
    chip_available_cores--;
};

template <bool SERIALIZE_WITH_TENSIX_DATACOPY>
static void serialize_edges_while_above_threshold(
    graphlib::Graph* graph,
    std::vector<producer_consumer_pair_t>::iterator& edge_iter,
    std::vector<producer_consumer_pair_t>::iterator edge_iter_end,
    int& stream_overage,
    int& producer_chip_available_cores,
    int& consumer_chip_available_cores,
    std::unordered_map<uint32_t, placer::PlacerSolution::EpochId> const& chip_id_to_epoch_map,
    placer::PlacerSolution const& placer_solution,
    std::unordered_map<producer_consumer_pair_t, chip_to_chip_data_edge_t> const& chip_to_chip_data_edges,
    std::vector<data_edge_serialization_spec_t>& edges_to_serialize,
    std::unordered_map<chip_boundary_id_t, std::unordered_set<producer_consumer_pair_t>>&
        chip_boundary_producer_consumer_pairs,
    std::unordered_map<chip_boundary_id_t, int>& chip_boundary_needed_streams)
{
    while (edge_iter != edge_iter_end && (!SERIALIZE_WITH_TENSIX_DATACOPY || (stream_overage > 0 && (producer_chip_available_cores > 0 || consumer_chip_available_cores > 0)))) 
    {
        auto [producer_name, consumer_name, operand_index] = *edge_iter;
        bool producer_is_q = placer_solution.name_to_op_placement.find(producer_name) == placer_solution.name_to_op_placement.end();
        bool consumer_is_q = placer_solution.name_to_op_placement.find(consumer_name) == placer_solution.name_to_op_placement.end();
        TT_ASSERT(!consumer_is_q || !producer_is_q);
        bool is_q_to_op = producer_is_q;
        bool producer_is_input = graph->get_node_by_name(producer_name)->node_type() == graphlib::NodeType::kInput;

        if (producer_is_input && SERIALIZE_WITH_TENSIX_DATACOPY) {
            // We can't currently support if producer is input because we need to be able to inherit characteristics from the producer (op) - currently we
            // can only inherit from ops.
            edge_iter++;
            continue;
        }
        
        uint32_t producer_chip = placer_solution.chip_id(producer_name);
        uint32_t consumer_chip = placer_solution.chip_id(consumer_name);
        placer::PlacerSolution::EpochId producer_epoch_id = chip_id_to_epoch_map.at(producer_chip);
        placer::PlacerSolution::EpochId consumer_epoch_id = chip_id_to_epoch_map.at(consumer_chip);
        log_trace(tt::LogPlacer,"\tProducer {} is on chip {} in epoch {}, consumer {} is on chip {} in epoch {} ", producer_name, producer_chip, producer_epoch_id, consumer_name, consumer_chip, consumer_epoch_id);
        // cleanup: separate concerns and handle incrementing separately from serializing
        if (SERIALIZE_WITH_TENSIX_DATACOPY)
        {
            if (is_q_to_op)
            {
                // only use the producer chip for q to -op since we may need to
                if (producer_chip_available_cores > 0)
                {
                    try_serialize_with_tensix_datacopy(
                        edge_iter,
                        stream_overage,
                        producer_chip_available_cores,
                        chip_to_chip_data_edges,
                        edges_to_serialize,
                        chip_boundary_producer_consumer_pairs,
                        producer_epoch_id,
                        chip_boundary_needed_streams);
                }
                else
                {
                    edge_iter++;
                }
            }
            else
            {
                if (producer_chip_available_cores > 0)
                {
                    try_serialize_with_tensix_datacopy(
                        edge_iter,
                        stream_overage,
                        producer_chip_available_cores,
                        chip_to_chip_data_edges,
                        edges_to_serialize,
                        chip_boundary_producer_consumer_pairs,
                        producer_epoch_id,
                        chip_boundary_needed_streams);
                }
                else if (consumer_chip_available_cores > 0)
                {
                    try_serialize_with_tensix_datacopy(
                        edge_iter,
                        stream_overage,
                        consumer_chip_available_cores,
                        chip_to_chip_data_edges,
                        edges_to_serialize,
                        chip_boundary_producer_consumer_pairs,
                        consumer_epoch_id,
                        chip_boundary_needed_streams);
                }
            }
        }
        else
        {
            log_trace(tt::LogPlacer, "\tSerializing edges while above threshold producer_epoch_id={} consumer_epoch_id={}", producer_epoch_id, consumer_epoch_id);
            try_serialize(
                edge_iter,
                chip_to_chip_data_edges,
                edges_to_serialize,
                chip_boundary_producer_consumer_pairs,
                producer_epoch_id); // the ethernet datacopy is "placed" on the same epoch as consumer
                                    // and the "dest_device" attribute denotes the consumer epoch chip
            ++edge_iter;
        }
    }
}

// get_directly_connected_ethernet_channels_between_chips
template <bool SERIALIZE_WITH_TENSIX_DATACOPY>
static std::vector<data_edge_serialization_spec_t> choose_chip_to_chip_data_edges_to_serialize(
    graphlib::Graph *graph,
    std::unordered_map<int, temporal_epoch_chip_to_chip_data_edges_t>& chip_to_chip_data_edges_per_temporal_epoch, 
    placer::PlacerSolution &placer_solution,
    balancer::BalancerSolution &balancer_solution,
    DeviceConfig const& device_config) 
{
    std::unordered_map<int, std::unordered_map<uint32_t, placer::PlacerSolution::EpochId>> temporal_epoch_chip_id_to_global_epoch_id_map;
    for (std::uint32_t e = 0; e < placer_solution.num_epochs; e++) {
        const auto &epoch_info = placer_solution.epoch_id_to_epoch_info.at(e);
        log_trace(tt::LogPlacer, "epoch {} has epoch_info(.global_epoch_id={}, .temporal_epoch_id={}, .spatial_epoch_id={}). epoch_id_to_chip -> {}", e, epoch_info.global_epoch_id, epoch_info.temporal_epoch_id, epoch_info.spatial_epoch_id, placer_solution.epoch_id_to_chip.at(e));
        temporal_epoch_chip_id_to_global_epoch_id_map[placer_solution.temporal_epoch_id(e)][placer_solution.epoch_id_to_chip.at(e)] = epoch_info.global_epoch_id;
    }

    // Temporary flag to pick correct number of eth links depending on setup, until general implementation comes
    bool eth_links_between_chips_nebula = (bool)env_as<int>("PYBUDA_ETH_LINKS_NEBULA", 0);

    // auto cluster_desc_uniq = tt_ClusterDescriptor::create_from_yaml(device_config.cluster_config_yaml);

    auto edges_to_serialize = std::vector<data_edge_serialization_spec_t>{};
    constexpr int ETH_STREAMS_PER_LINK = 8;
    auto available_cores_per_temporal_epoch = collect_available_cores_per_temporal_epoch_per_chip(placer_solution, device_config);
    
    for (auto& [temporal_epoch_id, temporal_epoch_chip_to_chip_data_edges_specs] : chip_to_chip_data_edges_per_temporal_epoch)
    {
        auto& chip_to_chip_data_edges = temporal_epoch_chip_to_chip_data_edges_specs.chip_to_chip_data_edges;
        auto& chip_boundary_producer_consumer_pairs = temporal_epoch_chip_to_chip_data_edges_specs.chip_boundary_producer_consumer_pairs;

        for (auto const& [chip_boundary, required_streams] : temporal_epoch_chip_to_chip_data_edges_specs.chip_boundary_needed_streams)
        {
            // Old code that only worked with topologies where adjacent chip IDs were always connected to each other
            // this isn't generally true. For now we hardcode to 4 links for galaxy setups and have a flag for nebula setups, but we should generalize
            // -> first to get the # links between any connected chips and assume that's the number for any other
            //    pair of connected chips. This generally true (assuming all links train on boot)
            // Then we need to update this pass to always serialize chip to chip if ethernet datacopy is enabled
            //auto const& links_between_chips = cluster_desc_uniq->get_directly_connected_ethernet_channels_between_chips(chip_boundary.first, chip_boundary.second);
            int eth_links_between_chips = eth_links_between_chips_nebula ? 2 : 4; //links_between_chips.size();
            TT_ASSERT(eth_links_between_chips >= 0, "Entries should only be produced for adjacent chips");
            int available_streams = eth_links_between_chips * ETH_STREAMS_PER_LINK;
            // For ethernet datacopy serialization, we serialize all chip to chip edges
            // For tensix datacopy we only can conditionally serialize to save cores
            if (SERIALIZE_WITH_TENSIX_DATACOPY && required_streams <= available_streams)
            {
                continue;
            }
            auto producer_consumer_pairs_sorted = std::vector<producer_consumer_pair_t>(chip_boundary_producer_consumer_pairs.at(chip_boundary).begin(), chip_boundary_producer_consumer_pairs.at(chip_boundary).end());
            TT_ASSERT(producer_consumer_pairs_sorted.size() > 0);
            std::sort(
                producer_consumer_pairs_sorted.begin(), 
                producer_consumer_pairs_sorted.end(), 
                [&chip_to_chip_data_edges] (auto const& pair_a, auto const& pair_b) 
                {
                    return chip_to_chip_data_edges.at(pair_a).streams_needed_per_hop > chip_to_chip_data_edges.at(pair_b).streams_needed_per_hop;
                }
            );
            TT_ASSERT(chip_to_chip_data_edges.at(producer_consumer_pairs_sorted.front()).streams_needed_per_hop >= chip_to_chip_data_edges.at(producer_consumer_pairs_sorted.back()).streams_needed_per_hop) ;
            
            // Choose which candidates to serialize
            int stream_overage = required_streams - available_streams;
            TT_ASSERT(!SERIALIZE_WITH_TENSIX_DATACOPY || stream_overage > 0);

            // There are a ton of different ways to choose which producer-consumer data edge to serialize but for
            // now we'll just default to choosing the one(s) that result in serializing the fewest edges get us to 
            // below the threshold. In practice this will mean choosing the largest ops first
            // Other options include:
            // - first choosing those with the largest overall stream usage (at the cost of perf)
            // or ... choosing the largest stream usage on the current edge
            // or ... iteratively serializing the smallest op grids until we meet the threshold

            auto const& producer_name = std::get<0>(producer_consumer_pairs_sorted.at(0));
            auto const& consumer_name = std::get<1>(producer_consumer_pairs_sorted.at(0));
            uint32_t producer_chip = placer_solution.chip_id(producer_name);
            uint32_t consumer_chip = placer_solution.chip_id(consumer_name);

            auto current_data_edge_iter = producer_consumer_pairs_sorted.begin();
            int temporal_epoch = get_producer_consumer_pair_temporal_epoch(placer_solution, producer_name, consumer_name);
            if (SERIALIZE_WITH_TENSIX_DATACOPY)
            {
                log_debug(
                    "Temporal epoch {} requires {} eth streams between chips {} and {} but {} are available by "
                    "default. {} empty cores on producer chip {}, {} empty cores on consumer chip {}",
                    static_cast<int>(temporal_epoch_id),
                    required_streams,
                    chip_boundary.first,
                    chip_boundary.second,
                    available_streams,
                    available_cores_per_temporal_epoch.at(temporal_epoch).at(producer_chip),
                    producer_chip,
                    available_cores_per_temporal_epoch.at(temporal_epoch).at(consumer_chip),
                    consumer_chip);
            }
            log_trace(tt::LogPlacer, "Serializing data edges between chips {} and {}", chip_boundary.first, chip_boundary.second);
            log_trace(tt::LogPlacer, "\tProducer op {}, Consumer op: {}", producer_name, consumer_name);
            serialize_edges_while_above_threshold<SERIALIZE_WITH_TENSIX_DATACOPY> (
                graph,
                current_data_edge_iter, 
                producer_consumer_pairs_sorted.end(),
                stream_overage, 
                available_cores_per_temporal_epoch.at(temporal_epoch).at(producer_chip),
                available_cores_per_temporal_epoch.at(temporal_epoch).at(consumer_chip),
                temporal_epoch_chip_id_to_global_epoch_id_map.at(temporal_epoch),
                placer_solution,
                chip_to_chip_data_edges, 
                edges_to_serialize, 
                chip_boundary_producer_consumer_pairs,
                temporal_epoch_chip_to_chip_data_edges_specs.chip_boundary_needed_streams);
        }
    }

    // Log the edges to serialize
    log_trace(tt::LogPlacer, "Chip to chip edges to serialize");
    for (auto const& [producer_consumer_pair, chip_to_chip_data_edge, epoch_id] : edges_to_serialize)
    {
        std::stringstream ss;
        auto const& producer_name = std::get<0>(producer_consumer_pair);
        auto const& consumer_name = std::get<1>(producer_consumer_pair);
        ss << producer_name << " -> " << consumer_name << " ";
        ss << "\n";

        Node *producer_op = graph->get_node_by_name(producer_name);
        ss << " producer_shape=(" << producer_op->shape().w() << "," << producer_op->shape().z() << "," << producer_op->shape().rt() << "," << producer_op->shape().ct() << "), ";
        if (producer_op->node_type() != graphlib::NodeType::kQueue) {

            tt_xy_pair const& producer_grid_size = get_op_or_queue_placed_grid_size(placer_solution, *producer_op);
            ss << " producer_grid_size=(" << producer_grid_size.y << "," << producer_grid_size.x << "), ";
            
            auto const& block_shape = balancer_solution.op_models.at(producer_name).output_buffers.at(0).block_shape;
            ss << " t=" << block_shape.t << ", ";
            ss << " mblock=(" << block_shape.mblock_m << "," << block_shape.mblock_n << "), ";
            ss << " ublock=(" << block_shape.ublock.rt << "," << block_shape.ublock.ct << ")";
        } else {
            ss << " producer is a queue - use producer shape info as reference.";
        }

        ss << "\n";

        Node *consumer_op = graph->get_node_by_name(consumer_name);
        ss << " consumer_shape=(" << consumer_op->shape().w() << "," << consumer_op->shape().z() << "," << consumer_op->shape().rt() << "," << consumer_op->shape().ct() << "), ";
        if (consumer_op->node_type() != graphlib::NodeType::kQueue) {

            tt_xy_pair const& consumer_grid_size = get_op_or_queue_placed_grid_size(placer_solution, *consumer_op);
            ss << " producer_grid_size=(" << consumer_grid_size.y << "," << consumer_grid_size.x << "), ";
            
            auto const& block_shape = balancer_solution.op_models.at(consumer_name).output_buffers.at(0).block_shape;
            ss << " t=" << block_shape.t << ", ";
            ss << " mblock=(" << block_shape.mblock_m << "," << block_shape.mblock_n << "), ";
            ss << " ublock=(" << block_shape.ublock.rt << "," << block_shape.ublock.ct << ")";
        } else {
            ss << ". consumer is a queue - use producer shape info as reference.";
        }

        log_trace(tt::LogPlacer, "{}", ss.str());
    }

    return edges_to_serialize;
}

std::tuple<Edge, graphlib::Node*, Edge> insert_serialized_dram_queue_between_ops(
    graphlib::Graph* graph,
    std::string const& producer_name,
    std::string const& consumer_name,
    graphlib::PortId consumer_input_port_id,
    int num_entries)
{
    std::stringstream name_ss;
    name_ss << producer_name << "_to_" << consumer_name << "_" << consumer_input_port_id << "_serialized_dram_queue";

    auto producer_node = graph->get_node_by_name(producer_name);

    if (num_entries < 0)
    {
        // num_entries < 0 means that num_entries wasn't set in the call of the method. Therefore we set it to
        // microbatch_size.
        num_entries = graph->get_microbatch();
    }
    graphlib::QueueNode *queue_node = graphlib::create_buffering_queue(graph, producer_node, name_ss.str(), num_entries);
    log_debug("\tCreating dram buffering queue node {} between {} and {}", name_ss.str(), producer_name, consumer_name);

    // Check port id, i.e. operand index if there is multiple edges between producer and consumer nodes
    std::uint32_t edge_index = 0;
    bool edge_found = false;
    std::vector<Edge> producer_outgoing_edges = graph->user_data_edges(producer_node);
    for (std::uint32_t i = 0; i < producer_outgoing_edges.size(); i++) {
        graphlib::NodeId producer_outgoing_node_id = producer_outgoing_edges[i].consumer_node_id;
        graphlib::Node* producer_outgoing_node = graph->node_by_id(producer_outgoing_node_id);
        if (producer_outgoing_node->name() == consumer_name && 
            producer_outgoing_edges[i].consumer_input_port_id == consumer_input_port_id) {
            edge_index = i;
            edge_found = true;
        }
    }

    TT_ASSERT(edge_found, "Edge with given consumer port id for given consumer node doesn't exist. ");

    Edge consumer_input_edge = producer_outgoing_edges[edge_index];

    Edge queue_input_edge = Edge(
            consumer_input_edge.producer_node_id,
            consumer_input_edge.producer_output_port_id,
            queue_node->id(),
            0,
            consumer_input_edge.edge_type);

    Edge queue_output_edge = Edge(
            queue_node->id(),
            0,
            consumer_input_edge.consumer_node_id,
            consumer_input_edge.consumer_input_port_id,
            consumer_input_edge.edge_type);
    graph->add_edge(queue_input_edge);
    graph->add_edge(queue_output_edge);
    graph->copy_edge_attributes(consumer_input_edge, queue_output_edge);
    graph->get_edge_attributes(queue_output_edge)->set_ublock_order(graph->get_edge_attributes(consumer_input_edge)->get_ublock_order());
    graph->remove_edge(consumer_input_edge);

    TT_ASSERT(graph->operand_data_edges(queue_node).size() == 1);

    return {queue_input_edge, queue_node, queue_output_edge};
}

static std::tuple<Edge,Node*,Edge> insert_datacopy_node(
    graphlib::Graph *graph, 
    Node *producer_op_node,
    std::string const& consumer_name, 
    graphlib::PortId operand_index
)
{
    std::string const& producer_op_name = producer_op_node->name();

    std::stringstream name_ss;
    name_ss << producer_op_name << "_serialized_to_" << consumer_name << "_" << operand_index;
    TT_ASSERT(producer_op_node != nullptr);

    auto consumer_node = graph->get_node_by_name(consumer_name);
    auto datacopy_node_uniq = producer_op_node->clone(name_ss.str());
    datacopy_node_uniq->set_node_type(graphlib::NodeType::kBudaOp);
    datacopy_node_uniq->set_output_df(producer_op_node->output_df());
    datacopy_node_uniq->as<graphlib::OpNode>()->change_op_type("nop");
    auto datacopy_node = graph->add_node(std::move(datacopy_node_uniq), graph->get_subgraph_id_for_node(producer_op_node->id()));
    TT_ASSERT(graph->operand_data_edges(datacopy_node).size() == 0, "Expected no operands yet");
    Edge consumer_input_edge = graph->operand_data_edges(consumer_node).at(operand_index);
    
    TT_ASSERT(consumer_input_edge.consumer_input_port_id == operand_index);

    // Insert the datacopy on the edge (`insert_node_on_edge` does some extra things we don't want)
    Edge datacopy_input_edge = Edge(
            consumer_input_edge.producer_node_id,
            consumer_input_edge.producer_output_port_id,
            datacopy_node->id(),
            0,
            consumer_input_edge.edge_type);

    Edge datacopy_output_edge_before_dram = Edge(
            datacopy_node->id(),
            0,
            consumer_input_edge.consumer_node_id,
            consumer_input_edge.consumer_input_port_id,
            consumer_input_edge.edge_type);
    graph->add_edge(datacopy_input_edge);
    graph->add_edge(datacopy_output_edge_before_dram);
    graph->copy_edge_attributes(consumer_input_edge, datacopy_output_edge_before_dram); // fails on lookup of datacopy_output_edge
    graph->get_edge_attributes(datacopy_output_edge_before_dram)->set_ublock_order(graph->get_edge_attributes(consumer_input_edge)->get_ublock_order());
    auto datacopy_input_block_order = graph->get_edge_attributes(consumer_input_edge)->get_ublock_order();
    graph->get_edge_attributes(datacopy_input_edge)->set_ublock_order(datacopy_input_block_order);
    graph->remove_edge(consumer_input_edge);

    auto const& datacopy_input_edges = graph->operand_edges(datacopy_node);
    TT_ASSERT(datacopy_input_edges.size() == 1, "Expected datacopy to only have 1 operand but it has " + std::to_string(datacopy_input_edges.size()));

    return {datacopy_input_edge, datacopy_node, datacopy_output_edge_before_dram};
}

static std::tuple<Edge, Node*, Edge> insert_ethernet_datacopy_node(
    graphlib::Graph* graph, placer::PlacerSolution const& placer_solution, Node* producer_op_node, std::string const& consumer_name, graphlib::PortId operand_index)
{
    std::string const& producer_op_name = producer_op_node->name();

    std::stringstream name_ss;
    name_ss << producer_op_name << "_eth_datacopy_to_" << consumer_name << "_" << operand_index;
    TT_ASSERT(producer_op_node != nullptr);

    auto consumer_node = graph->get_node_by_name(consumer_name);
    uint32_t consumer_chip = placer_solution.chip_id(consumer_name);
    // Want to clone the consumer because the consumer will have the applied TM
    // shapes since we want to push all TMs upwards to the ethernet datacopy
    // operand edge
    // This is the baseline scenario. We can get more sophisticated later if we want and move TMs
    // around
    if (producer_op_node->node_type() == tt::graphlib::NodeType::kQueue) {
        producer_op_node = graph->operands(producer_op_node).at(0);
    }
    graphlib::Node *datacopy_node = nullptr;
    if (producer_op_node->node_type() == tt::graphlib::NodeType::kQueue || producer_op_node->node_type() == tt::graphlib::NodeType::kInput) {
        datacopy_node = graph->add_node(consumer_node->clone(name_ss.str()), graph->get_subgraph_id_for_node(consumer_node->id()));
    } else {
        auto datacopy_node_uniq = producer_op_node->clone(name_ss.str());
        datacopy_node_uniq->as<graphlib::OpNode>()->change_op_type("ethernet_datacopy");
        datacopy_node = graph->add_node(std::move(datacopy_node_uniq), graph->get_subgraph_id_for_node(producer_op_node->id()));
    }

    datacopy_node->set_node_type(graphlib::NodeType::kBudaOp);
    datacopy_node->set_output_df(producer_op_node->output_df());

    auto ethernet_datacopy_op_attrs = BudaOpAttrs();
    ethernet_datacopy_op_attrs["dest_device"] = static_cast<int>(consumer_chip);
    datacopy_node->as<graphlib::OpNode>()->change_op_type(
        graphlib::OpType("ethernet_datacopy", {}, ethernet_datacopy_op_attrs));

    TT_ASSERT(graph->operand_data_edges(datacopy_node).size() == 0, "Expected no operands yet");
    Edge consumer_input_edge = graph->operand_data_edges(consumer_node).at(operand_index);
    TT_ASSERT(consumer_input_edge.consumer_input_port_id == operand_index);

    // Insert the datacopy on the edge (`insert_node_on_edge` does some extra things we don't want)
    Edge datacopy_input_edge = Edge(
        consumer_input_edge.producer_node_id,
        consumer_input_edge.producer_output_port_id,
        datacopy_node->id(),
        0,
        consumer_input_edge.edge_type);

    Edge datacopy_output_edge = Edge(
        datacopy_node->id(),
        0,
        consumer_input_edge.consumer_node_id,
        consumer_input_edge.consumer_input_port_id,
        consumer_input_edge.edge_type);
    graph->add_edge(datacopy_input_edge);
    graph->add_edge(datacopy_output_edge);
    graph->copy_edge_attributes(consumer_input_edge, datacopy_output_edge); // fails on lookup of datacopy_output_edge
    auto datacopy_input_block_order = graph->get_edge_attributes(consumer_input_edge)->get_ublock_order();
    graph->get_edge_attributes(datacopy_output_edge)
        ->set_ublock_order(datacopy_input_block_order);
    graph->get_edge_attributes(datacopy_input_edge)->set_ublock_order(datacopy_input_block_order);
    graph->remove_edge(consumer_input_edge);

    auto const& datacopy_input_edges = graph->operand_edges(datacopy_node);
    TT_ASSERT(
        datacopy_input_edges.size() == 1,
        "Expected datacopy to only have 1 operand but it has " + std::to_string(datacopy_input_edges.size()));

    return {datacopy_input_edge, datacopy_node, datacopy_output_edge};
}

template <bool SERIALIZE_WITH_TENSIX_DATACOPY>
static void add_datacopy_placement_entry(
    graphlib::Graph *graph, 
    placer::PlacerSolution &placer_solution,
    placer::PlacerSolution::EpochId epoch_id,
    std::vector<tt_xy_pair>& epoch_available_cores,
    Node const* datacopy_node)
{

    placer::Coord placed_cores_start = placer::Coord{.row = 0, .col = 0};
    placer::Coord placed_cores_end = placer::Coord{.row = 1, .col = 1};

    if constexpr (SERIALIZE_WITH_TENSIX_DATACOPY)
    {
        TT_ASSERT(epoch_available_cores.size() > 0);
        auto const& available_core = epoch_available_cores.front();
        placed_cores_start = placer::Coord{
            .row = static_cast<uint32_t>(available_core.y), .col = static_cast<uint32_t>(available_core.x)};
        placed_cores_end = placer::Coord{.row = placed_cores_start.row + 1, .col = placed_cores_start.col + 1};

        epoch_available_cores.erase(epoch_available_cores.begin());
    }
    auto operands = graph->operands(datacopy_node);
    TT_ASSERT(operands.size() == 1);
    auto operand_node = operands.at(0);
    auto users = graph->users(datacopy_node);
    TT_ASSERT(users.size() == 1);
    auto user_node = users.at(0);

    uint32_t target_chip = placer_solution.chip_id(operand_node->name());

    // Ethernet datacopy always gets placed on producer epoch
    bool operand_node_is_queue_or_input = operand_node->node_type() == graphlib::NodeType::kQueue || operand_node->node_type() == graphlib::NodeType::kInput;
    std::string const& operand_name = operand_node->name();
    auto placement_id = !operand_node_is_queue_or_input
                            ? placer_solution.name_to_op_placement.at(operand_name).epoch_id()
                            : placer_solution.name_to_op_placement.at(user_node->name()).epoch_id();

    //
    // Add the placement entry:
    //
    TT_ASSERT(epoch_id >= 0, "Invalid value for conversion");
    // Ethernet datacopy isn't placed on a tensix core so it doesn't have a valid XY. Otherwise we need to check for
    // valid XY
    TT_ASSERT(!SERIALIZE_WITH_TENSIX_DATACOPY || placed_cores_start.row < UINT_MAX, "Invalid value for conversion");
    TT_ASSERT(!SERIALIZE_WITH_TENSIX_DATACOPY || placed_cores_start.col < UINT_MAX, "Invalid value for conversion");
    auto const& placement = placer::OpPlacement{
        .id = placement_id,
        .name = datacopy_node->name(),
        .chip_id = target_chip,
        .global_epoch_id = static_cast<std::uint32_t>(epoch_id),
        .grid_transpose = false,
        .placed_cores = placer::CoordRange{
            .start = placed_cores_start,
            .end = placed_cores_end,
        }};
    placer_solution.name_to_op_placement.insert({datacopy_node->name(), placement});
    placer_solution.epoch_id_to_op_placement.at(epoch_id).push_back(placement);
    log_trace(tt::LogPlacer, "\tAdded ethernet datacopy placement for {} at global epoch {}, chip {}", datacopy_node->name(), epoch_id, target_chip);
}

static balancer::TStreamFactor set_datacopy_tstream_factor(
    graphlib::Graph* graph,
    balancer::BalancerSolution& balancer_solution,
    Node* producer_op_node,
    Node* /*consumer_op_node*/,
    Edge datacopy_input_edge,
    Node* datacopy_node)
{
    // 
    // Set tstreaming values for datacopy:
    //

    bool use_consumer_as_reference = false;
    if (producer_op_node->node_type() == tt::graphlib::kQueue) {
        producer_op_node = graph->operands(producer_op_node).at(0);
    }
    int tstream_r_factor = datacopy_node->shape().rt();
    int tstream_c_factor = datacopy_node->shape().ct();
    TT_ASSERT(tstream_r_factor > 0 && tstream_c_factor);
    log_trace(tt::LogPlacer, "producer grid size: {}r,{}c", balancer_solution.op_models.at(producer_op_node->name()).grid_shape.r, balancer_solution.op_models.at(producer_op_node->name()).grid_shape.c);
    log_trace(tt::LogPlacer, "Setting tstream factor for datacopy {} to {}r,{}c. Datacopy shape = (rt={},ct={}), reference_block is {}.",
        datacopy_node->name(), tstream_r_factor, tstream_c_factor,
        datacopy_node->shape().rt(), datacopy_node->shape().ct(),
        (use_consumer_as_reference ? "consumer": "producer"));

    auto input_edge_attr = graph->get_edge_attributes(datacopy_input_edge);
    auto datacopy_ublock_direction = input_edge_attr->get_ublock_order();
    auto datacopy_tstream_direction = (datacopy_ublock_direction == graphlib::UBlockOrder::R) ? balancer::TStreamDir::R : balancer::TStreamDir::C;
    auto datacopy_tstream_factor = balancer::TStreamFactor(datacopy_tstream_direction, tstream_r_factor, tstream_c_factor);

    return datacopy_tstream_factor;
}

static void add_datacopy_balancer_entry(
    graphlib::Graph *graph, 
    // placer::PlacerSolution& placer_solution,
    balancer::BalancerSolution& balancer_solution,
    DeviceConfig const& device_config,
    Node *producer_op_node,
    Edge datacopy_input_edge,
    Node *datacopy_node,
    balancer::TStreamFactor const& datacopy_tstream_factor)
{
    // First create the op model
    if (producer_op_node->node_type() == tt::graphlib::kQueue) {
        producer_op_node = graph->operands(producer_op_node).at(0);
    }
    auto datacopy_input_block_order = graph->get_edge_attributes(datacopy_input_edge)->get_ublock_order();
    std::size_t dst_size_tiles = balancer::calculate_dst_size_tiles(
        device_config.get_dst_size(), producer_op_node->output_df(), producer_op_node->shape().get_tile_volume());
    std::string customErrorMessage;

    // We don't care about caches here, so we create a dummy cache
    //
    auto dummy_cache = std::make_shared<balancer::BalancerCacheCollection>();

    auto [datacopy_op_model, failure_reason] = balancer::legalizer::calculate_op_model(
        graph,
        dummy_cache,
        datacopy_node->as<graphlib::BudaOpNode>(),
        balancer::GridShape(1, 1),
        datacopy_tstream_factor,
        datacopy_input_block_order,
        false, /*force_dram_parameters,*/
        dst_size_tiles,
        device_config.get_l1_size(), /*std::size_t l1_usable_size*/
        0 /*std::size_t dram_channel_capacity*/,
        customErrorMessage);
    if (failure_reason != tt::balancer::OpModelFailureReason::NoFailure) {
        graph->dump("eth_serialization_failure_" + datacopy_node->name());
        tt::log_error("Calculate op model failed for ethernet datacopy op {} with reason {}", datacopy_node->name(), failure_reason);
    }
    if (datacopy_op_model.output_buffers.size() == 0) {
        graph->dump("eth_serialization_failure_" + datacopy_node->name());
        tt::log_error("Calculate op model failed for ethernet datacopy op {}. No output buffers created", datacopy_node->name());
    }
    
    balancer_solution.block_shapes.insert({datacopy_node->name(), datacopy_op_model.block_shape()});
    balancer_solution.op_models.insert({datacopy_node->name(), datacopy_op_model});
}

template <bool SERIALIZE_WITH_TENSIX_DATACOPY>
static std::tuple<Edge, graphlib::Node*, Edge, balancer::TStreamFactor> insert_datacopy_between_ops(
    graphlib::Graph *graph, 
    placer::PlacerSolution &placer_solution, 
    balancer::BalancerSolution& balancer_solution, 
    DeviceConfig const& device_config,
    std::string const& producer_op_or_queue_name,
    std::string const& consumer_name, 
    graphlib::PortId operand_index, 
    placer::PlacerSolution::EpochId epoch_id,
    std::vector<tt_xy_pair>& epoch_available_cores)
{
    // this must be an op since we inherit attributes from it. It may not be the topologically connected operand if there is a queue in between
    auto producer_op_node = graph->get_node_by_name(producer_op_or_queue_name);

    auto [datacopy_input_edge, datacopy_node, datacopy_output_edge_before_dram] =
        SERIALIZE_WITH_TENSIX_DATACOPY
            ? insert_datacopy_node(graph, producer_op_node, consumer_name, operand_index)
            : insert_ethernet_datacopy_node(graph, placer_solution, producer_op_node, consumer_name, operand_index);

    add_datacopy_placement_entry<SERIALIZE_WITH_TENSIX_DATACOPY>(graph, placer_solution, epoch_id, epoch_available_cores, datacopy_node);

    auto datacopy_tstream_factor = set_datacopy_tstream_factor(graph, balancer_solution, producer_op_node, graph->get_node_by_name(consumer_name)/*producer_op_node*/, datacopy_input_edge, datacopy_node);

    bool consumer_is_op = placer_solution.name_to_op_placement.find(consumer_name) != placer_solution.name_to_op_placement.end();
    Edge final_output_edge = datacopy_output_edge_before_dram;
    if (consumer_is_op) {
        TT_ASSERT(graph->operand_data_edges(datacopy_node).size() == 1);
        auto [datacopy_output_edge, queue_node, queue_output_edge] = insert_serialized_dram_queue_between_ops(
            graph, 
            datacopy_node->name(),
            consumer_name, 
            operand_index);
        final_output_edge = queue_output_edge;
    } else {
        // Delete the placement entry since we are serializing the buffer and need to redo it's placement/allocation
        // Might not need to do this for eth datacopy
        placer_solution.name_to_queue_placement.erase(consumer_name);
    }

    add_datacopy_balancer_entry(
        graph, 
        balancer_solution, 
        device_config, 
        producer_op_node,
        datacopy_input_edge, 
        datacopy_node, 
        datacopy_tstream_factor);

    TT_ASSERT(graph->operand_data_edges(datacopy_node).size() == 1);
    TT_ASSERT(graph->user_data_edges(datacopy_node).size() == 1);

    return {datacopy_input_edge, datacopy_node, final_output_edge, datacopy_tstream_factor};
}

static std::unordered_map<placer::PlacerSolution::EpochId, std::vector<tt_xy_pair>> collect_available_cores_per_epoch(
    placer::PlacerSolution const& placer_solution,
    DeviceConfig const& device_config
) {
    auto available_cores = std::unordered_map<placer::PlacerSolution::EpochId, std::vector<tt_xy_pair>>{};
    auto occupied_cores = std::unordered_map<placer::PlacerSolution::EpochId, std::unordered_set<tt_xy_pair>>{};
    int num_epochs = placer_solution.epoch_id_to_op_placement.size();
    for (int i = 0; i < num_epochs; i++) 
    {
        available_cores[i] = {};
        occupied_cores[i] = {};
        for (auto const& placement : placer_solution.epoch_id_to_op_placement.at(i)) 
        {
            int start_row = placement.placed_cores.start.row;
            int end_row = placement.placed_cores.end.row;
            int start_col = placement.placed_cores.start.col;
            int end_col = placement.placed_cores.end.col;
            for (int r = start_row; r < end_row; r++) 
            {
                for (int c = start_col; c < end_col; c++) 
                {
                    occupied_cores[i].insert(tt_xy_pair(c,r));
                }
            }
        }
    }

    int chip_grid_r = device_config.grid_size.r;
    int chip_grid_c = device_config.grid_size.c;
    for (int i = 0; i < num_epochs; i++) 
    {
        auto const& epoch_used_cores = occupied_cores.at(i);
        auto& epoch_available_cores = available_cores[i];
        for (int r = 0; r < chip_grid_r; r++) 
        {
            for (int c = 0; c < chip_grid_c; c++) 
            {
                tt_xy_pair const& core = tt_xy_pair(c,r);
                bool core_unoccupied = epoch_used_cores.find(core) == epoch_used_cores.end();
                if (core_unoccupied) 
                {
                    epoch_available_cores.push_back(core);
                } 
            }
        }
    }

    return available_cores;
}

static void update_tstreaming_factors(
    graphlib::Graph *graph, 
    placer::PlacerSolution &placer_solution, 
    balancer::BalancerSolution& balancer_solution, 
    Edge datacopy_operand_edge,
    Edge consumer_new_operand_edge, 
    balancer::TStreamFactor const& datacopy_tstream_factor) 
{
    auto get_producer_tstreaming_factor = [&](graphlib::Graph *graph, balancer::BalancerSolution const& balancer_solution, Edge const& datacopy_operand_edge) -> balancer::TStreamFactor {
        auto op_producer_node = graph->node_by_id(datacopy_operand_edge.producer_node_id);
        if (op_producer_node->node_type() == graphlib::NodeType::kInput) {
            return balancer::TStreamFactor{balancer::TStreamDir::R, 1, 1};
        }
        if (op_producer_node->node_type() == graphlib::NodeType::kQueue) {
            op_producer_node = graph->data_operands(op_producer_node).at(0);
        }
        balancer::OpModel const& producer_op_model = balancer_solution.op_models.at(op_producer_node->name());
        return producer_op_model.t_stream_factor;
    };

    auto producer_t_stream_factor = get_producer_tstreaming_factor(graph, balancer_solution, datacopy_operand_edge);
    auto datacopy_input_edge_attr = graph->get_edge_attributes(datacopy_operand_edge);
    log_debug("update_tstreaming_factors for serializing datacopy op {}. Producer t-stream factor: {}. Consumer t-stream factor: {}",
        graph->node_by_id(datacopy_operand_edge.consumer_node_id)->name(), producer_t_stream_factor, datacopy_tstream_factor);
    constexpr bool kAfterTranspose = false;
    insert_t_stream_tms_for_eltwise(
        datacopy_input_edge_attr->get_tms(), datacopy_tstream_factor, producer_t_stream_factor, kAfterTranspose);

    auto const& consumer_name = graph->node_by_id(consumer_new_operand_edge.consumer_node_id)->name();
    bool consumer_is_op = placer_solution.name_to_op_placement.find(consumer_name) != placer_solution.name_to_op_placement.end();
    if (consumer_is_op) 
    {
        Node* consumer_node = graph->get_node_by_name(consumer_name);
        auto consumer_input_edge_attr = graph->get_edge_attributes(consumer_new_operand_edge);
        balancer::OpModel const& consumer_op_model = balancer_solution.op_models.at(consumer_node->name());
        auto consumer_t_stream_factor = consumer_op_model.t_stream_factor;

        log_debug("update_tstreaming_factors for consumer {} (from inserted ethernet serializing op {}). Producer t-stream factor: {}. Consumer t-stream factor: {}",
            consumer_node->name(), graph->node_by_id(datacopy_operand_edge.consumer_node_id)->name(), datacopy_tstream_factor, consumer_t_stream_factor);
        // We don't need to specify the consumer t-stream factor because it has already been calculated. If we pass the consumer factor in
        // this function will generate the net/merged t factor required to get from producer->consumer and won't take into account already
        // existing t stream tms introduced by the t stream pass originally
        insert_t_stream_tms_for_eltwise(consumer_input_edge_attr->get_tms(), {}/*consumer_t_stream_factor*/, datacopy_tstream_factor, kAfterTranspose);
    } 
    else 
    {
        for (auto e2e_user_edge : graph->user_data_edges(graph->get_node_by_name(consumer_name))) 
        {
            Node* consumer_node = graph->node_by_id(e2e_user_edge.consumer_node_id);
            balancer::OpModel const& consumer_op_model = balancer_solution.op_models.at(consumer_node->name());
            auto consumer_t_stream_factor = consumer_op_model.t_stream_factor;
            auto consumer_input_edge_attr = graph->get_edge_attributes(e2e_user_edge);
            log_debug("update_tstreaming_factors for e2e consumer {} (from inserted ethernet serializing op {}). Producer t-stream factor: {}. Consumer t-stream factor: {}",
                consumer_node->name(), graph->node_by_id(datacopy_operand_edge.consumer_node_id)->name(), datacopy_tstream_factor, consumer_t_stream_factor);
            insert_t_stream_tms_for_eltwise(consumer_input_edge_attr->get_tms(), {}/*consumer_t_stream_factor*/, datacopy_tstream_factor, kAfterTranspose);
        }
    }
}

template <bool SERIALIZE_WITH_TENSIX_DATACOPY>
static void serialize_chosen_chip_to_chip_data_edges(
    graphlib::Graph *graph, 
    placer::PlacerSolution &placer_solution, 
    balancer::BalancerSolution& balancer_solution,
    DeviceConfig const& device_config,
    std::vector<data_edge_serialization_spec_t> const& edges_to_serialize)
{
    std::unordered_map<placer::PlacerSolution::EpochId, std::vector<tt_xy_pair>> epoch_available_cores = {};
    if constexpr (SERIALIZE_WITH_TENSIX_DATACOPY)
    {
        epoch_available_cores = collect_available_cores_per_epoch(placer_solution, device_config);
    }

    std::vector<std::tuple<Edge, graphlib::Node*, Edge, balancer::TStreamFactor>> serialized_edge_specs;
    for (auto const &[producer_consumer_pair, data_edge, target_epoch_id] : edges_to_serialize) 
    {
        auto const &producer_name = std::get<0>(producer_consumer_pair);
        auto const &consumer_name = std::get<1>(producer_consumer_pair);

        // Insert the tensor serializing datacopy/nop op node
        TT_ASSERT(
            !SERIALIZE_WITH_TENSIX_DATACOPY ||
            epoch_available_cores.find(target_epoch_id) != epoch_available_cores.end());
        auto const& serialized_edge_spec = insert_datacopy_between_ops<SERIALIZE_WITH_TENSIX_DATACOPY>(
            graph,
            placer_solution,
            balancer_solution,
            device_config,
            producer_name,
            consumer_name,
            data_edge.operand_index,
            target_epoch_id,
            epoch_available_cores[target_epoch_id]);
        serialized_edge_specs.push_back(serialized_edge_spec);
    }

    for (auto [datacopy_operand_edge, datacopy_node, consumer_new_operand_edge, datacopy_tstream_factor] : serialized_edge_specs)
    {
        update_tstreaming_factors(
            graph, 
            placer_solution, 
            balancer_solution, 
            datacopy_operand_edge, 
            consumer_new_operand_edge, 
            datacopy_tstream_factor);
    }
}

void deallocate_dynamic_buffers(
    graphlib::Graph* graph, placer::DramPlacerConfig const& config, placer::PlacerSolution& placer_solution)
{
    log_debug("Eth stream reduction, deallocating dynamic buffers so they can be reallocated with serialized eth buffers");

    for (auto &[name, placement] : placer_solution.name_to_queue_placement)
    {
        auto node = graph->get_node_by_name(placement.name);
        bool output_on_host = is_output_host_queue(config.output_queues_on_host, graph, node);
        bool is_dynamic_queue = !output_on_host && !placement.is_static();

        if (is_dynamic_queue) 
        {
            log_debug("Deallocating dynamic buffer {}", node->name());
            placement.dram_buffers.clear();
            placement.epoch_allocate = -1;
            placement.epoch_deallocate = -1;
            TT_ASSERT(placer_solution.name_to_queue_placement.at(node->name()).dram_buffers.size() == 0);
        }

    }
}

void reduce_ethernet_stream_usage(
    PostPlacerConfig& config,
    graphlib::Graph* graph,
    balancer::BalancerSolution& balancer_solution,
    placer::PlacerSolution& placer_solution,
    DeviceConfig const& device_config)
{
    bool tensix_datacopy_eth_link_serialization_enabled = env_as<bool>("PYBUDA_ENABLE_ETH_SERIALIZATION");
    auto chip_to_chip_data_edges = collect_chip_to_chip_data_edges_per_temporal_epoch(graph, placer_solution);

    if (tensix_datacopy_eth_link_serialization_enabled)
    {
        auto const& edges_to_serialize = choose_chip_to_chip_data_edges_to_serialize<true>(
            graph, chip_to_chip_data_edges, placer_solution, balancer_solution, device_config);
        serialize_chosen_chip_to_chip_data_edges<true>(
            graph, placer_solution, balancer_solution, device_config, edges_to_serialize);
        // Deallocate here so we can reallocate them alongside the serialized buffers. Otherwise when we try to allocate
        // the serialized buffers we won't know the lifetimes of the previous dynamic buffers which will likely cause us
        // to allocate in overlapping memory regions
        deallocate_dynamic_buffers(graph, config.dram_placer_config, placer_solution);
    }
    else
    {
        auto const& edges_to_serialize = choose_chip_to_chip_data_edges_to_serialize<false>(
            graph, chip_to_chip_data_edges, placer_solution, balancer_solution, device_config);
        serialize_chosen_chip_to_chip_data_edges<false>(
            graph, placer_solution, balancer_solution, device_config, edges_to_serialize);
        // Deallocate here so we can reallocate them alongside the serialized buffers. Otherwise when we try to allocate
        // the serialized buffers we won't know the lifetimes of the previous dynamic buffers which will likely cause us
        // to allocate in overlapping memory regions
        deallocate_dynamic_buffers(graph, config.dram_placer_config, placer_solution);
    }
}

}; // namespace tt
