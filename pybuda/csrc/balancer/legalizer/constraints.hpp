// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#include "backend_api/device_config.hpp"
#include "balancer/balancer_cache_collection.hpp"
#include "balancer/types.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "utils/assert.hpp"
#include "utils/env.hpp"

namespace tt::graphlib
{
class Graph;
enum class EdgeType;
using EdgeUniqueId = std::tuple<NodeId, PortId, NodeId, PortId, EdgeType>;
}  // namespace tt::graphlib

namespace tt::balancer::legalizer
{
class GraphSolver;

// New constraint failure reason can be defined after initial NoConstraintFailure value but prior to
// MaxConstraintFailureReason.
//
// clang-format off
#define EdgeConstraintFailureReasons \
    X (NoConstraintFailure, "No Failure - Valid path") \
    X (Failed, "Failed") \
    X (TStreamIncompatibleConsumer, "Tstream consumer not compatible with producer TStream factor") \
    X (TStreamIllegalTStreamDir, "TStream direction illegal") \
    X (TStreamPipegenDoubleCounted, "TStream pipe cannot scatter to more than 4 destinations due to constraints with forking") \
    X (TStreamBlockIncompatible, "TStream block shape incompatible") \
    X (TStreamAntiUblockOrderStacking, "TStream anti ublock order stacking is dissallowed") \
    X (TStreamMatmulStreamingSide, "TStream if matmul is streaming then producer can only stream from respective side LHS=TStreamDir::R, RHS=TStreamDir::C") \
    X (TStreamNotDivisableRow, "TStream stack/grid not divisible - Row") \
    X (TStreamNotDivisableColumn, "TStream stack/grid not divisible - Column") \
    X (ConvSliceStackMatmulStreamingHang, "For convs there are slice/stack tms that immediately follow a sparse matmul which causes hangs on silicon if we then go into matmul streaming") \
    X (MaxCostExceeded, "Cost exceeded, cost too high") \
    X (EdgePathRemovedByPriorEdgeElimination, "Valid path removed by edge elimination") \
    X (OpToQueueMapping, "Special case for generating valid Op to Queue paths") \
    X (ExceedsDRAMInQueues, "Exceeds the maximum number of DRAM input queues per core") \
    X (MaxConstraintFailureReason, "")
// clang-format on

#define X(a, b) a,
enum ConstraintFailureReason
{
    EdgeConstraintFailureReasons
};
#undef X

#define X(a, b) b,
static std::string ConstraintFailureReasonDesc[] = {EdgeConstraintFailureReasons};
#undef X

#ifdef DEBUG
class EdgeConstraintDebugInfo
{
   private:
    int constraintFailureCountByType[MaxConstraintFailureReason];
    std::unordered_map<const graphlib::EdgeUniqueId, int, EdgeUniqueIdHash> eliminatingEdges;

   public:
    EdgeConstraintDebugInfo() : constraintFailureCountByType() {}
    void recordEdgeConstraintFailure(ConstraintFailureReason failureReason)
    {
        TT_ASSERT(failureReason < MaxConstraintFailureReason, "Invalid failure reason.");
        constraintFailureCountByType[failureReason]++;
    }

    int getConstraintFailureCountByType(ConstraintFailureReason failureReason) const
    {
        return constraintFailureCountByType[failureReason];
    }

    void addEliminatingEdge(const graphlib::EdgeUniqueId edge);
    std::string toString(const graphlib::Graph* = nullptr) const;
};
#endif

struct EdgeCost
{
    // TODO: Read all of these constants from DeviceConfig
    // tenstorrent/budabackend#2345
    static constexpr int kMaxBytesPerPhase = 38;
    static constexpr int kBackendReservedInQueues = 1;
    static constexpr int kMaxDRAMInQueues = 40 - kBackendReservedInQueues;
    static constexpr int kMaxDRAMOutQueues = 8;
    static constexpr int kMaxFanOutStreams = 16;
    int kMaxStreamPhasesProducer = 0;
    int kMaxStreamPhasesConsumer = 0;

    std::uint16_t fan_out_streams = 0;
    std::uint16_t producer_stream_phases = 0;
    std::uint16_t consumer_stream_phases = 0;
    std::uint16_t consumer_dram_in_queues = 0;

    EdgeCost() = default;

    EdgeCost(
        std::uint16_t fan_out_streams,
        std::uint16_t producer_stream_phases,
        std::uint16_t consumer_stream_phases,
        std::uint16_t consumer_dram_in_queues,
        const DeviceConfig* device_config = nullptr,
        const OpModel* producer = nullptr,
        const OpModel* consumer = nullptr) :
        fan_out_streams(fan_out_streams),
        producer_stream_phases(producer_stream_phases),
        consumer_stream_phases(consumer_stream_phases),
        consumer_dram_in_queues(consumer_dram_in_queues)
    {
        kMaxStreamPhasesProducer = calculate_max_stream_phases(producer, device_config);
        kMaxStreamPhasesConsumer = calculate_max_stream_phases(consumer, device_config);
    }

    int calculate_max_stream_phases(const OpModel* op_model, const DeviceConfig* device_config)
    {
        // Set default overlay size to BBE reserved space (64kb)
        // If there is space in given op model (producer or consumer), assume double the space, but don't edit op model
        // yet, we do that after we select the op models, as the usage of the space is dependent on the
        // producer/consumer op models.
        // We divide all the space by 2 because we want to give half to producers and half to consumers.

        if (not op_model)
        {
            return 0;  // Default value should be 0 as queues can be on the ends of these edges
        }

        // All calls with op_model should have device_config provided
        TT_ASSERT(device_config);
        TT_ASSERT(op_model->overlay_size == 0, "Expected overlay size to be set do default value of 0.");

        // TODO: read from device config
        // tenstorrent/budabackend#2344
        //
        constexpr int default_overlay_size = 1 << 16;  // 64kB
        constexpr int overlay_size_to_add = 1 << 16;   // 64kB

        // If there is a global override, use that one instead
        static int global_overlay_size_to_add = device_config->get_overlay_blob_extra_size();
        if (global_overlay_size_to_add)
        {
            return ((default_overlay_size + global_overlay_size_to_add) / kMaxBytesPerPhase) / 2;
        }

        int available_l1_space = device_config->get_l1_usable_size() - op_model->get_l1_memory_usage();
        if (available_l1_space >= overlay_size_to_add)
        {
            return ((default_overlay_size + overlay_size_to_add) / kMaxBytesPerPhase) / 2;
        }
        else
        {
            return (default_overlay_size / kMaxBytesPerPhase) / 2;
        }
    }

    bool exceeded() const
    {
        return (fan_out_streams > kMaxFanOutStreams) or (producer_stream_phases > kMaxStreamPhasesProducer) or
               (consumer_stream_phases > kMaxStreamPhasesConsumer) or (consumer_dram_in_queues > kMaxDRAMInQueues);
    }

    static EdgeCost producer_sum(EdgeCost a, EdgeCost b)
    {
        return EdgeCost(
            a.fan_out_streams + b.fan_out_streams, a.producer_stream_phases + b.producer_stream_phases, 0, 0);
    }

    static EdgeCost consumer_sum(EdgeCost a, EdgeCost b)
    {
        return EdgeCost(
            0,
            0,
            a.consumer_stream_phases + b.consumer_stream_phases,
            a.consumer_dram_in_queues + b.consumer_dram_in_queues);
    }

    static EdgeCost sum_fan_out_streams(EdgeCost a, EdgeCost b)
    {
        return EdgeCost(a.fan_out_streams + b.fan_out_streams, 0, 0, 0);
    }

    static EdgeCost sum_producer_stream_phases(EdgeCost a, EdgeCost b)
    {
        auto ret = EdgeCost(0, a.producer_stream_phases + b.producer_stream_phases, 0, 0);
        ret.kMaxStreamPhasesProducer = std::min(a.kMaxStreamPhasesProducer, b.kMaxStreamPhasesProducer);

        return ret;
    }

    static EdgeCost sum_consumer_stream_phases(EdgeCost a, EdgeCost b)
    {
        auto ret = EdgeCost(0, 0, a.consumer_stream_phases + b.consumer_stream_phases, 0);
        ret.kMaxStreamPhasesConsumer = std::min(a.kMaxStreamPhasesConsumer, b.kMaxStreamPhasesConsumer);

        return ret;
    }

    static EdgeCost sum_consumer_dram_in_queues(EdgeCost a, EdgeCost b)
    {
        return EdgeCost(0, 0, 0, a.consumer_dram_in_queues + b.consumer_dram_in_queues);
    }

    static bool sort_fan_out_streams(EdgeCost a, EdgeCost b)
    {
        if (a.fan_out_streams == b.fan_out_streams)  // Defer to producer_stream_phases if eq
        {
            return a.producer_stream_phases < b.producer_stream_phases;
        }
        return a.fan_out_streams < b.fan_out_streams;
    }

    static bool sort_producer_stream_phases(EdgeCost a, EdgeCost b)
    {
        if (a.producer_stream_phases == b.producer_stream_phases)  // Defer to fan_out_streams if eq
        {
            return a.fan_out_streams < b.fan_out_streams;
        }
        return a.producer_stream_phases < b.producer_stream_phases;
    }

    static bool sort_consumer_stream_phases(EdgeCost a, EdgeCost b)
    {
        if (a.consumer_stream_phases == b.consumer_stream_phases)  // Defer to consumer_dram_in_queuesif eq
        {
            return a.consumer_dram_in_queues < b.consumer_dram_in_queues;
        }
        return a.consumer_stream_phases < b.consumer_stream_phases;
    }

    static bool sort_consumer_dram_in_queues(EdgeCost a, EdgeCost b)
    {
        if (a.consumer_dram_in_queues == b.consumer_dram_in_queues)  // Defer to consumer_stream_phases if eq
        {
            return a.consumer_stream_phases < b.consumer_stream_phases;
        }
        return a.consumer_dram_in_queues < b.consumer_dram_in_queues;
    }

    static std::array<std::pair<decltype(&sort_fan_out_streams), decltype(&sum_fan_out_streams)>, 2> producer_cost_fns()
    {
        return {
            std::make_pair(sort_fan_out_streams, sum_fan_out_streams),
            std::make_pair(sort_producer_stream_phases, sum_producer_stream_phases)};
    }

    static std::array<std::pair<decltype(&sort_fan_out_streams), decltype(&sum_fan_out_streams)>, 2> consumer_cost_fns()
    {
        return {
            std::make_pair(sort_consumer_stream_phases, sum_consumer_stream_phases),
            std::make_pair(sort_consumer_dram_in_queues, sum_consumer_dram_in_queues)};
    }

    static std::array<std::pair<decltype(&sort_fan_out_streams), decltype(&sum_fan_out_streams)>, 4> cost_fns()
    {
        return {
            std::make_pair(sort_fan_out_streams, sum_fan_out_streams),
            std::make_pair(sort_producer_stream_phases, sum_producer_stream_phases),
            std::make_pair(sort_consumer_stream_phases, sum_fan_out_streams),
            std::make_pair(sort_consumer_dram_in_queues, sum_consumer_dram_in_queues)};
    }
};

//
// Constraint interface
//
struct Constraint
{
    // Reference to device config, whose lifetime is managed outside of this class. However, this class isn't expected
    // to outlive device config.
    const DeviceConfig& device_config;
    std::shared_ptr<BalancerCacheCollection> balancer_cache_collection;
    bool resource_usage_fallback_mode = false;

    Constraint(const DeviceConfig& device_config, std::shared_ptr<BalancerCacheCollection> balancer_cache_collection) :
        device_config(device_config), balancer_cache_collection(balancer_cache_collection)
    {
        resource_usage_fallback_mode = env_as<bool>("PYBUDA_RESOURCE_USAGE_FALLBACK_MODE");
    }

    virtual std::pair<EdgeCost, ConstraintFailureReason> queue_to_op_cost(
        graphlib::Graph const* graph,
        graphlib::Edge edge,
        std::optional<OpModel> queue_producer_op_model,
        OpModel const& consumer) = 0;

    virtual std::pair<EdgeCost, ConstraintFailureReason> op_to_op_cost(
        graphlib::Graph const* graph, graphlib::Edge edge, OpModel const& producer, OpModel const& consumer) = 0;

    virtual std::pair<EdgeCost, ConstraintFailureReason> op_to_queue_cost(
        graphlib::Graph const* graph,
        graphlib::Edge edge,
        OpModel const& producer,
        std::optional<OpModel> queue_consumer_op_model) = 0;

    virtual ~Constraint() {}
};

struct GrayskullConstraint : public Constraint
{
    GrayskullConstraint(
        const DeviceConfig& device_config, std::shared_ptr<BalancerCacheCollection> balancer_cache_collection) :
        Constraint(device_config, balancer_cache_collection)
    {
    }

    virtual std::pair<EdgeCost, ConstraintFailureReason> queue_to_op_cost(
        graphlib::Graph const* graph,
        graphlib::Edge edge,
        std::optional<OpModel> queue_producer_op_model,
        OpModel const& consumer) override;

    virtual std::pair<EdgeCost, ConstraintFailureReason> op_to_op_cost(
        graphlib::Graph const* graph, graphlib::Edge edge, OpModel const& producer, OpModel const& consumer) override;

    virtual std::pair<EdgeCost, ConstraintFailureReason> op_to_queue_cost(
        graphlib::Graph const* graph,
        graphlib::Edge edge,
        OpModel const& producer,
        std::optional<OpModel> queue_consumer_op_model) override;
};

using WormholeConstraint = GrayskullConstraint;

inline std::ostream& operator<<(std::ostream& os, EdgeCost const& cost)
{
    os << "EdgeCost{.fan_out_streams = " << cost.fan_out_streams
       << ", .producer_stream_phases = " << cost.producer_stream_phases
       << ", .consumer_stream_phases = " << cost.consumer_stream_phases
       << ", .consumer_dram_in_queues = " << cost.consumer_dram_in_queues << "}";
    return os;
}

}  // namespace tt::balancer::legalizer
