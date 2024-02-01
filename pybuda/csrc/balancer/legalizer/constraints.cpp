// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/legalizer/constraints.hpp"

#include "balancer/balancer_utils.hpp"
#include "balancer/exceptions.hpp"
#include "balancer/legalizer/graph_solver.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "passes/t_stream.hpp"
#include "utils/logger.hpp"

namespace tt::balancer::legalizer
{
static bool block_compatible(
    OpModel const& producer,
    OpModel const& consumer,
    graphlib::Edge edge,
    graphlib::UBlockOrder ublock_order,
    bool has_transpose,
    bool consumer_needs_matmul_streaming)
{
    TStreamFactor producer_t_stream_factor =
        has_transpose ? TStreamFactor::Transposed(producer.t_stream_factor) : producer.t_stream_factor;

    if (producer_t_stream_factor.none())
        return true;

    tt::graphlib::NodeId node_id = edge.consumer_node_id;
    TensorShape producer_shape = producer.effective_input_buffer_shape_for_user.at(node_id);
    BlockShape consumer_block_shape = consumer.input_buffers[edge.consumer_input_port_id].block_shape;
    int consumer_shape_rt = consumer_block_shape.rt() * consumer.grid_shape.r;
    int consumer_shape_ct = consumer_block_shape.ct() * consumer.grid_shape.c;
    UBlockShape consumer_ublock = consumer.input_buffers[edge.consumer_input_port_id].block_shape.ublock;

    bool producer_anti_ublock_order = not producer_t_stream_factor.dir.is_ublock_order(ublock_order);
    bool ublock_order_preserved = false;
    if (producer_anti_ublock_order or consumer_needs_matmul_streaming)
    {
        ublock_order_preserved =
            (ublock_order == graphlib::UBlockOrder::R)
                ? ((producer_shape.rt == consumer_ublock.rt) or (producer_shape.ct == consumer_shape_ct))
                : ((producer_shape.ct == consumer_ublock.ct) or (producer_shape.rt == consumer_shape_rt));
    }
    else  // ublock order == eltwise style streaming
    {
        // Either you've sliced all the way down to a single ublock, or you're not streaming in anti-ublock dimension
        ublock_order_preserved = (ublock_order == graphlib::UBlockOrder::R)
                                     ? ((producer_shape.rt == consumer_ublock.rt) or (producer_t_stream_factor.c == 1))
                                     : ((producer_shape.ct == consumer_ublock.ct) or (producer_t_stream_factor.r == 1));
    }

    return ((producer_shape.rt % consumer_ublock.rt) == 0) and ((producer_shape.ct % consumer_ublock.ct) == 0) and
           ublock_order_preserved;
}

static bool double_counted(graphlib::Graph const* graph, graphlib::Edge edge)
{
    auto operand_edges = graph->operand_data_edges(graph->node_by_id(edge.consumer_node_id));
    auto smallest_port_id = edge.consumer_input_port_id;
    for (graphlib::Edge operand : operand_edges)
    {
        if (operand.producer_node_id == edge.producer_node_id)
        {
            smallest_port_id = std::min(smallest_port_id, operand.consumer_input_port_id);
        }
    }
    return smallest_port_id != edge.consumer_input_port_id;
}

static bool legal_t_stream_dirs(
    TStreamFactor producer_t_stream_factor,
    TStreamFactor consumer_t_stream_factor,
    bool producer_has_z,
    bool consumes_rz_major)
{
    auto producer_dir = producer_t_stream_factor.dir;
    auto consumer_dir = consumer_t_stream_factor.dir;

    // TMs with pattern vslice(N) -> hstack(N) force RZ -> R streaming
    // If stream direction is identical this is always legal
    // If the producer has a z then the consumer must handle same streaming dir
    // otherwise only their primary streaming directions need to match
    if (consumes_rz_major)
        return producer_dir == TStreamDir::RZ and consumer_dir == TStreamDir::R;
    else if (producer_dir == consumer_dir)
        return true;
    else if (producer_has_z and producer_dir.z_major())
        return producer_dir == consumer_dir;
    else if (not producer_t_stream_factor.none() and not consumer_t_stream_factor.none())
        return producer_dir.primary_dir_compatible(consumer_dir);
    return true;
}

static bool legal_matmul_streaming(
    TStreamFactor producer_t_stream_factor,
    TStreamFactor consumer_t_stream_factor,
    graphlib::Edge edge,
    bool consumer_is_matmul)
{
    if (not consumer_is_matmul)
        return true;

    bool lhs_matmul = edge.consumer_input_port_id == 0;
    bool rhs_matmul = edge.consumer_input_port_id == 1;
    bool both_streaming = producer_t_stream_factor.is_streaming() and consumer_t_stream_factor.is_streaming();

    if (both_streaming and rhs_matmul and consumer_t_stream_factor.dir.r())
    {
        return false;
    }

    if (both_streaming and lhs_matmul and consumer_t_stream_factor.dir.c())
    {
        return false;
    }

    return true;
}

static ConstraintFailureReason legal_t_streaming(
    OpModel const& producer,
    OpModel const& consumer,
    graphlib::Edge edge,
    graphlib::UBlockOrder ublock_order,
    bool has_transpose,
    bool consumer_needs_matmul_streaming)
{
    TStreamFactor producer_t_stream_factor =
        has_transpose ? TStreamFactor::Transposed(producer.t_stream_factor) : producer.t_stream_factor;
    TStreamFactor consumer_t_stream_factor = consumer.t_stream_factor;

    if (producer_t_stream_factor.none() and consumer_t_stream_factor.none())
    {
        return NoConstraintFailure;  // early out if we're not streaming
    }

    if (not producer_t_stream_factor.compatible_consumer(
            consumer_t_stream_factor, consumer.is_sparse_matmul, consumer.consumes_rz_major))
    {
        return TStreamIncompatibleConsumer;
    }

    bool producer_has_z = consumer.op_shape.inputs[edge.consumer_input_port_id].z > consumer_t_stream_factor.t();
    if (not legal_t_stream_dirs(
            producer_t_stream_factor, consumer_t_stream_factor, producer_has_z, consumer.consumes_rz_major))
    {
        return TStreamIllegalTStreamDir;
    }

    bool producer_streaming = producer_t_stream_factor.is_streaming();
    bool consumer_streaming = consumer_t_stream_factor.is_streaming();

    // We must ensure block shape is compatible through the entire stream
    if (not block_compatible(producer, consumer, edge, ublock_order, has_transpose, consumer_needs_matmul_streaming))
    {
        return TStreamBlockIncompatible;
    }

    // This might be able to be relaxed in the future, but for now, disallow anti-ublock-order stacking
    auto orig_producer_t_stream_factor = producer.t_stream_factor;  // non-transposed
    bool producer_anti_ublock_order = not orig_producer_t_stream_factor.dir.is_ublock_order(ublock_order);
    if (producer_streaming and not consumer_streaming and producer_anti_ublock_order)
    {
        return TStreamAntiUblockOrderStacking;
    }

    // If matmul is streaming then producer can only stream from respective side LHS=TStreamDir::R, RHS=TStreamDir::C
    if (not legal_matmul_streaming(
            producer_t_stream_factor, consumer_t_stream_factor, edge, consumer_needs_matmul_streaming))
    {
        return TStreamMatmulStreamingSide;
    }

    // Stack / grid divsilibility constraints
    bool lhs_matmul = consumer.op_type() == "matmul" and edge.consumer_input_port_id == 0;
    bool rhs_matmul = consumer.op_type() == "matmul" and edge.consumer_input_port_id == 1;
    int stack_factor_r = producer_t_stream_factor.r / consumer_t_stream_factor.r;
    int stack_factor_c = producer_t_stream_factor.c / consumer_t_stream_factor.c;
    if (not rhs_matmul and stack_factor_r and not divisible_either_direction(stack_factor_r, consumer.grid_shape.r))
    {
        return TStreamNotDivisableRow;
    }

    if (not lhs_matmul and stack_factor_c and not divisible_either_direction(stack_factor_c, consumer.grid_shape.c))
    {
        return TStreamNotDivisableColumn;
    }

    return NoConstraintFailure;
}

std::pair<EdgeCost, ConstraintFailureReason> GrayskullConstraint::queue_to_op_cost(
    graphlib::Graph const* graph,
    graphlib::Edge edge,
    std::optional<OpModel> queue_producer_op_model,
    OpModel const& consumer)
{
    // This is the input edge case
    graphlib::Node const* producer_node = graph->node_by_id(edge.producer_node_id);
    graphlib::InputNode const* input = dynamic_cast<graphlib::InputNode const*>(producer_node);

    if (input and (input->is_constant() or input->is_parameter() or input->is_optimizer_parameter()))
    {
        // We can reblock parameters and constants to have the same grid shape as consumer
        // Therefore 1 to 1 mapping
        return std::make_pair(EdgeCost(0, 0, 0, 1, &device_config, nullptr, &consumer), NoConstraintFailure);
    }
    else if (queue_producer_op_model)
    {
        ResourceUsage usage =
            resource_usage_fallback_mode
                ? get_edge_resource_usage_simple(graph, edge, *queue_producer_op_model, consumer, true)
                : get_edge_resource_usage(
                      graph,
                      balancer_cache_collection->pipe_to_resource_usage_cache,
                      edge,
                      *queue_producer_op_model,
                      consumer,
                      true);
        ConstraintFailureReason constraint_failure =
            (usage.consumer_fan_in > EdgeCost::kMaxDRAMInQueues) ? ExceedsDRAMInQueues : NoConstraintFailure;
        return std::make_pair(
            EdgeCost(0, 0, 0, usage.consumer_fan_in, &device_config, nullptr, &consumer), constraint_failure);
    }
    else if (consumer.op_type() == "matmul")
    {
        return std::make_pair(
            EdgeCost(
                0,
                0,
                0,
                edge.consumer_input_port_id == 0 ? consumer.grid_shape.r : consumer.grid_shape.c,
                &device_config,
                nullptr,
                &consumer),
            NoConstraintFailure);
    }
    else
    {
        // If there is no queue_producer_op_model this is likely coming from an activation on 1x1 grid
        return std::make_pair(EdgeCost(0, 0, 0, 1, &device_config, nullptr, &consumer), NoConstraintFailure);
    }
}

std::pair<EdgeCost, ConstraintFailureReason> GrayskullConstraint::op_to_op_cost(
    graphlib::Graph const* graph, graphlib::Edge edge, OpModel const& producer, OpModel const& consumer)
{
    graphlib::Node const* producer_node = graph->node_by_id(edge.producer_node_id);
    if (producer_node->node_type() != graphlib::NodeType::kInput &&
        (graph->user_data_edges(producer_node).size() > EdgeCost::kMaxDRAMOutQueues))
    {
        throw BalancerError(
            fmt::format("Node exceeds kMaxDRAMOutQueues {}", producer_node->name()),
            BalancerError::NodeExceedsMaxOpForks(EdgeCost::kMaxDRAMOutQueues));
    }

    graphlib::OpNode const* consumer_node =
        dynamic_cast<graphlib::OpNode const*>(graph->node_by_id(edge.consumer_node_id));
    TT_ASSERT(consumer_node);
    bool consumer_needs_matmul_streaming = consumer_node->is_matmul() and not consumer_node->is_sparse_matmul();
    bool is_double_counted = double_counted(graph, edge);
    auto edge_attr = graph->get_edge_attributes(edge);
    bool has_transpose = edge_attr->has_tm("transpose");

    ConstraintFailureReason tStreamFailureReason = legal_t_streaming(
        producer, consumer, edge, edge_attr->get_ublock_order(), has_transpose, consumer_needs_matmul_streaming);

    if (NoConstraintFailure != tStreamFailureReason)
    {
        return std::make_pair(EdgeCost(0, 0, 0, 0, &device_config), tStreamFailureReason);
    }

    // Hack, for convs there are slice/stack tms that immediately follow a sparse matmul which
    // causes hangs on silicon if we then go into matmul streaming.
    graphlib::OpNode const* producer_op_node = dynamic_cast<graphlib::OpNode const*>(producer_node);
    if (producer.t_stream_factor.none() and not consumer.t_stream_factor.none() and producer_op_node != nullptr and
        producer_op_node->is_sparse_matmul() and consumer_needs_matmul_streaming)
    {
        return std::make_pair(EdgeCost(0, 0, 0, 0, &device_config), ConvSliceStackMatmulStreamingHang);
    }

    if (is_double_counted)
    {
        return std::make_pair(
            EdgeCost(0, 0, 0, 0, &device_config, &producer, &consumer),
            NoConstraintFailure);  // This edge has a sibling that connects the same pair of nodes
    }

    // Calculate edge cost
    ResourceUsage usage =
        resource_usage_fallback_mode
            ? get_edge_resource_usage_simple(graph, edge, producer, consumer)
            : get_edge_resource_usage(
                  graph, balancer_cache_collection->pipe_to_resource_usage_cache, edge, producer, consumer);

    return std::make_pair(
        EdgeCost(
            usage.producer_fan_out,
            usage.producer_phases,
            usage.consumer_phases,
            0,
            &device_config,
            &producer,
            &consumer),
        NoConstraintFailure);
}

std::pair<EdgeCost, ConstraintFailureReason> GrayskullConstraint::op_to_queue_cost(
    graphlib::Graph const* graph,
    graphlib::Edge edge,
    OpModel const& producer,
    std::optional<OpModel> queue_consumer_op_model)
{
    if (graph->node_by_id(edge.consumer_node_id)->node_type() == graphlib::NodeType::kQueue)
    {
        // As queue is inheriting OpModels from producer we need to make them tightly 1-1 coupled,
        // as this will cause invalid OpModel elimination propagation work properly from Queue to Producer and vice
        // versa.
        //
        OpModel* consumer_op_model = queue_consumer_op_model.has_value() ? &queue_consumer_op_model.value() : nullptr;

        if (consumer_op_model && consumer_op_model->id == producer.id)
        {
            // If there is an OpModel for consumer queue, then the ids must match
            return std::make_pair(
                EdgeCost(1, 0, 0, 0, &device_config, &producer, consumer_op_model), NoConstraintFailure);
        }
        else
        {
            return std::make_pair(EdgeCost(0, 0, 0, 0, &device_config, &producer, nullptr), OpToQueueMapping);
        }
    }
    else
    {
        return std::make_pair(
            EdgeCost(1, 0, 0, 0, &device_config, &producer, nullptr),
            NoConstraintFailure);  // This is the output edge case
    }
}

#ifdef DEBUG
std::string EdgeConstraintDebugInfo::toString(const Graph* graph) const
{
    std::string result = "Edge constraint failure counts by type: \n";

    for (int i = NoConstraintFailure; i < MaxConstraintFailureReason; i++)
    {
        result += ConstraintFailureReasonDesc[i] + ": " + std::to_string(constraintFailureCountByType[i]) + "\n";
    }

    if (nullptr != graph && eliminatingEdges.size() > 0)
    {
        result += "Edge elimination data for valid paths:\n";
        for (const std::pair<const graphlib::EdgeUniqueId, int>& pair : eliminatingEdges)
        {
            graphlib::NodeId producer_node_id = std::get<0>(pair.first);
            graphlib::NodeId consumer_node_id = std::get<2>(pair.first);

            if (0 == producer_node_id)
            {
                result += "Subgraph consistency update eliminated: " + std::to_string(pair.second) + " valid paths.\n";
            }
            else
            {
                std::string producerNodeName = graph->node_by_id(producer_node_id)->name();
                std::string consumerNodeName = graph->node_by_id(consumer_node_id)->name();
                result += "Edge " + producerNodeName + " -> " + consumerNodeName +
                          " eliminated: " + std::to_string(pair.second) + " valid paths.\n";
            }
        }
    }

    return result;
}

void EdgeConstraintDebugInfo::addEliminatingEdge(const graphlib::EdgeUniqueId edge)
{
    if (eliminatingEdges.count(edge) == 0)
    {
        eliminatingEdges.insert(std::make_pair(edge, 1));
    }
    else
    {
        eliminatingEdges[edge]++;
    }
}
#endif

}  // namespace tt::balancer::legalizer
