// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "t_stream.hpp"

#include "autograd/binding.hpp"
#include "graph_lib/node_types.hpp"

namespace tt
{
using namespace balancer;

static char opposite_dir(char dir) { return (dir == 'h') ? 'v' : 'h'; }
static std::vector<OpType>::iterator append_tm(std::vector<OpType>& tms, graphlib::OpType const& tm)
{
    return tms.insert(tms.end(), tm);
}
static std::vector<OpType>::iterator prepend_tm(std::vector<OpType>& tms, graphlib::OpType const& tm, bool after_transpose = true)
{
    auto iter = tms.begin();
    if (after_transpose)
    {
        // We must prepend after transpose tms since the calculations below use streaming dir post-transpose
        iter = std::find_if(tms.begin(), tms.end(), [](OpType const& op) { return op.op == "transpose"; });
        iter = (iter == tms.end()) ? tms.begin() : iter + 1;
    }
    return tms.insert(iter, tm);
}

static void insert_t_stream_stack_slice(
    std::vector<OpType>& tms,
    std::string dir,
    int consumer_factor,
    int producer_factor,
    bool after_transpose = true,
    bool producer_z_major = false,
    int group = 1)
{
    if (consumer_factor == producer_factor)
        return;

    TT_ASSERT(dir.size() == 1 and (dir[0] == 'h' or dir[0] == 'v'));

    int factor;
    std::string op_name;
    if (consumer_factor > producer_factor)
    {
        TT_ASSERT((consumer_factor % producer_factor) == 0);
        factor = consumer_factor / producer_factor;
        op_name = "slice";
    }
    else
    {
        TT_ASSERT((producer_factor % consumer_factor) == 0);
        factor = producer_factor / consumer_factor;
        op_name = "stack";
    }

    std::vector<graphlib::OpType>::iterator iter;
    graphlib::OpType op_type((dir + op_name), {factor});
    if (consumer_factor > producer_factor or producer_z_major)
    {
        if (tms.size() > 0 && (*tms.rbegin()).op == "buda_pad" && op_name == "stack")
        {
            OpType buda_pad = *tms.rbegin();
            tms.pop_back();
            tms.push_back(op_type);
            iter = tms.insert(tms.end(), buda_pad);
        }
        else
            iter = append_tm(tms, op_type);
    }
    else
    {
        iter = prepend_tm(tms, op_type, after_transpose);
    }

    if (group > 1)
    {
        auto group_dir = opposite_dir(dir[0]);
        graphlib::OpType group_op_type((group_dir + std::string("stack")), {group});
        iter = tms.insert(iter, group_op_type);
        graphlib::OpType ungroup_op_type((group_dir + std::string("slice")), {group});
        tms.insert(iter + 2, ungroup_op_type);
    }
}

static void insert_t_stream_default_tms(
    std::vector<OpType>& tms,
    TStreamFactor consumer_factor,
    TStreamFactor producer_factor,
    int group = 1,
    bool after_transpose = true)
{
    TT_ASSERT(not consumer_factor.none() or not producer_factor.none(), producer_factor, consumer_factor);

    TStreamDir dir = consumer_factor.none() ? producer_factor.dir : consumer_factor.dir;
    if (dir.r())
    {
        insert_t_stream_stack_slice(
            tms, "v", consumer_factor.r, producer_factor.r, after_transpose, producer_factor.dir.z_major(), group);
        insert_t_stream_stack_slice(tms, "h", consumer_factor.c, producer_factor.c, after_transpose);
    }
    else
    {
        insert_t_stream_stack_slice(
            tms, "h", consumer_factor.c, producer_factor.c, after_transpose, producer_factor.dir.z_major(), group);
        insert_t_stream_stack_slice(tms, "v", consumer_factor.r, producer_factor.r, after_transpose);
    }
}

void insert_t_stream_tms_for_eltwise(
    std::vector<graphlib::OpType>& tms,
    TStreamFactor consumer_factor,
    TStreamFactor producer_factor,
    bool after_transpose)
{
    return insert_t_stream_default_tms(tms, consumer_factor, producer_factor, 1 /*group*/, after_transpose);
}

static void insert_t_stream_tms_for_matmul(
    std::vector<OpType>& tms, TStreamFactor consumer_factor, TStreamFactor producer_factor, int operand_idx)
{
    if (consumer_factor.none())
        return insert_t_stream_default_tms(tms, consumer_factor, producer_factor);

    if (consumer_factor.dir.r())
    {
        if (operand_idx == 0)
        {
            insert_t_stream_default_tms(tms, consumer_factor, producer_factor);
        }
        else if (operand_idx == 1)
        {
            // If matmul R streaming RHS must be fully buffered so unstream the RHS
            // This has been proved to be safe to do via constraints
            if (producer_factor.is_streaming())
                insert_t_stream_default_tms(tms, TStreamFactor{}, producer_factor);
            graphlib::OpType broadcast("broadcast", {3, consumer_factor.r});
            append_tm(tms, broadcast);
            insert_t_stream_stack_slice(tms, "h", consumer_factor.r, 1);
        }
        else
        {
            // Fused bias
            insert_t_stream_default_tms(tms, consumer_factor, producer_factor);
        }
    }
    else
    {
        if (operand_idx == 0)
        {
            // If matmul C streaming LHS must be fully buffered so unstream the LHS
            // This has been proved to be safe to do via constraints
            if (producer_factor.is_streaming())
                insert_t_stream_default_tms(tms, TStreamFactor{}, producer_factor);
            graphlib::OpType broadcast("broadcast", {2, consumer_factor.c});
            append_tm(tms, broadcast);
            insert_t_stream_stack_slice(tms, "v", consumer_factor.c, 1);
        }
        else if (operand_idx == 1)
        {
            insert_t_stream_default_tms(tms, consumer_factor, producer_factor);
        }
        else
        {
            // Fused bias
            insert_t_stream_default_tms(tms, consumer_factor, producer_factor);
        }
    }
}

static void insert_t_stream_tms_for_sparse_matmul(
    std::vector<OpType>& tms, TStreamFactor consumer_factor, TStreamFactor producer_factor, int operand_idx)
{
    if (operand_idx == 1)
    {
        TT_ASSERT(not consumer_factor.none() or not producer_factor.none());
        if (producer_factor.dir.r())
        {
            insert_t_stream_stack_slice(tms, "v", 1, producer_factor.r);
        }

        if (producer_factor.dir.c() or consumer_factor.dir.c())
        {
            insert_t_stream_stack_slice(tms, "h", consumer_factor.c, producer_factor.c);
        }
    }
}

static void insert_t_stream_tms_for_op(
    graphlib::OpNode const* op_node,
    std::vector<OpType>& tms,
    TStreamFactor consumer_factor,
    TStreamFactor producer_factor,
    int operand_idx,
    int group = 1)
{
    TT_ASSERT(op_node);
    if (consumer_factor.none() and producer_factor.none())
        return;

    if (op_node->is_sparse_matmul())
    {
        TT_ASSERT(group == 1, "unsupported");
        return insert_t_stream_tms_for_sparse_matmul(tms, consumer_factor, producer_factor, operand_idx);
    }
    else if (op_node->is_matmul())
    {
        TT_ASSERT(group == 1, "unsupported");
        return insert_t_stream_tms_for_matmul(tms, consumer_factor, producer_factor, operand_idx);
    }
    else
    {
        return insert_t_stream_default_tms(tms, consumer_factor, producer_factor, group);
    }
}

static void assert_t_stream_factors(
    graphlib::OpNode const* consumer,
    TStreamFactor producer_factor,
    TStreamFactor consumer_factor,
    int group,
    bool consumes_rz_major)
{
    bool r_slicing = (consumer_factor.r > producer_factor.r);
    bool c_slicing = (consumer_factor.c > producer_factor.c);
    bool r_stacking = (consumer_factor.r < producer_factor.r);
    bool c_stacking = (consumer_factor.c < producer_factor.c);
    bool slicing = r_slicing or c_slicing;
    bool stacking = r_stacking or c_stacking;
    bool eq = (consumer_factor.r == producer_factor.r and consumer_factor.c == producer_factor.c);
    TT_ASSERT(
        producer_factor.compatible_consumer(
            consumer_factor, consumer->is_sparse_matmul(), (group > 1) and consumes_rz_major),
        consumer->name(),
        producer_factor,
        consumer_factor,
        consumer->is_sparse_matmul(),
        group,
        consumes_rz_major);
    TT_LOG_ASSERT(
        (slicing != stacking) or eq,
        "Illegal combination of slicing/stacking: {}\n{}\n{}",
        consumer->name(),
        producer_factor,
        consumer_factor);
}

static void lower_broadcast_z(graphlib::Shape shape, std::vector<OpType>& tms)
{
    auto match = std::find_if(
        tms.begin(),
        tms.end(),
        [&shape](auto const& tm) { return shape.z() > 1 and tm.op == "broadcast" and std::get<int>(tm.attr[0]) == 1; });

    if (match == tms.end())
        return;

    // net2pipe doesn't support Z bcast when t>1 so if we're streaming, turn Z bcast into C broadcast and hslice
    OpType broadcast = *match;
    std::get<int>(broadcast.attr[0]) = 3;
    int factor = std::get<int>(broadcast.attr[1]);
    graphlib::OpType hslice("hslice", {factor});

    auto insert_pos = tms.erase(match);
    insert_pos = tms.insert(insert_pos, hslice);
    insert_pos = tms.insert(insert_pos, broadcast);
}

static void consteval_t_stream_shape_for_loopback(graphlib::Graph* graph, graphlib::InputNode* loopback_queue, TStreamFactor producer_factor)
{
    if (producer_factor.none())
        return;

    std::vector<OpType> tms;
    insert_t_stream_default_tms(tms, producer_factor, TStreamFactor());

    graphlib::ConstEvalGraph* consteval_graph = loopback_queue->get_consteval_graph(graph, true, true);

    consteval_graph->pad_output_to_buda_dims("t_stream");

    graphlib::Shape current_shape = consteval_graph->get_output()->shape();
    for (graphlib::OpType const& op_type : tms)
    {
        std::vector<graphlib::Shape> input_shapes = {current_shape};
        auto [shape, bcast_dims] = ::get_op_shape(op_type, input_shapes, false);
        auto tm = graphlib::create_node<graphlib::PyOpNode>(op_type.op + "_" + loopback_queue->name(), op_type);
        tm->set_shape(shape);
        tm->set_epoch_type(loopback_queue->get_epoch_type());
        tm->set_output_df(loopback_queue->output_df());
        consteval_graph->promote_node(std::move(tm));
        current_shape = shape;
    }

    // This is for loopback so set needs autograd to true
    consteval_graph->set_needs_autograd(true);
    consteval_graph->autograd();
}

// See insert_t_stream_tms declaration documentation for what `group` means in the context
// of inserting t-stream TMs. This is somewhat of a special case when t-streaming RZ dir
// through a queue.  In order to get back to the canonical form, we need to undo the z-major
// tile ordering.  We do this by grouping together chunks of some inner "group" factor,
// performing the regular t-streaming TMs, and then undoing the grouping.  This is a form of
// Z/R permute of tiles.
static int calculate_group_factor(
    bool is_queue,
    TStreamFactor consumer_factor,
    TStreamFactor producer_factor,
    int operand_idx,
    std::vector<OpType> const& tms,
    bool is_mm,
    bool is_sparse_mm,
    bool consumes_rz_major)
{
    TStreamFactor none;
    bool directly_compatible = producer_factor.compatible_consumer(consumer_factor, is_sparse_mm, consumes_rz_major);
    bool reorder_without_grouping = producer_factor.compatible_consumer(none, false, false) and
                                    none.compatible_consumer(consumer_factor, is_sparse_mm, consumes_rz_major);
    bool is_non_primary_matmul_streaming =
        is_mm and ((consumer_factor.dir.r() and operand_idx != 0) or (consumer_factor.dir.c() and operand_idx != 1));
    if (not is_queue or directly_compatible or reorder_without_grouping or is_non_primary_matmul_streaming)
        return 1;

    int internal_slice_stack_factor = 1;
    for (auto const& tm : tms)
    {
        if (tm.op == "vslice")
        {
            internal_slice_stack_factor *= std::get<int>(tm.attr[0]);
        }
        else if (tm.op == "hstack")
        {
            TT_ASSERT(internal_slice_stack_factor % std::get<int>(tm.attr[0]) == 0);
            internal_slice_stack_factor /= std::get<int>(tm.attr[0]);
        }
        else if (tm.op == "hslice")
        {
            internal_slice_stack_factor *= std::get<int>(tm.attr[0]);
        }
        else if (tm.op == "vstack")
        {
            TT_ASSERT(internal_slice_stack_factor % std::get<int>(tm.attr[0]) == 0);
            internal_slice_stack_factor /= std::get<int>(tm.attr[0]);
        }
        else if (tm.op == "transpose")
        {
            // nothing to do
        }
        else
        {
            TT_LOG_ASSERT(false, "Unhandled tm type for grouping {}", tm.op);
        }
    }
    return internal_slice_stack_factor;
}

void insert_t_stream_tms(
    graphlib::OpNode const* consumer,
    std::vector<OpType>& tms,
    TStreamFactor consumer_factor,
    TStreamFactor producer_factor,
    int operand_idx,
    bool through_queue,
    int group,
    bool consumes_rz_major)
{
    bool has_transpose =
        std::find_if(tms.begin(), tms.end(), [](OpType const& op) { return op.op == "transpose"; }) != tms.end();
    producer_factor = has_transpose ? TStreamFactor::Transposed(producer_factor) : producer_factor;

    int producer_group = producer_factor.is_streaming() ? group : 1;
    int consumer_group = consumer_factor.is_streaming() ? group : 1;
    if (through_queue and producer_factor.is_streaming())
    {
        // If we come from a queue, first undo all producer streaming TMs into canonical form
        insert_t_stream_default_tms(tms, TStreamFactor{}, producer_factor, producer_group);
        // Then apply (below) consumer t-stream TMs
        producer_factor = TStreamFactor{};
    }

    assert_t_stream_factors(consumer, producer_factor, consumer_factor, 1, consumes_rz_major);
    insert_t_stream_tms_for_op(consumer, tms, consumer_factor, producer_factor, operand_idx, consumer_group);
}

void insert_t_stream_tms(Graph* graph, balancer::OpModelMap const& op_models)
{
    std::unordered_set<graphlib::InputNode*> visted_loopback_queues;
    for (auto const& [node_id, edges] : graph->operands_map())
    {
        graphlib::OpNode* consumer = dynamic_cast<graphlib::OpNode*>(graph->node_by_id(node_id));
        if (not consumer)
            continue;

        OpModel const& consumer_op_model = op_models.at(consumer->name());
        for (Edge const& edge : edges)
        {
            if (edge.edge_type != graphlib::EdgeType::kData and edge.edge_type != graphlib::EdgeType::kDataLoopback)
            {
                continue;
            }
            TT_ASSERT(node_id == edge.consumer_node_id);
            auto edge_attrs = graph->get_edge_attributes(edge);
            Node* producer = graph->node_by_id(edge.producer_node_id);
            graphlib::InputNode* loopback_queue = nullptr;

            std::vector<graphlib::Node*> producer_operands = graph->data_operands(producer);
            if (producer_operands.empty())
            {
                TT_ASSERT(producer->node_type() == graphlib::NodeType::kInput);
                insert_t_stream_tms_for_op(
                    consumer,
                    edge_attrs->get_tms(),
                    consumer_op_model.t_stream_factor,
                    TStreamFactor(),
                    edge.consumer_input_port_id);
                continue;
            }

            // If this edge is cut, treat it as virtual edge which will be bypassed with a queue.
            //
            bool is_queue = producer->node_type() != graphlib::NodeType::kBudaOp;
            if (is_queue)
            {
                TT_ASSERT(producer->node_type() == graphlib::NodeType::kInput or producer->node_type() == graphlib::NodeType::kQueue);
                TT_ASSERT(producer_operands.size() == 1);
                loopback_queue = dynamic_cast<graphlib::InputNode*>(producer);
                producer = producer_operands[0];
                TT_ASSERT(producer->node_type() == graphlib::NodeType::kBudaOp);
            }

            OpModel const& producer_op_model = op_models.at(producer->name());
            if (producer_op_model.t_stream_factor.none() and consumer_op_model.t_stream_factor.none())
                continue;

            log_trace(
                LogTStream,
                "Insert t stream tms: {}[{}]({}) -> {}[{}]({}) {}",
                producer->name(),
                edge.producer_output_port_id,
                producer->get_epoch_type(),
                consumer->name(),
                edge.consumer_input_port_id,
                consumer->get_epoch_type(),
                edge_attrs->get_ublock_order());
            log_trace(
                LogTStream,
                "    {} {} {}",
                producer_op_model.grid_shape,
                producer_op_model.block_shape(),
                producer_op_model.t_stream_factor);
            log_trace(
                LogTStream,
                "    {} {} {}",
                consumer_op_model.grid_shape,
                consumer_op_model.block_shape(),
                consumer_op_model.t_stream_factor);

            auto& tms = edge_attrs->get_tms();
            int group = calculate_group_factor(
                is_queue,
                consumer_op_model.t_stream_factor,
                producer_op_model.t_stream_factor,
                edge.consumer_input_port_id,
                tms,
                consumer->is_matmul(),
                consumer->is_sparse_matmul(),
                consumer_op_model.consumes_rz_major);
            insert_t_stream_tms(
                consumer,
                tms,
                consumer_op_model.t_stream_factor,
                producer_op_model.t_stream_factor,
                edge.consumer_input_port_id,
                is_queue,
                group,
                consumer_op_model.consumes_rz_major);

            lower_broadcast_z(producer->shape(), edge_attrs->get_tms());

            if (loopback_queue and visted_loopback_queues.find(loopback_queue) == visted_loopback_queues.end())
            {
                consteval_t_stream_shape_for_loopback(graph, loopback_queue, producer_op_model.t_stream_factor);
                visted_loopback_queues.insert(loopback_queue);
            }
        }
    }

    // Calculate undo t streaming for golden reconstruction
    for (Node* node : graph->nodes())
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
            continue;

        TStreamFactor t_stream_factor = op_models.at(node->name()).t_stream_factor;
        if (t_stream_factor.none())
            continue;

        // Set `after_transpose=false` for golden because it requires the TMs
        // to be undone in exactly the opposite order in which they were applied
        constexpr bool after_transpose = false;
        constexpr int group = 1;
        insert_t_stream_default_tms(
            node->as<graphlib::OpNode>()->get_golden_transforms(),
            TStreamFactor{},
            t_stream_factor,
            group,
            after_transpose);
    }
}
}  // namespace tt
