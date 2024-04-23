// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "post_placer_buda_passes.hpp"

#include "lower_to_buda/common.hpp"
#include "utils/env.hpp"

namespace tt
{

graphlib::QueueNode* create_buffering_queue(Graph* graph, graphlib::Node* producer_node,std::string name, int num_entries);

void set_prologue_queues(Graph *graph, balancer::OpModelMap const &op_model_map)
{
    for (Node *node : graph->nodes())
    {
        if (node->node_type() == graphlib::NodeType::kInput)
        {
            balancer::OpModel const &op_model = op_model_map.at(node->name());
            node->as<graphlib::InputNode>()->set_prologue(op_model.input_prologue);
            log_trace(LogGraphCompiler, "Prologue {}: {}", node->name(), op_model.input_prologue);
        }
    }
}

static void set_kernel_broadcast(graphlib::BudaOpNode *op, balancer::OpModel const &op_model)
{
    graphlib::OpType &type = op->op_type();
    int input_idx = 0;
    for (auto const &input_buffer : op_model.input_buffers)
    {
        std::string input_name = "input_" + std::to_string(input_idx++);
        if (input_buffer.kernel_broadcast_tiles == 0)
            continue;
        auto &attr = type.buda_attrs["kernel_broadcast"];
        if (not std::holds_alternative<BudaKernelBroadcastInputs>(attr))
            attr = BudaKernelBroadcastInputs{};
        std::get<BudaKernelBroadcastInputs>(attr)[input_name] = input_buffer.kernel_broadcast_tiles;
    }
}

static void set_l1_accumulate(graphlib::BudaOpNode *op, bool is_wormhole_b0)
{
    
    /*
    
        This function should provide accumulation in L1 memory.
        
        This are the constraints related to this feature:
        
            - (1) Intermediate format can be float32, float16_b, int32 (float16_a is excluded due to the hw bug)
                  Note: We will skip format int32 for now.

            - (2) If intermed and output format are the same then buffer sharing is enabled (interm and output buffer share same physical address range in l1). 
                  We'll get performance gain when m_k > 1.

            - (3) If interm and output format are different then buffers are split and we don't need to double buffer output (buf_size_mb can be set to 1). 
                  We'll get performance gain if m_k>2. When m_k is 2 we need to spill and reload once to repack data in different format and there is only single pass through interm buffers. 
                  For the case m_k=2 we'll introduce overhead and reduce perf if l1 acc is enabled.

            - (4) If m_k=1 l1 acc won't make any difference as we don't spill into l1.

            - (5) For matmul accumulations and accumulate data format Fp32, intermediate data format must be Fp32.

            - (6) If the operation is reduce skip L1 accumulation.

    */

    // Flag for disabling and debugging L1 accumaulation feature.
    if (env_as<bool>("PYBUDA_DISABLE_L1_ACCUMULATE"))
        return;

    // L1 accumulation is only supported on WH B0
    if (!is_wormhole_b0)
        return;

    // Check data formats (1)
    bool is_not_float32 = op->intermediate_df() != DataFormat::Float32;
    bool is_not_float16_b = op->intermediate_df() != DataFormat::Float16_b;

    bool is_acc_float32 = op->accumulate_df() == DataFormat::Float32;

    if (is_not_float32 && is_not_float16_b)
        return;

    // Check matmul and, intermediate and accumulate data format. (5)
    if (op->is_matmul() && is_acc_float32 && is_not_float32)
        return;

    graphlib::OpType &type = op->op_type();

    // Check reduce op. (6)
    if (type.op == "reduce")
        return;

    // If m_k exists as attribute retrieve it, otherwise don't apply L1 accumulation.
    if (type.buda_attrs.find("m_k") == type.buda_attrs.end())
        return;

    int m_k = std::get<int>(type.buda_attrs["m_k"]);

    // (4)
    if (m_k == 1)
        return;

    bool relu_present = (type.buda_attrs.find("relu_en") != type.buda_attrs.end());

    // Compare intermediate and output data formats
    // (2)
    if (op->intermediate_df() == op->output_df())
    {
        if (m_k > 1 && !relu_present)
            type.buda_attrs["l1_acc"] = true;
    }
    // (3)
    else
    {
        if (m_k > 2 || relu_present)
            type.buda_attrs["l1_acc"] = true;
    }
    
}

static std::tuple<int, int, int> calculate_sparse_mm_inner_dim(
    graphlib::Graph *graph, graphlib::BudaOpNode *op, balancer::OpModelMap const &op_model_map)
{
    balancer::OpModel const &op_model = op_model_map.at(op->name());
    auto operands = graph->operand_data_edges(op);
    TT_ASSERT(operands.size() >= 3);
    graphlib::Node *input1 = graph->node_by_id(operands.at(1).producer_node_id);
    balancer::BlockShape input1_block_shape = op_model.input_buffers[1].block_shape;
    auto const &tms = graph->get_edge_attributes(operands.at(1))->get_tms();

    auto input1_canonical_form = post_tms_shape(input1->shape(), tms);

    int act_t = input1_canonical_form.z();
    int u_kt = input1_block_shape.ublock.rt;
    int m_k = op_model.op_shape.inputs[1].rt / u_kt;
    return std::make_tuple(act_t, m_k, u_kt);
}

void post_placer_lower_buda_attrs(
    Graph *graph, DeviceConfig const &device_config, balancer::OpModelMap const &op_model_map)
{
    for (Node *node : graph->nodes())
    {
        if (node->node_type() == graphlib::NodeType::kBudaOp)
        {
            graphlib::BudaOpNode *op = node->as<graphlib::BudaOpNode>();
            graphlib::OpType &type = op->op_type();
            balancer::OpModel const &op_model = op_model_map.at(node->name());

            if (op->is_dense_matmul())
            {
                balancer::BlockShape input0_block_shape = op_model.input_buffers[0].block_shape;
                type.buda_attrs["m_k"] = op_model.op_shape.inputs[0].ct / input0_block_shape.ublock.ct;
                type.buda_attrs["u_kt"] = input0_block_shape.ublock.ct;
            }
            else if (op->is_sparse_matmul())
            {
                auto [act_t, m_k, u_kt] = calculate_sparse_mm_inner_dim(graph, op, op_model_map);
                type.buda_attrs["act_t"] = act_t;
                type.buda_attrs["m_k"] = m_k;
                type.buda_attrs["u_kt"] = u_kt;
            }
            else if (op->is_depthwise_matmul())
            {
                type.buda_attrs["u_kt"] = 1;  // hlk limitation
                type.buda_attrs["m_k"] = op_model.op_shape.inputs[1].rt;  // inner-dim of in1 in tiles
            }
            else if (type.op == "reduce")
            {
                balancer::BlockShape input0_block_shape = op_model.input_buffers[0].block_shape;
                if (std::get<std::string>(type.buda_attrs.at("dim")) == "r")
                {
                    type.buda_attrs["m_k"] = op_model.op_shape.inputs[0].rt / input0_block_shape.ublock.rt;
                    type.buda_attrs["u_kt"] = input0_block_shape.ublock.rt;
                }
                else if (std::get<std::string>(type.buda_attrs.at("dim")) == "c")
                {
                    type.buda_attrs["m_k"] = op_model.op_shape.inputs[0].ct / input0_block_shape.ublock.ct;
                    type.buda_attrs["u_kt"] = input0_block_shape.ublock.ct;
                }
            }

            set_kernel_broadcast(op, op_model);
            set_l1_accumulate(op, device_config.is_wormhole_b0());
        }
    }
}

graphlib::QueueNode *insert_epoch_to_epoch_queue(
    graphlib::Graph *graph,
    const std::string &name,
    graphlib::Edge edge,
    graphlib::UBlockOrder op_ublock_order,
    bool cross_epoch_type,
    bool cross_chip_type,
    graphlib::NodeEpochType user_epoch_type,
    graphlib::QueueNode *q = nullptr)
{
    TT_ASSERT(edge.edge_type == graphlib::EdgeType::kData, "Only data edge can be broken up with e2e queues");

    if (q == nullptr)
    {
        // Create new queue
        q = graph->add_node(
            graphlib::create_node<graphlib::EpochToEpochQueueNode>(name, cross_epoch_type, cross_chip_type),
            graph->get_subgraph_id_for_node(edge.producer_node_id));
        q->set_shape(graph->node_by_id(edge.producer_node_id)->shape());
        q->set_output_df(graph->node_by_id(edge.producer_node_id)->output_df());
        q->set_epoch_type(user_epoch_type);

        Edge node_to_q_edge(edge.producer_node_id, edge.producer_output_port_id, q->id(), 0, graphlib::EdgeType::kData);
        graph->add_edge(node_to_q_edge);
        graph->get_edge_attributes(node_to_q_edge)->set_ublock_order(op_ublock_order);
    }

    // Add edge from queue to consumer
    graphlib::Edge q_to_node_edge =
        Edge(q->id(), 0, edge.consumer_node_id, edge.consumer_input_port_id, graphlib::EdgeType::kData);
    graph->add_edge(q_to_node_edge);
    graph->copy_edge_attributes(edge, q_to_node_edge);
    graph->remove_edge(edge);

    return q;
}

graphlib::QueueNode *insert_buffering_queue(
    graphlib::Graph *graph,
    const std::string &name,
    graphlib::Edge edge,
    graphlib::UBlockOrder op_ublock_order,
    graphlib::QueueNode *q = nullptr)
{
    TT_ASSERT(edge.edge_type == graphlib::EdgeType::kData, "Only data edge can be broken up with this queue!");

    if (q == nullptr)
    {
        Node* producer_node = graph->node_by_id(edge.producer_node_id);
        q = graphlib::create_buffering_queue(graph, producer_node, name, graph->get_microbatch());

        Edge node_to_q_edge(edge.producer_node_id, edge.producer_output_port_id, q->id(), 0, graphlib::EdgeType::kData);
        graph->add_edge(node_to_q_edge);
        graph->get_edge_attributes(node_to_q_edge)->set_ublock_order(op_ublock_order);
    }

    // Add edge from queue to consumer
    graphlib::Edge q_to_node_edge =
        Edge(q->id(), 0, edge.consumer_node_id, edge.consumer_input_port_id, graphlib::EdgeType::kData);
    graph->add_edge(q_to_node_edge);
    graph->copy_edge_attributes(edge, q_to_node_edge);
    graph->remove_edge(edge);

    return q;
}

graphlib::Node *get_existing_fwd_e2e_queue(
    graphlib::Graph *graph, const placer::PlacerSolution &placer_solution, graphlib::Node *recompute_node)
{
    Node *fwd_node = graphlib::get_fwd_from_recompute(graph, recompute_node);
    std::uint32_t src_temporal_epoch = placer_solution.temporal_epoch_id(fwd_node->name());
    for (Edge e : graph->user_data_edges(fwd_node))
    {
        graphlib::Node *dest_node = graph->node_by_id(e.consumer_node_id);
        // If destination node is kQueue then placer_solution.temporal_epoch_id(dest_node->name()) doesn't contain
        // dest_node->name() To infer dest_temporal_epoch we use consumer
        if (dest_node->node_type() == tt::graphlib::kQueue)
        {
            dest_node = graph->node_by_id(graph->user_data_edges(dest_node)[0].consumer_node_id);
        }

        std::uint32_t dest_temporal_epoch = placer_solution.temporal_epoch_id(dest_node->name());
        if (src_temporal_epoch != dest_temporal_epoch)
        {
            log_debug(
                LogGraphCompiler,
                "recompute_node: {} mapped to fwd_node: {} and e2e queue: {}",
                recompute_node->name(),
                fwd_node->name(),
                dest_node->name());
            return fwd_node;
        }
    }
    return nullptr;
}

static void reconnect_recompute_consumers_to_fwd_queue(
    graphlib::Graph *graph, graphlib::Node *recompute_node, graphlib::Node *queue)
{
    for (Edge e : graph->user_data_edges(recompute_node))
    {
        graphlib::Edge new_edge = Edge(
            queue->id(),
            e.producer_output_port_id,
            e.consumer_node_id,
            e.consumer_input_port_id,
            graphlib::EdgeType::kData);
        graph->add_edge(new_edge);
        graph->copy_edge_attributes(e, new_edge);
        graph->remove_edge(e);
        log_debug(
            LogGraphCompiler,
            "\t Removing edge connecting: {}->{}",
            recompute_node->name(),
            graph->node_by_id(e.consumer_node_id)->name());
    }
}

void replace_recompute_with_checkpoint(graphlib::Graph *graph, const placer::PlacerSolution &placer_solution)
{
    std::vector<graphlib::Node *> nodes = graphlib::topological_sort(*graph);
    std::deque<std::pair<std::string, bool>> nodes_to_delete;

    for (graphlib::Node *node : nodes)
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
        {
            continue;
        }
        if (not graphlib::is_recompute(graph, node))
        {
            continue;
        }
        if (graph->num_users(node) == 0)
        {
            nodes_to_delete.emplace_back(node->name(), false);
        }

        if (graphlib::Node *queue = get_existing_fwd_e2e_queue(graph, placer_solution, node); queue != nullptr)
        {
            reconnect_recompute_consumers_to_fwd_queue(graph, node, queue);
            nodes_to_delete.emplace_back(node->name(), false);
            continue;
        }

        try
        {
            std::uint32_t src_temporal_epoch = placer_solution.temporal_epoch_id(node->name());
            for (Edge e : graph->user_data_edges(node))
            {
                graphlib::Node *dest_node = graph->node_by_id(e.consumer_node_id);
                if (dest_node->node_type() != graphlib::NodeType::kBudaOp)
                    continue;
                if (dest_node->node_type() == graphlib::NodeType::kQueue)
                    continue;

                try
                {
                    std::uint32_t dest_temporal_epoch = placer_solution.temporal_epoch_id(dest_node->name());

                    if (src_temporal_epoch > dest_temporal_epoch)
                    {
                        Node *fwd_node = graphlib::get_fwd_from_recompute(graph, node);

                        graphlib::Edge new_edge = Edge(
                            fwd_node->id(),
                            e.producer_output_port_id,
                            dest_node->id(),
                            e.consumer_input_port_id,
                            graphlib::EdgeType::kData);

                        graph->add_edge(new_edge);
                        log_trace(
                            LogGraphCompiler,
                            "Bypassing {}. Connecting {} to {}.",
                            node->name(),
                            fwd_node->name(),
                            dest_node->name());
                        log_trace(LogGraphCompiler, "src= {}, dst={}.", src_temporal_epoch, dest_temporal_epoch);

                        graph->copy_edge_attributes(e, new_edge);
                        graph->remove_edge(e);

                        nodes_to_delete.emplace_back(node->name(), false);
                        for (auto operand : graph->data_operands(node))
                        {
                            nodes_to_delete.emplace_back(operand->name(), false);
                        }
                    }
                }
                catch (std::out_of_range &e)
                {
                    throw std::runtime_error(
                        "Placement solution missing for node " + dest_node->name() +
                        " while inserting epoch_to_epoch queues");
                }
            }
        }
        catch (std::out_of_range &e)
        {
            throw std::runtime_error(
                "Placement solution missing for node " + node->name() + " while inserting epoch_to_epoch queues");
        }
    }
    while (not nodes_to_delete.empty())
    {
        auto [node_to_delete_name, force_delete] = nodes_to_delete.front();
        nodes_to_delete.pop_front();

        if (graph->has_node_with_name(node_to_delete_name))
        {
            Node *node_to_delete = graph->get_node_by_name(node_to_delete_name);
            if (force_delete)
            {
                graph->remove_node(node_to_delete);
            }
            else
            {
                if (graph->num_users(node_to_delete->id()) == 0)
                {
                    for (Node *operand : graph->data_operands(node_to_delete))
                    {
                        nodes_to_delete.emplace_back(operand->name(), false);
                    }
                    graph->remove_node(node_to_delete);
                }
            }
        }
    }
}

static bool feeds_remote_chips(
    const placer::PlacerSolution &placer_solution, graphlib::Graph *graph, graphlib::Node *producer)
{
    std::uint32_t producer_chip_id = placer_solution.chip_id(producer->name());
    for (const Edge &e : graph->user_data_edges(producer))
    {
        graphlib::Node *dest_node = graph->node_by_id(e.consumer_node_id);
        if (dest_node->node_type() == graphlib::NodeType::kBudaOp)
        {
            if (producer_chip_id != placer_solution.chip_id(dest_node->name()))
            {
                return true;
            }
        }
    }
    return false;
}

void validate_subgraph_placement(Graph *graph, placer::PlacerSolution &placer_solution)
{
    for (size_t epoch = 0; epoch < placer_solution.epoch_id_to_epoch_info.size(); epoch++)
    {
        std::vector<placer::OpPlacement> ops = placer_solution.epoch_id_to_op_placement[epoch];
        int epoch_subgraph_index = -1;
        for (placer::OpPlacement op : ops)
        {
            if (not graph->has_node_with_name(ops[0].name))
                continue;

            if (epoch_subgraph_index == -1)
                epoch_subgraph_index = graph->get_subgraph_id_for_node(graph->get_node_by_name(ops[0].name)->id());

            TT_ASSERT(graph->get_subgraph_id_for_node(graph->get_node_by_name(ops[0].name)->id()) == (unsigned int)epoch_subgraph_index);
        }
        // If a subgraph index was found, set it for the epoch
        if (epoch_subgraph_index != -1)
            placer_solution.epoch_id_to_subgraph_index[epoch] = epoch_subgraph_index;
        else
        {
            // otherwise for empty epochs deafult to 0
            placer_solution.epoch_id_to_subgraph_index[epoch] = 0;
        }
    }
}

static bool feeds_multiple_remote_consumers(
    const placer::PlacerSolution &placer_solution, graphlib::Graph *graph, graphlib::Node *producer)
{
    std::set<std::uint32_t> consumer_chip_ids;
    for (const Edge &e : graph->user_data_edges(producer))
    {
        graphlib::Node *dest_node = graph->node_by_id(e.consumer_node_id);
        if (dest_node->node_type() == graphlib::NodeType::kBudaOp)
        {
            consumer_chip_ids.insert(placer_solution.chip_id(dest_node->name()));
        }
    }
    return consumer_chip_ids.size() > 1;
}

bool any_consumers_cross_epoch(graphlib::Graph *graph, graphlib::Node *producer)
{
    bool cross_epoch_type_across_all_users = false;
    for (const Edge &e : graph->user_data_edges(producer))
    {
        graphlib::Node *dest_node = graph->node_by_id(e.consumer_node_id);
        cross_epoch_type_across_all_users |= producer->get_epoch_type() != dest_node->get_epoch_type();
    }
    return cross_epoch_type_across_all_users;
}

// Remove buffering queues connecting cross epoch nodes so that E2E queues can be inserted instead.
void remove_buffering_queues_from_cross_epoch_edges(
    graphlib::Graph *graph, const placer::PlacerSolution &placer_solution)
{
    for (graphlib::Node *node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() != graphlib::NodeType::kQueue)
            continue;

        graphlib::QueueNode *queue = static_cast<graphlib::QueueNode *>(node);

        if (queue->queue_type() == graphlib::QueueNodeType::Buffering)
        {
            Node *source_node = graph->data_operands(queue).back();

            if (source_node->node_type() == graphlib::NodeType::kBudaOp)
            {
                std::uint32_t src_temporal_epoch = placer_solution.temporal_epoch_id(source_node->name());
                auto user_edges = graph->user_data_edges(queue);
                bool no_users_in_src_epoch = true;
                for (std::size_t i = 0; i < user_edges.size(); i++)
                {
                    Node *dest_node = graph->node_by_id(user_edges[i].consumer_node_id);
                    if (dest_node->node_type() == graphlib::NodeType::kBudaOp)
                    {
                        std::uint32_t dest_temporal_epoch = placer_solution.temporal_epoch_id(dest_node->name());
                        if (src_temporal_epoch != dest_temporal_epoch)
                        {
                            // if this is the last user of the queue and none of the previous users were in the same
                            // epoch as producer, then all users are connected to source by e2e queue and we should
                            // remove buffering queue.
                            bool remove_queue = (i == user_edges.size() - 1 && no_users_in_src_epoch);
                            connect_queue_src_to_queue_user(graph, queue, user_edges[i], remove_queue);
                        }
                        else
                        {
                            no_users_in_src_epoch = false;
                        }
                    }
                }
            }
        }
    }
}

// Insert a queue between every two ops that are not in the same epoch
void insert_epoch_to_epoch_queues(
    graphlib::Graph *graph,
    const placer::PlacerSolution &placer_solution,
    const std::unordered_set<graphlib::NodeEpochType> &epoch_types,
    const balancer::CutEdges &graph_solver_cut_edges)
{
    bool firmware_looping_enabled = env_as<int>("NUM_EXEC_LOOP_ITERATIONS", 0) > 1;
    for (graphlib::Node *node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
            continue;
        if (epoch_types.find(node->get_epoch_type()) == epoch_types.end())
            continue;

        int added_q_count = 0;

        bool minimize_remote_dram_queues = env_as<bool>("PYBUDA_MINIMIZE_REMOTE_DRAM_QUEUES");

        try
        {
            std::uint32_t src_temporal_epoch = placer_solution.temporal_epoch_id(node->name());

            // Only create one e2e queue for each destination epoch
            std::unordered_map<std::uint32_t, graphlib::QueueNode *> e2e_queues;
            graphlib::QueueNode *e2e_q = nullptr;
            graphlib::QueueNode *buf_q = nullptr;

            const bool producer_feeds_multiple_remote_consumers =
                feeds_multiple_remote_consumers(placer_solution, graph, node);
            const bool producer_feeds_cross_epoch_consumer = any_consumers_cross_epoch(graph, node);
            const bool producer_feeds_remote_chip = feeds_remote_chips(placer_solution, graph, node);

            for (Edge e : graph->user_data_edges(node))
            {
                bool graph_solver_cut_edge = graph_solver_cut_edges.count(e) > 0;
                graphlib::Node *dest_node = graph->node_by_id(e.consumer_node_id);
                if (dest_node->node_type() != graphlib::NodeType::kBudaOp)
                    continue;
                if (dest_node->node_type() == graphlib::NodeType::kQueue)
                    continue;

                try
                {
                    std::uint32_t dest_temporal_epoch = placer_solution.temporal_epoch_id(dest_node->name());
                    TT_ASSERT(placer_solution.epoch_id_to_subgraph_index.at(src_temporal_epoch) == placer_solution.epoch_id_to_subgraph_index.at(dest_temporal_epoch), "e2e queues across subgraphs not allowed");
                    if (src_temporal_epoch > dest_temporal_epoch)
                    {
                        log_error(
                            "Error creating e2e queue (likely an issue with pybuda placer):"
                            "producer op ({}, {}) is placed in a later epoch than the dest op ({}, {}).",
                            node->name(),
                            src_temporal_epoch,
                            dest_node->name(),
                            dest_temporal_epoch);
                    }

                    bool should_insert_e2e_queue = src_temporal_epoch != dest_temporal_epoch;
                    bool should_insert_buffering_queue = graph_solver_cut_edge;

                    if (node->as<OpNode>()->is_gradient_op() and node->get_epoch_type() != dest_node->get_epoch_type())
                    {
                        connect_gradient_accum_queue(graph, node, e);
                    }
                    else if (
                        producer_feeds_multiple_remote_consumers and should_insert_e2e_queue and
                        not minimize_remote_dram_queues)
                    {
                        Edge operand = graph->operand_data_edges(node).back();
                        graphlib::UBlockOrder op_ublock_order = graph->get_edge_attributes(operand)->get_ublock_order();
                        bool cross_epoch_type = node->get_epoch_type() != dest_node->get_epoch_type();

                        graphlib::QueueNode *e2e_q = nullptr;
                        auto it = e2e_queues.find(dest_temporal_epoch);
                        if (it != e2e_queues.end())
                            e2e_q = it->second;

                        e2e_q = insert_epoch_to_epoch_queue(
                            graph,
                            "e2e_" + node->name() + "_" + std::to_string(added_q_count++),
                            e,
                            op_ublock_order,
                            cross_epoch_type,
                            producer_feeds_remote_chip,
                            dest_node->get_epoch_type(),
                            e2e_q);

                        e2e_queues[dest_temporal_epoch] = e2e_q;
                    }
                    else if (should_insert_e2e_queue)
                    {
                        Edge operand = graph->operand_data_edges(node).back();
                        graphlib::UBlockOrder op_ublock_order = graph->get_edge_attributes(operand)->get_ublock_order();

                        e2e_q = insert_epoch_to_epoch_queue(
                            graph,
                            "e2e_" + node->name() + "_" + std::to_string(added_q_count++),
                            e,
                            op_ublock_order,
                            producer_feeds_cross_epoch_consumer,
                            producer_feeds_remote_chip,
                            dest_node->get_epoch_type(),
                            e2e_q);
                    }
                    else if (should_insert_buffering_queue)
                    {
                        // For graph solver cut edge add bufferring queue instead but only if E2E queue is not already
                        // added.
                        //
                        Edge operand = graph->operand_data_edges(node).back();
                        graphlib::UBlockOrder op_ublock_order = graph->get_edge_attributes(operand)->get_ublock_order();

                        auto q_ptr = insert_buffering_queue(
                            graph,
                            "buf_" + node->name() + "_" + std::to_string(added_q_count++),
                            e,
                            op_ublock_order,
                            buf_q);
                        if (!firmware_looping_enabled)
                        {
                            // With FW looping, we duplicate the buffering to avoid a deadlock scenario bug. This is a
                            // workaround otherwise, we reuse the buf_q (this path)
                            buf_q = q_ptr;
                        }
                    }
                }
                catch (std::out_of_range &e)
                {
                    throw std::runtime_error(
                        "Placement solution missing for node " + dest_node->name() +
                        " while inserting epoch_to_epoch queues");
                }
            }
        }
        catch (std::out_of_range &e)
        {
            throw std::runtime_error(
                "Placement solution missing for node " + node->name() + " while inserting epoch_to_epoch queues");
        }
    }
}

void connect_gradient_accum_queue(graphlib::Graph *graph, Node *node, const graphlib::Edge &edge)
{
    // fetch the gradient queue, reconnect to optimizer input and delete old edge
    for (Node *user : graph->data_users(node))
    {
        if (user->node_type() == graphlib::NodeType::kQueue and user->as<graphlib::QueueNode>()->is_grad_accumulator())
        {
            graphlib::Edge q_to_node_edge =
                Edge(user->id(), 0, edge.consumer_node_id, edge.consumer_input_port_id, graphlib::EdgeType::kData);
            graph->add_edge(q_to_node_edge);
            graph->copy_edge_attributes(edge, q_to_node_edge);
            graph->remove_edge(edge);
        }
    }
}

// Set queue entry sizes based on the configuration for different types of queues
void set_queue_sizes(graphlib::Graph *graph, PostPlacerConfig &config, const placer::PlacerSolution &placer_solution)
{
    for (graphlib::Node *node : graph->nodes())
    {
        int size = -1;
        if (node->node_type() == graphlib::NodeType::kInput)
        {
            bool constant = (node->as<graphlib::InputNode>()->is_constant());  // hacky, fix!
            bool optimizer_param = (node->as<graphlib::InputNode>()->is_optimizer_parameter());
            if ((node->as<graphlib::InputNode>()->is_parameter()) || constant || optimizer_param)
            {
                // TODO(jchu): This needs to get updated for repeat structures
                size = 1;
            }
            else
            {
                size = env_as<int>("PYBUDA_OVERRIDE_INPUT_QUEUE_ENTRIES", config.input_queue_multiplier * graph->get_microbatch());
            }
        }
        else if (node->node_type() == graphlib::NodeType::kOutput)
        {
            size = config.output_queue_multiplier * graph->get_microbatch();
        }
        else if (node->node_type() == graphlib::NodeType::kQueue)
        {
            graphlib::QueueNode *qnode = node->as<graphlib::QueueNode>();
            if (qnode->is_grad_accumulator())
            {
                size = 1;  // TODO: grad accumulator should be a ram
            }
            else if (qnode->is_epoch_to_epoch())
            {
                graphlib::EpochToEpochQueueNode *e2e = qnode->as<graphlib::EpochToEpochQueueNode>();

                if (e2e->is_cross_epoch_type())
                {
                    // TODO: Check for invalid epoch-to-epoch crossing
                    size = config.microbatch_size * config.microbatch_count;
                }
                else
                {
                    // Need to cover the delta between epochs within one loop
                    std::uint32_t last_epoch_use = get_last_epoch_use(graph, qnode, placer_solution);
                    std::uint32_t first_epoch_producer = get_first_epoch_producer(graph, qnode, placer_solution);

                    TT_LOG_ASSERT(
                        last_epoch_use >= first_epoch_producer,
                        "e2e queue: {} is going backwards in epochs.",
                        qnode->name());
                    // Need to cover the delta between chips if set in a pipeline
                    // TODO: for wormhole, we don't need this much buffering if it's from one
                    // temporal epoch to another!
                    try
                    {
                        int first_epoch_chip_id = placer_solution.epoch_id_to_chip.at(first_epoch_producer);
                        int last_epoch_chip_id = placer_solution.epoch_id_to_chip.at(last_epoch_use);
                        int chip_to_chip_delta = -1;
                        if (config.enable_cross_chip_buffering)
                        {
                            // on WH, the chip id may not be consecutive
                            chip_to_chip_delta = abs(last_epoch_chip_id - first_epoch_chip_id) + 1;
                        }
                        else
                        {
                            chip_to_chip_delta = 1;
                        }

                        if (e2e->get_epoch_type() == graphlib::NodeEpochType::Optimizer)
                        {
                            size = chip_to_chip_delta;  // optimizer always works on one element only
                        }
                        else
                        {
                            size = config.microbatch_size * chip_to_chip_delta;
                        }
                    }
                    catch (std::out_of_range &e)
                    {
                        log_fatal("Not all epochs in chip map");
                    }
                }
            }
            else if (qnode->is_buffering())
            {
                // Skip. The size is either user-configured or set earlier by some pass
            }
            else
            {
                // TODO: what else falls in here?
                TT_ASSERT(false);
            }
        }

        if (size > 0)
        {
            node->as<graphlib::QueueNode>()->set_num_entries(size);
        }
    }
}

std::vector<std::uint32_t> get_consumer_epoch_ids(
    const graphlib::Graph *graph, const graphlib::Node *node, const placer::PlacerSolution &placer_solution)
{
    std::vector<std::uint32_t> consumer_epoch_ids;
    std::vector<graphlib::Node *> users = graph->data_users(node);
    try
    {
        for (Node *user : users)
        {
            consumer_epoch_ids.push_back(placer_solution.temporal_epoch_id(user->name()));
        }
        return consumer_epoch_ids;
    }
    catch (std::out_of_range &e)
    {
        log_fatal("Placement missing for a user of {}", node->name());
        return {};
    }
}

std::uint32_t get_last_epoch_use(
    const graphlib::Graph *graph, const graphlib::Node *node, const placer::PlacerSolution &placer_solution)
{
    std::vector<std::uint32_t> consumer_epoch_ids = get_consumer_epoch_ids(graph, node, placer_solution);
    return *std::max_element(consumer_epoch_ids.begin(), consumer_epoch_ids.end());
}

// Return first/last epoch in which this node's output is used
std::uint32_t get_first_epoch_producer(
    const graphlib::Graph *graph, const graphlib::Node *node, const placer::PlacerSolution &placer_solution)
{
    std::vector<graphlib::Node *> operands = graph->operands(node);
    try
    {
        std::uint32_t min_epoch = placer_solution.temporal_epoch_id(operands[0]->name());
        for (std::size_t i = 1; i < operands.size(); i++)
        {
            std::uint32_t epoch = placer_solution.temporal_epoch_id(operands[i]->name());
            if (epoch < min_epoch)
                min_epoch = epoch;
        }
        return min_epoch;
    }
    catch (std::out_of_range &e)
    {
        log_fatal("Placement missing for an operand of {}", node->name());
        return 0;
    }
}

void validate_multichip_queue_placements(
    const PostPlacerConfig &config, const graphlib::Graph *graph, const placer::PlacerSolution &placer_solution)
{
    if (not config.device_config.is_grayskull() or config.device_config.chip_ids.size() == 1)
    {
        return;
    }

    bool pass = true;
    for (graphlib::Node *node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() != graphlib::NodeType::kQueue)
        {
            continue;
        }
        graphlib::QueueNode *queue_node = node->as<graphlib::QueueNode>();

        if (queue_node->is_grad_accumulator())
        {
            Node *gradient_op = graph->data_operands(queue_node).at(0);
            if (placer_solution.chip_id(gradient_op->name()) != placer_solution.chip_id(queue_node->name()))
            {
                log_error(
                    "Error: gradient op ({}, chip_id: {}) but grad_accum queue ({}, chip_id: {})",
                    gradient_op->name(),
                    placer_solution.chip_id(gradient_op->name()),
                    queue_node->name(),
                    placer_solution.chip_id(queue_node->name()));
                pass = false;
            }
        }
        else if (queue_node->is_epoch_to_epoch())
        {
            Node *producer_op = graph->data_operands(queue_node).at(0);
            Node *consumer_op = graph->data_users(queue_node).at(0);
            int producer_chip_id = placer_solution.chip_id(producer_op->name());
            int consumer_chip_id = placer_solution.chip_id(consumer_op->name());

            if (producer_op->get_epoch_type() == graphlib::NodeEpochType::Forward and
                consumer_op->get_epoch_type() == graphlib::NodeEpochType::Forward)
            {
                bool valid_dataflow =
                    (consumer_chip_id == producer_chip_id) || (consumer_chip_id == producer_chip_id + 1);
                if (not valid_dataflow)
                {
                    log_error(
                        "Error: producer op ({}, chip_id: {}) but consumer op({}, chip_id: {})",
                        producer_op->name(),
                        placer_solution.chip_id(producer_op->name()),
                        consumer_op->name(),
                        placer_solution.chip_id(consumer_op->name()));
                    pass = false;
                }
            }
            else if (
                producer_op->get_epoch_type() == graphlib::NodeEpochType::Backward and
                consumer_op->get_epoch_type() == graphlib::NodeEpochType::Backward)
            {
                bool valid_dataflow =
                    (consumer_chip_id == producer_chip_id) || (consumer_chip_id == producer_chip_id - 1);
                if (not valid_dataflow)
                {
                    log_error(
                        "Error: producer op ({}, chip_id: {}) but consumer op({}, chip_id: {})",
                        producer_op->name(),
                        placer_solution.chip_id(producer_op->name()),
                        consumer_op->name(),
                        placer_solution.chip_id(consumer_op->name()));
                    pass = false;
                }
            }
        }
    }
    if (not pass)
    {
        log_fatal("validate_multichip_queue_placements FAILED");
    }
}

}  // namespace tt


