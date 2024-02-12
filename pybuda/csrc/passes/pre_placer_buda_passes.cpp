// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/pre_placer_buda_passes.hpp"

#include "autograd/binding.hpp"
#include "passes/fuse_ops.hpp"
#include "passes/lowering_context.hpp"
#include "passes/passes_utils.hpp"
#include "placer/lower_to_placer.hpp"
#include "utils/logger.hpp"

namespace tt {

void lower_to_buffering_queues(Graph* graph) {
    vector<Node*> nodes = graphlib::topological_sort(*graph);
    for (Node* node : nodes) {
        if (node->node_type() == NodeType::kBudaOp and node->as<graphlib::BudaOpNode>()->op_name() == "dram_queue") {
            int entries = std::get<int>(node->as<graphlib::BudaOpNode>()->op_attrs().at(0));
            auto buffering_queue = graph->add_node(
                graphlib::create_node<graphlib::BufferingQueueNode>("lowered_" + node->name(), entries),
                graph->get_subgraph_id_for_node(node->id()));
            graphlib::replace_node(graph, node, buffering_queue, false);
        }
    }
}

void split_unsupported_gradient_ops(graphlib::Graph *graph, const DeviceConfig &device_config)
{
    std::unordered_set<std::string> supported_gradient_ops = {"matmul", "multiply", "nop"};

    // On WH_B0, supported_gradient_ops include add/subtract
    if(device_config.is_wormhole_b0()) supported_gradient_ops.insert({"add", "subtract"});

    auto is_unsupported_gradient_op = [&supported_gradient_ops](Node *n)
    {
        if (n->node_type() != graphlib::kBudaOp)
            return false;
        return n->as<graphlib::BudaOpNode>()->is_gradient_op() and
               supported_gradient_ops.find(n->as<graphlib::BudaOpNode>()->op_name()) == supported_gradient_ops.end();
    };

    std::vector<Node *> unsupported_gradient_ops = graphlib::topological_sort(*graph, is_unsupported_gradient_op);
    for (Node *node : unsupported_gradient_ops)
    {
        graphlib::BudaOpNode *op = dynamic_cast<graphlib::BudaOpNode *>(node);
        TT_ASSERT(op);
        auto users = graph->user_data_edges(node);
        TT_ASSERT(users.size() == 1, "Gradient ops should only have a single user");
        auto user_edge = users[0];
        graphlib::BudaOpNode *gradient_accumulator =
            graph->add_node(
                op->clone(op->name() + "_gradient_accumulator"),
                graph->get_subgraph_id_for_node(op->id()))->as<graphlib::BudaOpNode>();
        gradient_accumulator->change_op_type(OpType("nop", {}, {}));
        insert_node_on_edge(graph, user_edge, gradient_accumulator);
        gradient_accumulator->set_gradient_op(true);
        op->set_gradient_op(false);
    }
}

static bool compatible_relu_attrs(BudaOpAttrs const &dst_attrs, BudaOpAttrs const &src_attrs)
{
    if (dst_attrs.find("relu_en") == dst_attrs.end())
        return true;

    std::string dst_relu_mode =
        (dst_attrs.find("relu_mode") != dst_attrs.end()) ? std::get<string>(dst_attrs.at("relu_mode")) : "min";
    std::string src_relu_mode =
        (src_attrs.find("relu_mode") != src_attrs.end()) ? std::get<string>(src_attrs.at("relu_mode")) : "min";
    return dst_relu_mode == src_relu_mode;
}

static bool can_hoist_relu(graphlib::Graph *graph, Node *nop)
{
    vector<Node *> operand_nodes = graph->data_operands(nop);
    TT_ASSERT(operand_nodes.size() == 1);
    graphlib::BudaOpNode *producer = dynamic_cast<graphlib::BudaOpNode *>(operand_nodes[0]);

    // Can only hoist relu into ops
    if (not producer)
        return false;

    // Cannot hoist gradient nops
    if (nop->as<graphlib::BudaOpNode>()->is_gradient_op() or producer->is_gradient_op())
        return false;

    if (producer->is_fused_op())
        return false;

    bool producer_forks = graph->data_users(producer).size() > 1;
    if (producer_forks)
        return false;

    BudaOpAttrs const &producer_attrs = producer->buda_attrs();
    BudaOpAttrs const &nop_attrs = nop->as<graphlib::BudaOpNode>()->buda_attrs();
    for (auto const &[k, v] : nop_attrs)
    {
        if (k != "relu_en" and k != "relu_mode" and k != "relu_threshold")
            return false;
    }

    if (not compatible_relu_attrs(producer_attrs, nop_attrs))
        return false;

    return true;
}

static void hoist_relu(graphlib::Graph *graph, Node *nop)
{
    vector<Node *> operand_nodes = graph->data_operands(nop);
    TT_ASSERT(operand_nodes.size() == 1);
    Node *producer = operand_nodes[0];

    BudaOpAttrs const &producer_attrs = producer->as<graphlib::BudaOpNode>()->buda_attrs();
    BudaOpAttrs relu_attrs = nop->as<graphlib::BudaOpNode>()->buda_attrs();

    if (producer_attrs.find("relu_threshold") != producer_attrs.end())
    {
        std::string relu_mode = (producer_attrs.find("relu_mode") != producer_attrs.end())
                                    ? std::get<string>(producer_attrs.at("relu_mode"))
                                    : "min";
        TT_ASSERT(relu_mode == "min" or relu_mode == "max");
        float producer_threshold = std::get<float>(producer_attrs.at("relu_threshold"));
        float nop_threshold = (relu_attrs.find("relu_threshold") != relu_attrs.end())
                                  ? std::get<float>(relu_attrs.at("relu_threshold"))
                                  : 0.0;
        relu_attrs["relu_threshold"] = (relu_mode == "min") ? std::max(producer_threshold, nop_threshold)
                                                            : std::min(producer_threshold, nop_threshold);
    }

    graphlib::OpType op_type = producer->as<graphlib::BudaOpNode>()->op_type();
    for (auto const &[k, v] : relu_attrs) op_type.buda_attrs[k] = v;
    producer->as<graphlib::BudaOpNode>()->change_op_type(op_type);
}

// Remove NOPs from the graph that have no TMs
void remove_nops(graphlib::Graph *graph)
{
    auto is_nop_node = [](Node *n) {
        return (n->node_type() == graphlib::kBudaOp) && (n->as<graphlib::BudaOpNode>()->op_type() == "nop");
    };

    auto cannot_bypass_nary_tm_operand = [graph](Node *n)
    {
        bool operand_is_nary_tm = graph->data_operands(n)[0]->node_type() == graphlib::kBudaNaryTM;
        bool user_is_op = graph->data_operands(n)[0]->node_type() == graphlib::kBudaOp;
        return operand_is_nary_tm and not user_is_op;
    };

    auto is_transpose_lhs_to_matmul = [graph](Node *n)
    {
        std::vector<graphlib::Edge> operands = graph->operand_data_edges(n);
        TT_ASSERT(operands.size() == 1);
        auto edge_attributes = graph->get_edge_attributes(operands[0]);
        if (not edge_attributes->has_tm("transpose"))
            return false;

        for (graphlib::Edge const &edge : graph->user_data_edges(n))
        {
            graphlib::OpNode *user_op = dynamic_cast<graphlib::OpNode *>(graph->node_by_id(edge.consumer_node_id));
            if (user_op and user_op->is_matmul() and edge.consumer_input_port_id == 0)
                return true;
        }
        return false;
    };

    // copy into separate vector because of iterator invalidation from removing nodes during iteration
    vector<Node*> nop_nodes = graphlib::topological_sort(*graph, is_nop_node);

    for (Node *node : nop_nodes)
    {
        if ((node->as<tt::graphlib::TaggedNode>()->has_tag("padding_nop")) or (node->as<tt::graphlib::TaggedNode>()->has_tag("dont_remove"))) {
            continue;
        }

        if (cannot_bypass_nary_tm_operand(node))
        {
            continue;
        }
        if (can_hoist_relu(graph, node))
        {
            hoist_relu(graph, node);
            graphlib::bypass_node(graph, node, true);
            continue;
        }
        if (not node->as<graphlib::BudaOpNode>()->buda_attrs().empty()) {
            // by default, just skip for now until we get a clearer picture on which nops can be skipped
            continue;
        }
        if (node->as<graphlib::BudaOpNode>()->is_gradient_op()) {
            // don't remove nops that have the `is_gradient_op` flag tagged
            continue;
        }
        if (is_transpose_lhs_to_matmul(node))
        {
            continue;
        }

        vector<Edge> operand_edges = graph->operand_data_edges(node);
        TT_ASSERT(operand_edges.size() == 1);
        const Edge& producer_to_nop_edge = operand_edges[0];
        Node* producer = graph->node_by_id(producer_to_nop_edge.producer_node_id);

        bool is_output_node_immediate_user = false;
        for (Node* user : graph->data_users(node)) {
            if (user->node_type() == graphlib::NodeType::kOutput or user->node_type() == graphlib::NodeType::kQueue) {
                is_output_node_immediate_user = true;
            }
        }

        if (producer->node_type() == graphlib::NodeType::kInput and is_output_node_immediate_user)
        {
            // avoid situations where we would have an input_node feeding an output node
            // causing a few issues down the stack.
            continue;
        }

        if (not graph->get_edge_attributes(producer_to_nop_edge)->get_tms().empty() and is_output_node_immediate_user)
        {
            // avoid situations where we delete a nop with input tms feeding an output node
            continue;
        }

        graphlib::bypass_node(graph, node, true);
    }
}


void tag_subgraph_nodes(graphlib::Graph *graph)
{
    auto is_op_node = [](graphlib::Node *n)
    {
        return (n->node_type() == graphlib::kBudaOp);
    };
    auto is_input_node = [](graphlib::Node *n)
    {
        return (n->node_type() == graphlib::kInput);
    };
    std::vector<std::vector<graphlib::Node *>> subgraphs;
    for (graphlib::Node *output_node: graph->ordered_module_outputs())
    {
        std::vector<graphlib::Node *> reachable = reachable_nodes(graph, output_node, is_op_node);
        if (reachable.size() == graph->nodes_by_type(graphlib::kBudaOp).size())  // No disjoint subgraphs
            return;

        bool found = false;
        unsigned int subgraph_index = 0;
        unsigned int io_subgraph_index;
        for (auto subgraph : subgraphs)
        {            
            if (std::is_permutation(subgraph.begin(), subgraph.end(), reachable.begin()))
            {
                found = true;
                io_subgraph_index = subgraph_index;
            }
            subgraph_index++;
        }

        if (not found)
        {
            subgraphs.push_back(reachable);
            io_subgraph_index = subgraphs.size() - 1;
            log_info(LogGraphCompiler, "New subgraph from: {}", output_node->name());
        }
        graph->move_node_to_subgraph(output_node->id(), io_subgraph_index);
        std::vector<graphlib::Node *> inputs = reachable_nodes(graph, output_node, is_input_node);
        for (graphlib::Node *input_node : inputs)
            graph->move_node_to_subgraph(input_node->id(), io_subgraph_index);
    }
    unsigned int subgraph_index = 0;
    log_info(LogGraphCompiler, "Dividing graph into: {} subgraphs", subgraphs.size());
    for (auto subgraph : subgraphs)
    {
        for (auto node : subgraph)
        {
            graph->move_node_to_subgraph(node->id(), subgraph_index);
            log_debug(LogGraphCompiler, "Node: {} subgraph index: {}", node->name(), subgraph_index);
        }
        subgraph_index++;
    }

    return;
} 

// Remove transposes on srcB by inserting nops where needed. Supports WH_b0 Architecture
void fix_transposes(graphlib::Graph *graph, const DeviceConfig &device_config)
{
    for (graphlib::Node *node : graphlib::topological_sort(*graph)) 
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
            continue;

        graphlib::BudaOpNode *opnode = node->as<graphlib::BudaOpNode>();

        // No need to insert nop before a nop
        if(opnode->op_type().op == "nop")
            continue;
         
        std::vector<Edge> operand_data_edges = graph->operand_data_edges(node);
        std::vector<Edge> edges_to_check;
        edges_to_check.reserve(2);

        // WH_b0 Hardware supports transpose feeding into srcA reg, so no need to insert a nop for element-wise ops for input_indx == 0
        if(device_config.is_wormhole_b0() and graphlib::is_eltwise(opnode) and (graphlib::is_eltwise_unary(opnode) or graphlib::is_eltwise_binary(opnode))){
            TT_ASSERT(operand_data_edges.size() > 0);
            
            // EXCEPTION = We dont support if input 1 == matmul
            bool is_input1_matmul = (operand_data_edges.size() > 1) && [&]() -> bool{
                auto producer_input1 = graph->node_by_id(operand_data_edges.at(1).producer_node_id);
                return (producer_input1->node_type() == graphlib::NodeType::kBudaOp) && (producer_input1->as<graphlib::BudaOpNode>()->is_matmul());
            }();
            
            auto start_indx = is_input1_matmul ? 0 : 1;
            edges_to_check = std::vector<Edge>(operand_data_edges.begin() + start_indx, operand_data_edges.end());
        }
        else if (opnode->is_matmul()) {
            // matmul sends operand 0 to src B.. maybe it shouldn't?
            edges_to_check.push_back(operand_data_edges[0]);
        }
        else {
            // All other ops do not support transpose TM anywhere, check all data edges
            edges_to_check = operand_data_edges;
        }

        for (std::size_t i = 0; i < edges_to_check.size(); i++) {
            Edge edge = edges_to_check[i];
            try_insert_nop_on_transpose_edge(graph, edge);
        }
    }
}

// Insert NOPs on outputs that have a TM on their input, since outputs aren't capable of performing TMs
void fix_tms_on_output(graphlib::Graph *graph)
{
    for (Node *n: graph->nodes_by_type(graphlib::NodeType::kOutput))
    {
        std::vector<Edge> edges = graph->operand_data_edges(n);
        TT_ASSERT(edges.size() == 1);
        std::vector<graphlib::OpType> tms = graph->get_edge_attributes(edges[0])->get_tms();

        if (tms.size() == 0)
            continue;

        graphlib::BudaOpNode *nop = graph->add_node(graphlib::create_node<graphlib::BudaOpNode>(
                    n->name() + "_tm_nop", "nop"), graph->get_subgraph_id_for_node(n->id()));
        nop->copy_parent_op_attributes(graph->node_by_id(edges[0].producer_node_id)->as<graphlib::BudaOpNode>());
        graphlib::insert_node_on_edge(graph, edges[0], nop);
    }
}

void insert_queues_for_op_intermediates(graphlib::Graph *graph, const std::vector<std::string> &op_intermediates_to_save)
{
    for (const std::string &op_intermediate : op_intermediates_to_save)
    {
        if (not graph->has_node_with_name(op_intermediate))
        {
            log_error(
                "Intermediate queue insertion on node with name {}, but {} not found in graph",
                op_intermediate,
                op_intermediate);
        }
        auto node = graph->get_node_by_name(op_intermediate);
        if (node->node_type() != graphlib::NodeType::kBudaOp)
        {
            log_error(
                "Intermediate queue insertion on node with name {}, but {} is not a BudaOpNode",
                op_intermediate,
                op_intermediate);
        }

        auto intermediate_output =
            graph->add_node(
                graphlib::create_node<graphlib::OutputNode>(node->name() + "_intermediate_output"),
                graph->get_subgraph_id_for_node(node->id()));
        graph->add_edge(node, intermediate_output);
        intermediate_output->set_shape(Shape::create(node->shape().as_vector()));
        intermediate_output->set_output_df(node->output_df());
        intermediate_output->set_saved_intermediate(true);
        intermediate_output->set_epoch_type(node->get_epoch_type());
    }
}

void sanitize_past_cache_ios(graphlib::Graph *graph)
{
    auto is_partial_datacopy_edge = [](Edge e) {
        return (e.edge_type == graphlib::EdgeType::kPartialDataCopy);
    };

    for (Node *node : graph->nodes_by_type(NodeType::kOutput))
    {
        std::vector<graphlib::Edge> partial_datacopy_edges = graph->user_edges(node, is_partial_datacopy_edge);
        if (partial_datacopy_edges.size() == 0)
            continue;
        
        auto output_node = node->as<graphlib::OutputNode>();
        output_node->set_untilize(false);
    }
}

// Insert NOPs on output-feeding ops that can't feed untilizer directly:
// - matmul
// - ops that feed something other than output
void fix_untilized_outputs(graphlib::Graph *graph, const DeviceConfig &device_config)
{
    // if multichip-wormhole, we add these untilize-nops indiscriminately for now
    // and will place these ops on an MMIO-capable device.
    bool is_multichip_wormhole = device_config.is_wormhole() and device_config.chip_ids.size() > 1;

    std::unordered_map<Node *, int> output_nop_indices; // For the cases where one op feeds multiple outputs
    for (Node *n: graph->nodes_by_type(graphlib::NodeType::kOutput))
    {
        auto output_node = n->as<graphlib::OutputNode>();
        if (not output_node->untilize())
            continue;

        std::vector<Edge> edges = graph->operand_data_edges(n);
        TT_ASSERT(edges.size() == 1);
        Node *source = graph->node_by_id(edges[0].producer_node_id);
        TT_ASSERT(source->node_type() == NodeType::kBudaOp);

        graphlib::BudaOpNode *node = source->as<graphlib::BudaOpNode>();

        bool is_reduce_z =
            (node->op_name() == "reduce") and (std::get<std::string>(node->buda_attrs().at("dim")) == "z");
        bool needs_nop =
            node->is_matmul() || is_reduce_z || (graph->data_users(node).size() > 1) || (is_multichip_wormhole) || node->is_fused_op();

        if (!needs_nop)
            continue;

        graphlib::BudaOpNode *nop = graph->add_node(graphlib::create_node<graphlib::BudaOpNode>(
                    node->name() + "_output_nop_" + std::to_string(output_nop_indices[node]++), "nop"),
                    graph->get_subgraph_id_for_node(n->id()));
        nop->copy_parent_op_attributes(source->as<graphlib::BudaOpNode>());
        nop->as<graphlib::TaggedNode>()->add_tags(source->as<graphlib::TaggedNode>()->get_tags());
        graphlib::insert_node_on_edge(graph, edges[0], nop);
    }
}

// Replace 'buffer' with 'nop' ops
void replace_buffers_with_nops (graphlib::Graph *graph)
{
    for (Node *n : graph->nodes_by_type(NodeType::kBudaOp))
    {
        auto op = n->as<graphlib::BudaOpNode>();
        if (op->op_type().op == "buffer") {
            auto op_type = op->op_type();
            op_type.op = "nop";
            op->change_op_type(op_type);
        }
    }
}

void insert_nop_on_matmul_input(graphlib::Graph *graph)
{
    for (Node *n : graph->nodes_by_type(NodeType::kBudaOp))
    {
        auto op = n->as<graphlib::BudaOpNode>();
        if (op->op_type().op == "matmul") {
            auto operand_edges = graph->operand_data_edges(op);
            if (operand_edges[0].producer_node_id == operand_edges[1].producer_node_id)
            {
                graphlib::BudaOpNode *nop = graph->add_node(graphlib::create_node<graphlib::BudaOpNode>(
                            op->name() + "_input_nop", "nop"), graph->get_subgraph_id_for_node(n->id()));
                nop->copy_parent_op_attributes(op->as<graphlib::BudaOpNode>());
                graphlib::insert_node_on_edge(graph, operand_edges[1], nop);
            }
        }
    }
}

void insert_tilize_op_on_input(graphlib::Graph *graph){

    for (Node* n: graph->nodes_by_type(NodeType::kInput))
    {
        auto op = n->as<graphlib::InputNode>();
        graphlib::RuntimeTensorTransform runtime_tensor_transform = op->get_runtime_tensor_transform();
        if(runtime_tensor_transform.type != graphlib::RuntimeTensorTransformType::Prestride && op->is_activation())
        {
            for (auto user_edge : graph->user_edges(op))
            {
                graphlib::BudaOpNode *tilize = graph->add_node(graphlib::create_node<graphlib::BudaOpNode>
                (op->name() + "_"+graph->node_by_id(user_edge.consumer_node_id)->name()+"_tilizer","tilizer"), graph->get_subgraph_id_for_node(n->id()));
                graphlib::insert_node_on_edge(graph, user_edge, tilize, true, true, 0, true);
            }
        }
    }
}


placer::PlacerConfigUpdate schedule_pre_placer_graph(
    graphlib::Graph *graph,
    DeviceConfig const &device_config,
    scheduler::SchedulerConfig const &scheduler_config,
    std::vector<std::uint32_t> const &chip_ids,
    std::vector<std::vector<std::string>> const &op_names_to_chip_break,
    std::vector<std::vector<std::string>> const &op_names_to_epoch_break,
    passes::FractureChipIdAssignments const &fracture_chip_id_assignments,
    std::string const &nops_remote_devices_postfix,
    bool use_interactive_placer)
{
    scheduler::Schedule scheduled_ops = run_scheduler(scheduler_config, graph);
    placer::ChipPlacerConfig chip_placer_config = {
        .chip_ids = chip_ids,
        .arch_name = device_config.arch_name,
        .op_to_epoch_type = placer::lowering::get_op_to_epoch_type_mapping(graph, scheduled_ops),
        .ops_tagged_for_chip_id_break =
            placer::lowering::tag_ops_for_chip_break(device_config.arch_name, op_names_to_chip_break, scheduled_ops, graph, use_interactive_placer),
        .ops_tagged_for_epoch_break = placer::lowering::tag_ops_for_epoch_break(
            device_config.arch_name,
            op_names_to_epoch_break,
            op_names_to_chip_break,
            scheduled_ops,
            graph,
            use_interactive_placer),
        .fracture_chip_id_assignments = fracture_chip_id_assignments,
        .fwd_to_bwd_nodes = placer::lowering::get_fwd_to_bwd_nodes(graph),
        .fwd_to_opt_nodes = placer::lowering::get_fwd_to_opt_nodes(graph, scheduled_ops),
        .output_ops = placer::lowering::get_output_nodes(graph),
        .chips_with_mmio = device_config.chips_with_mmio,
    };
    placer::OpToChipIdAssignment op_to_chip_id_assignment =
        get_op_to_chip_id_assignment(chip_placer_config, scheduled_ops);

    // update chip-id assignment, epoch breaking to accommodate fractured ops
    std::vector<std::vector<std::string>> updated_op_names_to_epoch_break = op_names_to_epoch_break;
    if (not fracture_chip_id_assignments.empty())
    {
        std::tie(op_to_chip_id_assignment, updated_op_names_to_epoch_break) = update_config_for_fractured_ops(
            chip_placer_config, op_names_to_epoch_break, scheduled_ops, op_to_chip_id_assignment);
    }
    // updated_op_names_to_epoch_break = update_epoch_breaks_for_partial_datacopy(graph, updated_op_names_to_epoch_break);

    if (device_config.is_grayskull() and chip_ids.size() > 1)
    {
        // we assume any forking chip_ids are ordered in the same way that they will be pipelineed.
        insert_nops_forking_to_remote_devices(graph, chip_ids, op_to_chip_id_assignment, nops_remote_devices_postfix);
    }
    // Recalculate shapes
    recalculate_shapes(graph);

    calculate_ublock_order(graph);

    // Run TM optimizations, we want minimal TMs so balancer can accurately calculate output buffer requirements
    optimize_tms(graph);

    validate_buffering_queues(graph);

    return placer::PlacerConfigUpdate(
        op_to_chip_id_assignment, op_names_to_chip_break, updated_op_names_to_epoch_break);
}

static void insert_nop_fork(
    graphlib::Graph *graph,
    Node *node,
    const std::vector<std::uint32_t> &chip_ids,
    const std::unordered_map<uint32_t, uint32_t> &chip_id_to_pipeline_index,
    const std::map<string, std::uint32_t> &user_remote_device_ids,
    placer::OpToChipIdAssignment &op_to_chip_id_assignment,
    std::string const &postfix)

{
    // case 3:
    // process with remote-device ids in sorted order
    std::vector<Node*> users = graph->data_users(node);
    for (Node* user : users) {
        if (user_remote_device_ids.find(user->name()) == user_remote_device_ids.end()) {
            log_fatal("{} is not tagged with a chip_id assignment", user->name());
        }
    }

    std::sort(users.begin(), users.end(), [&user_remote_device_ids, &chip_id_to_pipeline_index](const Node* a, const Node* b) {
        auto chip_a = user_remote_device_ids.at(a->name());
        auto chip_b = user_remote_device_ids.at(b->name());
        return chip_id_to_pipeline_index.at(chip_a) < chip_id_to_pipeline_index.at(chip_b);
    });

    // from receiver nodes,  min receive chip id and max-receive chip id
    TT_ASSERT(users.size() >= 2);
    std::uint32_t first_user_chip_id = user_remote_device_ids.at(users.front()->name());
    std::uint32_t last_user_chip_id = user_remote_device_ids.at(users.back()->name());

    int start_pipeline_index = chip_id_to_pipeline_index.at(first_user_chip_id);
    int end_pipeline_index = chip_id_to_pipeline_index.at(last_user_chip_id);
    std::unordered_map<std::uint32_t, Node*> pipeline_index_to_source;

    if (node->get_epoch_type() == graphlib::NodeEpochType::Backward and not graphlib::is_recompute(graph, node)) {
        pipeline_index_to_source[end_pipeline_index] = node;

        for (int pipeline_index = end_pipeline_index - 1; pipeline_index >= start_pipeline_index; --pipeline_index)
        {
            graphlib::BudaOpNode *nop = graph->add_node(graphlib::create_node<graphlib::BudaOpNode>(
                node->name() + "_chip_to_chip_nop_" + std::to_string(pipeline_index) + postfix, "nop"),
                graph->get_subgraph_id_for_node(node->id()));
            Node* previous_source = pipeline_index_to_source.at(pipeline_index+1);
            graph->add_edge(previous_source, nop);
            graph->copy_node_attributes(previous_source, nop);

            // If the producer was in backward (or optimizer) epoch, and there are fwd->bwd edges going to it,
            // the need to go to the new op, too
            for (Edge &e : graph->operand_edges(previous_source)) {
                // Adjust control & autograd edges
                if ( (e.edge_type == EdgeType::kAutogradFwdToBwd) || 
                     (e.edge_type == EdgeType::kAutogradFwdToOptimizer) || 
                     (e.edge_type == EdgeType::kAutogradFwdToRecompute)) 
                {
                    graph->add_edge(
                            graph->node_by_id(e.producer_node_id),
                            nop,
                            e.producer_output_port_id,
                            0,
                            e.edge_type);
                }
            }

            op_to_chip_id_assignment[nop->name()] = chip_ids.at(pipeline_index + 1);
            pipeline_index_to_source[pipeline_index] = nop;
        }

        Node* source_node = node;
        for (size_t user_index = 0; user_index < users.size(); ++user_index)
        {
            Node* user = users.at(user_index);
            std::uint32_t remote_chip_id = op_to_chip_id_assignment.at(user->name());
            int remote_chip_pipeline_index = chip_id_to_pipeline_index.at(remote_chip_id);
            if (remote_chip_pipeline_index == end_pipeline_index) {continue;}

            Node* new_source = pipeline_index_to_source.at(remote_chip_pipeline_index);

            for (auto old_edge : graph->get_edges(source_node, user)) {
                if (old_edge.edge_type == graphlib::EdgeType::kData) {
                    graphlib::Edge new_edge=
                        Edge(
                            new_source->id(),
                            old_edge.producer_output_port_id,
                            user->id(),
                            old_edge.consumer_input_port_id,
                            graphlib::EdgeType::kData);


                    graph->add_edge(new_edge);
                    graph->copy_edge_attributes(old_edge, new_edge);
                    graph->remove_edge(old_edge);
                }
            }
        }

    } else {
        pipeline_index_to_source[start_pipeline_index] = node;

        for (int pipeline_index = start_pipeline_index + 1; pipeline_index <= end_pipeline_index; ++pipeline_index)
        {
            graphlib::BudaOpNode *nop = graph->add_node(graphlib::create_node<graphlib::BudaOpNode>(
                node->name() + "_chip_to_chip_nop_" + std::to_string(pipeline_index) + postfix, "nop"),
                graph->get_subgraph_id_for_node(node->id()));
            Node* previous_source = pipeline_index_to_source.at(pipeline_index-1);
            graph->add_edge(previous_source, nop);
            graph->copy_node_attributes(previous_source, nop);

            // If the producer was in backward (or optimizer) epoch, and there are fwd->bwd edges going to it,
            // the need to go to the new op, too
            for (Edge &e : graph->operand_edges(previous_source)) {
                // Adjust control & autograd edges
                if ( (e.edge_type == EdgeType::kAutogradFwdToBwd) || 
                     (e.edge_type == EdgeType::kAutogradFwdToOptimizer) || 
                     (e.edge_type == EdgeType::kAutogradFwdToRecompute)) 
                {
                    graph->add_edge(
                            graph->node_by_id(e.producer_node_id),
                            nop,
                            e.producer_output_port_id,
                            0,
                            e.edge_type);
                }
            }

            op_to_chip_id_assignment[nop->name()] = chip_ids.at(pipeline_index - 1);
            pipeline_index_to_source[pipeline_index] = nop;
        }
        Node* source_node = node;
        for (size_t user_index = 1; user_index < users.size(); ++user_index)
        {

            Node* user = users.at(user_index);
            std::uint32_t remote_chip_id = op_to_chip_id_assignment.at(user->name());
            int remote_chip_pipeline_index = chip_id_to_pipeline_index.at(remote_chip_id);
            if (remote_chip_pipeline_index == start_pipeline_index) {continue;}

            Node* new_source = pipeline_index_to_source.at(remote_chip_pipeline_index);

            for (auto old_edge : graph->get_edges(source_node, user)) {
                if (old_edge.edge_type == graphlib::EdgeType::kData) {
                    graphlib::Edge new_edge=
                        Edge(
                            new_source->id(),
                            old_edge.producer_output_port_id,
                            user->id(),
                            old_edge.consumer_input_port_id,
                            graphlib::EdgeType::kData);


                    graph->add_edge(new_edge);
                    graph->copy_edge_attributes(old_edge, new_edge);
                    graph->remove_edge(old_edge);
                }
            }
        }

    }
}

std::vector<std::vector<std::string>> update_epoch_breaks_for_partial_datacopy(
    graphlib::Graph *graph,
    std::vector<std::vector<std::string>> const &op_names_to_epoch_break)
{
    std::vector<std::vector<std::string>> updated_op_names_to_epoch_break = op_names_to_epoch_break;
    for (auto node : graph->nodes())
    {
        auto partial_datacopy_edges = graph->operand_edges(node, [](Edge e) { return e.edge_type == graphlib::EdgeType::kPartialDataCopy; });
        if (partial_datacopy_edges.empty())
            continue;

        // All consumers of partial_datacopy user must be on a new epoch
        for (auto user_node : graph->data_users(node))
        {
            updated_op_names_to_epoch_break.push_back({user_node->name()});
        }
    }
    return updated_op_names_to_epoch_break;
}
std::pair<placer::OpToChipIdAssignment, std::vector<std::vector<std::string>>> update_config_for_fractured_ops(
    const placer::ChipPlacerConfig& config,
    std::vector<std::vector<std::string>> const &op_names_to_epoch_break,
    const std::vector<std::string>& scheduled_ops,
    const placer::OpToChipIdAssignment& op_to_chip_id_assignment)
{
    placer::OpToChipIdAssignment updated_op_to_chip_id_assignment = op_to_chip_id_assignment;
    std::vector<std::vector<std::string>> updated_op_names_to_epoch_break = op_names_to_epoch_break;

    if (config.arch_name == "grayskull")
    {
        log_fatal("Multichip fracturing is not supported for grayskull architecture");
    }
    else
    {
        // For Wormhole, we need to add the fractured ops to the epoch break list
        // so that placer knows to assign each fractured op to its designated chip and
        // guarantee it will be the only op
        for (uint32_t i = 0; i < scheduled_ops.size(); ++i)
        {
            if (auto it = config.fracture_chip_id_assignments.find(scheduled_ops[i]); it != config.fracture_chip_id_assignments.end())
            {
                log_debug(LogGraphCompiler, "Assigning fractured op: {} to chip_id: {}:", scheduled_ops[i], it->second);
                updated_op_to_chip_id_assignment[scheduled_ops[i]] = it->second;
            }
        }
    }
    return {updated_op_to_chip_id_assignment, updated_op_names_to_epoch_break};
}

void insert_nops_forking_to_remote_devices(
    graphlib::Graph *graph,
    const std::vector<std::uint32_t> &chip_ids,
    placer::OpToChipIdAssignment &op_to_chip_id_assignment,
    std::string const &postfix)
{
    std::unordered_map<uint32_t, uint32_t> chip_id_to_pipeline_index;
    for (size_t pipeline_index = 0; pipeline_index < chip_ids.size(); ++pipeline_index) {
        chip_id_to_pipeline_index[chip_ids[pipeline_index]] = pipeline_index;
    }


    for (graphlib::Node *node : graphlib::topological_sort(*graph))
    {
        // There are three cases we need to check for:
        // 1) source input node is a constant -> we will default to replication
        // 2) producer feeding a single consumer with a chip-hop-jistance > 1
        // 3) op forking to multiple remote devices
        if (graph->data_users(node).size() == 0)  { continue; }

        std::map<string, std::uint32_t> user_remote_device_ids;
        std::uint32_t min_chip_id_found = UINT_MAX;
        std::uint32_t max_chip_id_found = 0;
        for (Node *user : graph->data_users(node)) {
            if (op_to_chip_id_assignment.find(user->name()) != op_to_chip_id_assignment.end()) {
                std::uint32_t user_chip_id = op_to_chip_id_assignment.at(user->name());
                user_remote_device_ids.emplace(user->name(), user_chip_id);
                min_chip_id_found = std::min(min_chip_id_found, user_chip_id);
                max_chip_id_found = std::max(max_chip_id_found, user_chip_id);
            }
        }

        // NB: this is to correspond to the fwd/bwd data-flow across the chips for grayskull pipelining
        std::uint32_t source_chip_id = op_to_chip_id_assignment.find(node->name()) != op_to_chip_id_assignment.end()
                                ? op_to_chip_id_assignment.at(node->name())
                                : node->get_epoch_type() == graphlib::NodeEpochType::Backward ? max_chip_id_found : min_chip_id_found;
        std::uint32_t source_pipeline_index = chip_id_to_pipeline_index.at(source_chip_id);

        int num_remote_consumers_with_more_one_hop = 0;
        int num_remote_consumers = 0;
        bool is_fork = (graph->data_users(node).size() > 1);

        for (Node *user : graph->data_users(node)) {
            if (auto it = op_to_chip_id_assignment.find(user->name()); it != op_to_chip_id_assignment.end()) {
                std::uint32_t dest_pipeline_index = chip_id_to_pipeline_index.at(it->second);
                if (dest_pipeline_index - source_pipeline_index > 1) {
                    num_remote_consumers_with_more_one_hop += 1;
                }
                if (source_pipeline_index != dest_pipeline_index) {
                    num_remote_consumers += 1;
                }
            }
        }

        if (node->node_type() == graphlib::NodeType::kInput and node->as<graphlib::InputNode>()->is_constant())
        {
            // case 1:
            std::vector<Edge> user_data_edges = graph->user_data_edges(node);

            for (size_t user_edge_index = 0; user_edge_index < user_data_edges.size(); ++user_edge_index) {
                const Edge& user_edge = user_data_edges[user_edge_index];

                Node* user = graph->node_by_id(user_edge.consumer_node_id);
                if (auto it = op_to_chip_id_assignment.find(user->name()); it != op_to_chip_id_assignment.end() and it->second != source_chip_id) {

                    auto cloned_constant_up = node->clone();
                    cloned_constant_up->set_name(node->name() + "_" + std::to_string(user_edge_index) + postfix);
                    Node* cloned_constant = graph->add_node(
                        std::move(cloned_constant_up), graph->get_subgraph_id_for_node(node->id()));

                    graphlib::Edge cloned_constant_to_remote_consumer =
                        Edge(
                            cloned_constant->id(),
                            0,
                            user->id(),
                            user_edge.consumer_input_port_id,
                            graphlib::EdgeType::kData);

                    graph->add_edge(cloned_constant_to_remote_consumer);
                    graph->copy_edge_attributes(user_edge, cloned_constant_to_remote_consumer);
                    graph->remove_edge(user_edge);
                }
            }
        }
        else if (num_remote_consumers_with_more_one_hop > 0 and not is_fork) {
            // case 2:
            Node* source_node = node;
            Node* user = graph->data_users(node).at(0);
            std::uint32_t remote_chip_id = op_to_chip_id_assignment.at(user->name());
            std::uint32_t start_pipeline_index = chip_id_to_pipeline_index.at(source_chip_id) + 1;
            std::uint32_t end_pipeline_index = chip_id_to_pipeline_index.at(remote_chip_id);

            for (std::uint32_t pipeline_index = start_pipeline_index; pipeline_index < end_pipeline_index; ++pipeline_index)
            {
                graphlib::BudaOpNode *nop = graph->add_node(graphlib::create_node<graphlib::BudaOpNode>(
                    node->name() + "_chip_to_chip_nop_" + std::to_string(pipeline_index) + postfix, "nop"),
                    graph->get_subgraph_id_for_node(node->id()));
                graph->copy_node_attributes(source_node, nop);
                op_to_chip_id_assignment[nop->name()] = chip_ids.at(pipeline_index);

                for (auto user_edge : graph->get_edges(source_node, user)) {
                    if (user_edge.edge_type == graphlib::EdgeType::kData) {
                        graphlib::insert_node_on_edge(graph, user_edge, nop);
                    }
                }
                source_node = nop;
                source_chip_id = chip_ids.at(pipeline_index);
            }

        }
        else if (is_fork and num_remote_consumers > 0)
        {
            insert_nop_fork(
                graph,
                node,
                chip_ids,
                chip_id_to_pipeline_index,
                user_remote_device_ids,
                op_to_chip_id_assignment,
                postfix);
        }
    }
}

void calculate_ublock_order(graphlib::Graph *graph) {
    auto eval_module = py::module_::import("pybuda.op.eval.buda");
    py::function pybuda_input_ublock_order = eval_module.attr("get_f_pybuda_input_ublock_order");

    auto relate = [pybuda_input_ublock_order](
                      graphlib::Graph const *graph,
                      Edge root,
                      graphlib::UBlockOrder ublock_order,
                      std::map<graphlib::UBlockOrder, std::set<Edge>> &equivalence_sets,
                      graphlib::NodeEpochType epoch_type)
    {
        std::vector<std::pair<Edge, graphlib::UBlockOrder>> needs_visit = {std::make_pair(root, ublock_order)};
        while (!needs_visit.empty())
        {
            auto [edge, order] = needs_visit.back();
            needs_visit.pop_back();
            if (
                equivalence_sets.at(graphlib::UBlockOrder::R).find(edge) != equivalence_sets.at(graphlib::UBlockOrder::R).end()
                or equivalence_sets.at(graphlib::UBlockOrder::C).find(edge) != equivalence_sets.at(graphlib::UBlockOrder::C).end()
            )
                continue;

            equivalence_sets.at(order).insert(edge);

            Node *node = graph->node_by_id(edge.producer_node_id);
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
            if (not op)
                continue;

            if (op->get_epoch_type() != epoch_type)
                continue;

            std::vector<Edge> operands = graph->operand_data_edges(node);
            auto maybe_input_ublock_order = pybuda_input_ublock_order(op->op_type(), operands.size())
                                                .cast<std::optional<std::vector<graphlib::UBlockOrder>>>();

            bool no_input_ublock_reqs = not maybe_input_ublock_order;
            if (no_input_ublock_reqs)
            {
                for (auto operand : operands) {
                    auto edge_attrs = graph->get_edge_attributes(edge);
                    if (edge_attrs->has_tm("transpose")) {
                        needs_visit.insert(needs_visit.end(),
                            {std::make_pair(operand, order == graphlib::UBlockOrder::R ? graphlib::UBlockOrder::C : graphlib::UBlockOrder::R)}
                        );

                    } else if (edge_attrs->has_tm("vslice") and edge_attrs->get_tms().size() == 1) {
                        needs_visit.insert(needs_visit.end(),
                            {std::make_pair(operand, graphlib::UBlockOrder::R)}
                        );
                    } else if (edge_attrs->has_tm("hslice") and edge_attrs->get_tms().size() == 1) {
                        needs_visit.insert(needs_visit.end(),
                            {std::make_pair(operand, graphlib::UBlockOrder::C)}
                        );
                    } else {
                        needs_visit.insert(needs_visit.end(), {std::make_pair(operand, order) });
                    }
                }
            }
        }
    };


    std::map<graphlib::UBlockOrder, std::set<Edge>> order_equivalence = {
        {graphlib::UBlockOrder::R, std::set<Edge>{}},
        {graphlib::UBlockOrder::C, std::set<Edge>{}},
    };
    for (Node *node : graph->nodes())
    {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
        if (not op)
            continue;

        std::vector<Edge> operands = graph->operand_data_edges(node);
        auto maybe_input_ublock_order = pybuda_input_ublock_order(op->op_type(), operands.size())
                                            .cast<std::optional<std::vector<graphlib::UBlockOrder>>>();

        if (not maybe_input_ublock_order)
            continue;

        std::vector<graphlib::UBlockOrder> const &input_ublock_order = *maybe_input_ublock_order;

        if (op->is_matmul())
            // Matmul could have 2 or 3 inputs, depending on whether the bias has been fused or not
            TT_ASSERT(input_ublock_order.size() >= operands.size());
        else
            TT_ASSERT(input_ublock_order.size() == operands.size());
        for (std::size_t i = 0; i < operands.size(); ++i)
        {
            relate(graph, operands[i], input_ublock_order[i], order_equivalence, op->get_epoch_type());
        }
    }

    // Set C first so that R wins on overlapping equivalencies (see test_balancer.py::test_ublock_order_fork)
    for (auto ublock_order : {graphlib::UBlockOrder::C, graphlib::UBlockOrder::R})
    {
        auto const &equivalence = order_equivalence.at(ublock_order);
        log_trace(LogGraphCompiler, "Resolve {}:", ublock_order);
        for (auto const &edge : equivalence)
        {
            Node *producer = graph->node_by_id(edge.producer_node_id);
            Node *node = graph->node_by_id(edge.consumer_node_id);
            log_trace(LogGraphCompiler, "  {} -> {}", producer->name(), node->name());
            auto edge_attrs = graph->get_edge_attributes(edge);
            edge_attrs->set_ublock_order(ublock_order);
        }
    }

    // Set order on output queues
    for (Node *node : graphlib::topological_sort(*graph))
    {
        if (not dynamic_cast<graphlib::OpNode *>(node))
            continue;

        // operands.back should always give the correct ublock order for an op, matmul / eltwise / etc.
        std::vector<Edge> operands = graph->operand_data_edges(node);
        graphlib::UBlockOrder op_ublock_order = get_output_ublock_order(graph, node);

        // Output queues always inherit ublock order of producer op
        std::vector<Edge> users = graph->user_data_edges(node);
        for (Edge edge : users)
        {
            if (dynamic_cast<graphlib::QueueNode *>(graph->node_by_id(edge.consumer_node_id)))
            {
                auto edge_attrs = graph->get_edge_attributes(edge);
                edge_attrs->set_ublock_order(op_ublock_order);
            }
        }
    }

    // Set order on partial datacopy edges consumers
    for (Node * node : graph->nodes())
    {
        auto is_partial_datacopy_edge = [](Edge e) {
            return (e.edge_type == graphlib::EdgeType::kPartialDataCopy);
        };
        std::vector<graphlib::Edge> partial_datacopy_edges = graph->operand_edges(node, is_partial_datacopy_edge);

        if (partial_datacopy_edges.empty())
            continue;
        
        // Fetch producers' edges because they are the kData edges that have ublock order
        std::vector<graphlib::Edge> producer_edges;
        for (auto edge : partial_datacopy_edges) {
            auto output_node = graph->node_by_id(edge.producer_node_id);
            TT_ASSERT(graph->operand_edges(output_node).size() == 1, "Output node should only have 1 producer");
            producer_edges.push_back(
                graph->operand_edges(output_node).front()
            );
        }

        // partial datacopy edges should always have the same ublock order from all producers
        auto ublock_order = graph->get_edge_attributes(producer_edges[0])->get_ublock_order();

        for (auto edge : producer_edges) {
            auto edge_attrs = graph->get_edge_attributes(edge);
            if (edge_attrs->get_ublock_order() != ublock_order) {
                // Insert NOP to transpose ublock order
                auto producer_node = graph->node_by_id(edge.producer_node_id);
                auto nop = graph->add_node(graphlib::create_node<graphlib::BudaOpNode>(
                    producer_node->name() + "_partial_datacopy_nop", "nop"),
                    graph->get_subgraph_id_for_node(producer_node->id()));
                
                auto [new_edge0, new_edge1] = graphlib::insert_node_on_edge(graph, edge, nop);
                graph->get_edge_attributes(new_edge1)->set_ublock_order(ublock_order);
                graphlib::calculate_and_set_node_shape(graph, nop);
            }
        }
    }

    // Set linking on subgraphs
    for (Node * node : graph->nodes())
    {
        auto is_subgraph_link_edge = [](Edge e) {
            return (e.edge_type == graphlib::EdgeType::kSubgraphLink);
        };
        std::vector<graphlib::Edge> subgraph_link_edge = graph->operand_edges(node, is_subgraph_link_edge);

        if (subgraph_link_edge.empty())
            continue;
        
        // Fetch producers' edges because they are the kData edges that have ublock order
        auto output_node = graph->node_by_id(subgraph_link_edge[0].producer_node_id);
        TT_ASSERT(graph->operand_edges(output_node).size() == 1, "Output node should only have 1 producer");
        auto producer_ublock_order = graph->get_edge_attributes(graph->operand_edges(output_node)[0])->get_ublock_order();

        // Consumer should have the same ublock order as producer
        for (auto e : graph->user_data_edges(node)) {
            auto edge_attrs = graph->get_edge_attributes(e);
            if (edge_attrs->get_ublock_order() != producer_ublock_order) {
                // Insert NOP to transpose ublock order
                auto nop = graph->add_node(graphlib::create_node<graphlib::BudaOpNode>(
                    node->name() + "_subgraph_link_nop", "nop"),
                    graph->get_subgraph_id_for_node(node->id()));
                auto original_ublock_order = edge_attrs->get_ublock_order();
                
                auto [new_edge0, new_edge1] = graphlib::insert_node_on_edge(graph, e, nop);
                graph->get_edge_attributes(new_edge0)->set_ublock_order(producer_ublock_order);
                graph->get_edge_attributes(new_edge1)->set_ublock_order(original_ublock_order);
                graphlib::calculate_and_set_node_shape(graph, nop);
            }
        }
    }



#ifdef DEBUG
    //
    // Assert that
    // if input_ublock_order is defined
    //     that we actually picked the correct ublock order
    // if input_ublock_order is not defined
    //     We are eltwise so all ublock orders must match
    //
    for (Node *node : graphlib::topological_sort(*graph))
    {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
        if (not op)
            continue;

        std::vector<Edge> operands = graph->operand_data_edges(node);
        auto maybe_input_ublock_order = pybuda_input_ublock_order(op->op_type(), operands.size())
                                            .cast<std::optional<std::vector<graphlib::UBlockOrder>>>();
        std::vector<graphlib::UBlockOrder> const required_input_ublock_order =
            maybe_input_ublock_order
                ? *maybe_input_ublock_order
                : std::vector<graphlib::UBlockOrder>(
                      operands.size(), graph->get_edge_attributes(operands.front())->get_ublock_order());
        for (std::size_t i = 0; i < operands.size(); ++i)
        {
            TT_ASSERT(graph->get_edge_attributes(operands[i])->get_ublock_order() == required_input_ublock_order[i], "UBlock order mismatch on {} and {}",
                graph->node_by_id(operands[i].producer_node_id)->name(), node->name());
        }
    }
#endif
}

void split_broadcasts(Graph* graph)
{
    std::function<bool(graphlib::Edge)> has_broadcast_filter = [&graph](graphlib::Edge edge)
    {
        auto edge_attributes = graph->get_edge_attributes(edge);
        std::vector<graphlib::OpType> &tms = edge_attributes->get_tms();
        return std::any_of(tms.begin(), tms.end(), [](graphlib::OpType& ot) { return ot.op == "broadcast"; });
    };

    // Create a list of edges with broadcast TMs eligible for splitting
    std::vector<Edge> edge_candidates;
    for (Node *node : graph->nodes())
    {
        if (node->node_type() != NodeType::kBudaOp)
        {
            continue;
        }

        for (auto &edge : graph->operand_data_edges(node, has_broadcast_filter))
        {
            std::vector<graphlib::OpType> &tms = graph->get_edge_attributes(edge)->get_tms();

            int cnt_broadcast_tms = 0;
            for (graphlib::OpType& opType : tms)
            {
                if (opType.op == "broadcast")
                {
                    cnt_broadcast_tms++;
                    if (cnt_broadcast_tms > 1)
                    {
                        edge_candidates.push_back(edge);
                        break;
                    }
                }
            }
        }
    }

    // For each edge in the list, iteratively keep adding nops and splitting TM attributes
    for (Edge& edge_candidate : edge_candidates)
    {
        Node *consumer = graph->node_by_id(edge_candidate.consumer_node_id);
        int curr_producer_id = edge_candidate.producer_node_id;
        bool found = true;
        while (found)
        {
            found = false;

            std::vector<Edge> edges = graph->operand_data_edges(
                consumer, [&curr_producer_id](Edge e) { return e.producer_node_id == curr_producer_id; });
            TT_ASSERT(edges.size() == 1);  // Will this break multiply when squaring? If so, remove...
            Edge edge = edges.front();

            std::vector<graphlib::OpType> &tms = graph->get_edge_attributes(edge)->get_tms();

            // Sort tms so that t-dim broadcast stays on original producer -> first inserted nop, since nops can't
            // buffer t-dim
            int first_non_t_broadcast_idx = -1;
            for (size_t i = 0; i < tms.size(); i++)
            {
                if (tms[i].op == "broadcast")
                {
                    int broadcast_dim = std::get<int>(tms[i].attr[0]);
                    broadcast_dim = broadcast_dim < 0? broadcast_dim + graphlib::Shape::BUDA_DIM_COUNT: broadcast_dim;
                    if (broadcast_dim != 1 && (first_non_t_broadcast_idx == -1))
                    {
                        // non t-dim broadcast
                        first_non_t_broadcast_idx = i;
                    }
                    else
                    {
                        // t-dim broadcast
                        if (first_non_t_broadcast_idx == -1)
                        {
                            break;
                        }
                        std::swap(tms[first_non_t_broadcast_idx], tms[i]);
                    }
                }
            }

            int cnt_broadcast_tms = 0;
            int prev_broadcast_tm_idx = -1;
            for (size_t idx = 0; idx < tms.size(); idx++)
            {
                if (tms[idx].op == "broadcast")
                {
                    cnt_broadcast_tms++;

                    if (cnt_broadcast_tms > 1)
                    {
                        auto [inserted, _, __]  = split_broadcast(graph, consumer, edge, prev_broadcast_tm_idx);
                        curr_producer_id = inserted->id();

                        found = true;
                        break;
                    }

                    // Update prev idx
                    prev_broadcast_tm_idx = idx;
                }
            }
        }
    }
}

std::tuple<Node *, Edge, Edge> split_broadcast(Graph *graph, Node *consumer, Edge &edge, int idx_of_broadcast_tm)
{
    std::vector<graphlib::OpType> tms = graph->get_edge_attributes(edge)->get_tms();
    TT_ASSERT(tms[idx_of_broadcast_tm].op == "broadcast");

    // Add a NOP on the edge
    int broadcast_dim = std::get<int>(tms[idx_of_broadcast_tm].attr[0]);

    std::uint32_t index = 0;
    std::string nop_name;
    while (true) {
        nop_name = graph->node_by_id(edge.producer_node_id)->name() + "_splt_brcst_" + std::to_string(broadcast_dim) + "_" + std::to_string(index);
        if (graph->has_node_with_name(nop_name)) {
            index ++;
            continue;
        }
        break;
    }
    graphlib::BudaOpNode *nop =
        graph->add_node(graphlib::create_node<graphlib::BudaOpNode>(nop_name, "nop"),
        graph->get_subgraph_id_for_node(consumer->id()));
    nop->copy_parent_op_attributes(consumer->as<graphlib::BudaOpNode>());
    auto [new_edge0, new_edge1] = graphlib::insert_node_on_edge(graph, edge, nop);

    // new_edge0 gets all tms up to and including specified broadcast op (at idx_of_broadcast_tm), edge1 gets the rest
    graph->get_edge_attributes(new_edge0)->set_tms(
        std::vector<graphlib::OpType>(tms.begin(), tms.begin() + idx_of_broadcast_tm + 1));
    graph->get_edge_attributes(new_edge1)->set_tms(
        std::vector<graphlib::OpType>(tms.begin() + idx_of_broadcast_tm + 1, tms.end()));

    // Recalculate the shape of the new nop node
    graphlib::calculate_and_set_node_shape(graph, nop);

    return std::make_tuple(nop, new_edge0, new_edge1);
}

void validate_buffering_queues(graphlib::Graph *graph) {
    for (graphlib::Node *node : graphlib::topological_sort(*graph))  {
        if (node->node_type() != graphlib::NodeType::kQueue) {
            continue;
        }
        if (node->as<graphlib::QueueNode>()->is_buffering()) {
            // verify that we're only buffering queues are inserted between ops
            for (Node *operand : graph->data_operands(node)) {
                if (operand->node_type() != graphlib::NodeType::kBudaOp) {
                    log_fatal("buffering queue: {} has operand: {} that is not a BudaOp", node->name(), operand->name());
                }
            }
            for (Node *user : graph->data_users(node)) {
                if (user->node_type() != graphlib::NodeType::kBudaOp) {
                    log_fatal("buffering queue: {} has user: {} that is not a BudaOp", node->name(), user->name());
                }
            }
        }
    }
}


void insert_partial_datacopy_tms(graphlib::Graph *graph) {

    auto is_partial_datacopy_edge = [](Edge e) {
        return (e.edge_type == graphlib::EdgeType::kPartialDataCopy);
    };
    for (graphlib::Node *node : graphlib::topological_sort(*graph))  {
        std::vector<graphlib::Edge> partial_datacopy_edges = graph->user_edges(node, is_partial_datacopy_edge);
        if (node->node_type() != graphlib::NodeType::kOutput or partial_datacopy_edges.empty()) {
            continue;
        }
        for (auto edge : partial_datacopy_edges)
        {
            auto input_node = graph->node_by_id(edge.consumer_node_id);
            // We don't need tms for full datacopy (i.e. past cache produced in single iteration)
            TT_ASSERT(input_node->shape().volume() % node->shape().volume() == 0, "Node volume {} and {} aren't multiples", input_node->name(), node->name());
            std::vector<graphlib::OpType> stack_tms;
            std::vector<graphlib::OpType> slice_tms;
            int v_stack = input_node->shape().canonical()[-2] / node->shape().canonical()[-2];
            if (v_stack > 1)
            {
                graphlib::OpType stack("vstack", {v_stack});
                stack_tms.push_back(stack);
                graphlib::ConstEvalGraph *consteval_graph = input_node->as<graphlib::InputNode>()->get_consteval_graph(graph, true, true);
                graphlib::OpType slice("vslice", {v_stack});
                slice_tms.push_back(slice);
                auto slice_node = graphlib::create_node<graphlib::PyOpNode>(input_node->name() + "_vslice", slice);
                graphlib::Shape shape = input_node->shape();
                shape[-2] /= v_stack;
                shape[-3] *= v_stack;
                slice_node->set_shape(shape);
                slice_node->set_epoch_type(input_node->get_epoch_type());
                slice_node->set_output_df(input_node->output_df());
                consteval_graph->promote_node(std::move(slice_node));
                input_node->set_shape(shape);
            }
            int h_stack = input_node->shape().canonical()[-1] / node->shape().canonical()[-1];
            if (h_stack > 1)
            {
                graphlib::OpType stack("hstack", {h_stack});
                stack_tms.push_back(stack);
                graphlib::ConstEvalGraph *consteval_graph = input_node->as<graphlib::InputNode>()->get_consteval_graph(graph, true, true);
                graphlib::OpType slice("hslice", {h_stack});
                slice_tms.push_back(slice);
                auto slice_node = graphlib::create_node<graphlib::PyOpNode>(input_node->name() + "_hslice", slice);
                graphlib::Shape shape = input_node->shape();
                shape[-1] /= h_stack;
                shape[-3] *= h_stack;
                slice_node->set_shape(shape);
                slice_node->set_epoch_type(input_node->get_epoch_type());
                slice_node->set_output_df(input_node->output_df());
                consteval_graph->promote_node(std::move(slice_node));
                input_node->set_shape(shape);
            }

            if (stack_tms.size() > 0)
            {
                for (auto consumer_edge : graph->user_data_edges(input_node))
                {
                    auto tms = graph->get_edge_attributes(consumer_edge)->get_tms();
                    tms.insert(tms.begin(), stack_tms.begin(), stack_tms.end());
                    graph->get_edge_attributes(consumer_edge)->set_tms(tms);
                }
                for (auto producer_edge : graph->operand_data_edges(input_node))
                {
                    if (producer_edge.edge_type != EdgeType::kDataLoopback)
                        continue;
                    auto producer = graph->node_by_id(producer_edge.producer_node_id);
                    graphlib::BudaOpNode *nop = graph->add_node(graphlib::create_node<graphlib::BudaOpNode>(
                                input_node->name() + "_tm_nop", "nop"), graph->get_subgraph_id_for_node(producer->id()));
                    graph->remove_edge(producer_edge);
                    graphlib::Edge data_edge = Edge(producer->id(), 0, nop->id(), 0, EdgeType::kData);
                    graph->add_edge(data_edge);
                    auto tms = graph->get_edge_attributes(data_edge)->get_tms();
                    tms.insert(tms.begin(), slice_tms.begin(), slice_tms.end());
                    graph->get_edge_attributes(data_edge)->set_tms(tms);
                    graph->add_edge(Edge(nop->id(), 0, input_node->id(), 0, EdgeType::kDataLoopback));
                }
            }
        }
    }
}

void constant_pre_broadcast(Graph *graph)
{
    auto promote = []
    (graphlib::Graph* graph, graphlib::ConstantInputNode *input, std::vector<graphlib::OpType> const &broadcasts)
    {
        graphlib::ConstEvalGraph *consteval_graph = input->get_consteval_graph(graph, true, true);

        graphlib::Shape current_shape = input->shape();
        for (graphlib::OpType op_type : broadcasts)
        {
            std::vector<graphlib::Shape> input_shapes = {current_shape};
            auto [shape, bcast_dims] = get_op_shape(op_type, input_shapes, true);
            op_type.op = "repeat_dim";
            auto broadcast = graphlib::create_node<graphlib::PyOpNode>(
                "broadcast" + std::to_string(std::get<int>(op_type.attr[0])) + "_" + input->name(), op_type);
            broadcast->set_shape(shape);
            broadcast->set_epoch_type(input->get_epoch_type());
            broadcast->set_output_df(input->output_df());
            consteval_graph->promote_node(std::move(broadcast));
            current_shape = shape;
        }
    };

    for (graphlib::Node *node : graph->nodes())
    {
        graphlib::ConstantInputNode *potential_input = dynamic_cast<graphlib::ConstantInputNode *>(node);
        if (not potential_input)
            continue;

        // if there are multiple users they each might need a different set of broadcasts
        // trivially just clone mutliple users
        {
            bool has_broadcast = false;
            std::vector<graphlib::Edge> users = graph->user_data_edges(node);
            for (graphlib::Edge const &edge : graph->user_data_edges(node))
            {
                auto edge_attr = graph->get_edge_attributes(edge);
                for (auto const &tm : edge_attr->get_tms()) has_broadcast |= (tm.op == "broadcast");
            }

            if (not has_broadcast)
                continue;

            for (int i = 1; i < (int)users.size(); ++i)
            {
                graphlib::Edge edge = users[i];
                auto edge_attr = graph->get_edge_attributes(edge);

                auto input_clone = potential_input->clone();
                input_clone->set_name(potential_input->name() + "_clone" + std::to_string(i));
                auto graph_input = graph->add_node<graphlib::Node>(
                    std::move(input_clone), graph->get_subgraph_id_for_node(edge.consumer_node_id));
                graph->remove_edge(edge);
                graph->add_edge(
                    graphlib::Edge(
                        graph_input->id(), 0, edge.consumer_node_id, edge.consumer_input_port_id, edge.edge_type),
                    edge_attr);
            }
        }

        for (graphlib::Edge const &edge : graph->user_data_edges(node))
        {
            graphlib::ConstantInputNode *input =
                dynamic_cast<graphlib::ConstantInputNode *>(graph->node_by_id(edge.producer_node_id));
            auto edge_attr = graph->get_edge_attributes(edge);
            std::vector<graphlib::OpType> &tms = edge_attr->get_tms();

            int index = 0;
            std::vector<int> indices;
            std::vector<graphlib::OpType> broadcasts;
            for (auto const &tm : tms)
            {
                if (tm.op == "broadcast")
                {
                    indices.push_back(index);
                    broadcasts.push_back(tm);
                }
                ++index;
            }

            if (broadcasts.empty())
                continue;

            for (auto iter = indices.rbegin(); iter != indices.rend(); ++iter)
                tms.erase(tms.begin() + *iter);

            promote(graph, input, broadcasts);
        }
    }
}

// Convert PyBuda graph to Buda graph
std::unique_ptr<Graph> lower_to_buda_ops(Graph *graph)
{

    auto new_graph = std::make_unique<Graph>(graphlib::IRLevel::IR_BUDA, graph->name());

    new_graph->set_microbatch(graph->get_microbatch());

    // Mapping of old nodes to new ones. Where the old node maps to multiple new ones,
    // the output node is recorded as "new", because it will be used as operand into
    // future new nodes.
    std::unordered_map<Node *, Node *> old_to_new;

    for (Node *node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() == NodeType::kInput)
            continue;

        if (node->node_type() == NodeType::kPyOp) {
            // Ops get converted
            log_trace(LogGraphCompiler, "Lowering node: {}", node->name());
            lower_node(LoweringContext(graph, new_graph.get(), node->as<graphlib::PyOpNode>(), old_to_new));
        } else {
            log_trace(LogGraphCompiler, "Lowering queue: {}", node->name());
            lower_queue(graph, new_graph.get(), node, old_to_new);
        }
    }

    new_graph->copy_module_inputs(graph, old_to_new);
    new_graph->copy_module_outputs(graph, old_to_new);
    new_graph->copy_module_targets(graph, old_to_new);

    // Copy over loopback edges now that all the nodes are available, as well as "gradient_op" flag
    std::vector<graphlib::NodeId> ordered_old_nodes_to_process;
    for (auto &[old_node, new_node] : old_to_new)
    {
        ordered_old_nodes_to_process.emplace_back(old_node->id());
    }
    std::sort(ordered_old_nodes_to_process.begin(), ordered_old_nodes_to_process.end());
    for (graphlib::NodeId old_node_id : ordered_old_nodes_to_process)
    {
        Node* old_node = graph->node_by_id(old_node_id);
        Node* new_node = old_to_new.at(old_node);

        new_node->set_pybuda_id(old_node->id());
        copy_operand_edges_to_new_graph(graph, new_graph.get(), old_node, new_node, old_to_new, true, true);

        if (old_node->node_type() == NodeType::kPyOp and new_node->node_type() == NodeType::kPyOp)
            new_node->as<graphlib::BudaOpNode>()->set_gradient_op(old_node->as<graphlib::PyOpNode>()->is_gradient_op());
    }

    return new_graph;
}


static Node* create_recompute_op(graphlib::Graph* graph, Node* fwd_node, std::unordered_map<Node*, Node*>& fwd_to_recompute)
{
    graphlib::BudaOpNode *recompute_node = graph->add_node(
        graphlib::create_node<graphlib::BudaOpNode>(fwd_node->name() + "_recompute", fwd_node->as<graphlib::OpNode>()->op_type()),
        graph->get_subgraph_id_for_node(fwd_node->id()));

    recompute_node->set_shape(fwd_node->shape());
    recompute_node->set_output_df(fwd_node->output_df());
    recompute_node->set_intermediate_df(fwd_node->as<graphlib::BudaOpNode>()->intermediate_df());
    recompute_node->set_accumulate_df(fwd_node->as<graphlib::BudaOpNode>()->accumulate_df());
    recompute_node->set_epoch_type(graphlib::NodeEpochType::Backward);

    if (fwd_node->as<graphlib::BudaOpNode>()->is_fused_op())
    {
        std::shared_ptr<FusedOp> fused_op_clone = fwd_node->as<graphlib::BudaOpNode>()->get_fused_op()->clone(recompute_node);
        recompute_node->set_fused_op(fused_op_clone);
    }

    std::unordered_map<graphlib::NodeId, graphlib::NodeId> producer_remap;

    // For sparse matmul we much duplicate the constant inputs, these need to be unique per instance
    if (fwd_node->as<graphlib::BudaOpNode>()->is_sparse_matmul())
    {
        auto operands = graph->data_operands(fwd_node);
        TT_ASSERT(operands.size() == 3);
        auto *in0 = operands[0];
        auto *in2 = operands[2];
        auto *recompute_in0 = graph->add_node(
            in0->clone(in0->name() + "_recompute"), graph->get_subgraph_id_for_node(fwd_node->id()));
        auto *recompute_in2 = graph->add_node(
            in2->clone(in2->name() + "_recompute"), graph->get_subgraph_id_for_node(fwd_node->id()));
        recompute_in0->set_epoch_type(graphlib::NodeEpochType::Backward);
        recompute_in2->set_epoch_type(graphlib::NodeEpochType::Backward);
        producer_remap[in0->id()] = recompute_in0->id();
        producer_remap[in2->id()] = recompute_in2->id();
    }

    fwd_to_recompute[fwd_node] = recompute_node;

    // Hook up Recompute Node to its operands
    for (const Edge &edge : graph->operand_data_edges(fwd_node))
    {
        Node* fwd_operand = graph->node_by_id(edge.producer_node_id);
        bool same_epoch =
            (fwd_operand->get_epoch_type() == graphlib::NodeEpochType::Forward and
             fwd_operand->node_type() != graphlib::NodeType::kInput);
        graphlib::NodeId producer_id = same_epoch ? fwd_to_recompute.at(fwd_operand)->id() : fwd_operand->id();
        if (producer_remap.find(producer_id) != producer_remap.end())
            producer_id = producer_remap.at(producer_id);
        Edge recompute_edge = Edge(
            producer_id,
            edge.producer_output_port_id,
            recompute_node->id(),
            edge.consumer_input_port_id,
            edge.edge_type);
        graph->add_edge(recompute_edge);
        auto fwd_edge_attr = graph->get_edge_attributes(edge);
        auto recompute_edge_attr = graph->get_edge_attributes(recompute_edge);
        recompute_edge_attr->copy_from(*fwd_edge_attr);
    }
    return recompute_node;
}


static void add_edges_for_recompute_op(
    graphlib::Graph* graph, const graphlib::Edge& fwd_to_bwd_edge, std::unordered_map<Node*, Node*>& fwd_to_recompute)
{
    Node* fwd_node = graph->node_by_id(fwd_to_bwd_edge.producer_node_id);
    if (fwd_to_recompute.find(fwd_node) == fwd_to_recompute.end())
    {
        return;
    }
    Node* recompute_node = fwd_to_recompute.at(fwd_node);


    // Replace the original FWD->BWD edge with RECOMPUTE->BWD instead.
    graph->add_edge(Edge(
        recompute_node->id(),
        fwd_to_bwd_edge.producer_output_port_id,
        fwd_to_bwd_edge.consumer_node_id,
        fwd_to_bwd_edge.consumer_input_port_id,
        fwd_to_bwd_edge.edge_type),
        graph->get_edge_attributes(fwd_to_bwd_edge)
    );
    graph->remove_edge(fwd_to_bwd_edge);
}

void insert_recompute_ops(graphlib::Graph *graph)
{
    std::unordered_map<Node*, Node*> forward_to_recompute;
    std::vector<Node*> topo_order = graphlib::topological_sort(*graph);

    // We loop through only the backward nodes and look for operand nodes that are marked
    // as FWD. Each instance is an opportunity to perform a potential recompute on the original
    // FWD op.
    // TODO(jchu): extend this pretty easily to support configurable op-specific recompute.
    // rather than an all-or-nothing sort-of deal.
    std::deque<std::string> recompute_node_names;

    for (Node* node : topo_order)
    {
        if (node->get_epoch_type() == graphlib::NodeEpochType::Forward and
            (node->node_type() == graphlib::NodeType::kPyOp or node->node_type() == graphlib::NodeType::kBudaOp))
        {
            Node *recompute_node = create_recompute_op(graph, node, forward_to_recompute);
            recompute_node_names.push_back(recompute_node->name());
        }
    }

    for (Node* bwd_node : topo_order)
    {
        if (bwd_node->get_epoch_type() != graphlib::NodeEpochType::Backward)
        {
            continue;
        }

        std::vector<Edge> bwd_operand_data_edges = graph->operand_data_edges(bwd_node);
        for (const Edge& operand_edge : bwd_operand_data_edges)
        {
            add_edges_for_recompute_op(graph, operand_edge, forward_to_recompute);
        }
    }

    std::unordered_map<Node*, std::string> forward_to_recompute_name;
    for (const auto[fwd_node, recompute_node] : forward_to_recompute)
    {
        forward_to_recompute_name[fwd_node] = recompute_node->name();
    }

    while (not recompute_node_names.empty()) {
        std::string recompute_node_name = recompute_node_names.front();
        recompute_node_names.pop_front();

        if (graph->has_node_with_name(recompute_node_name)) {
            Node* recompute_node = graph->get_node_by_name(recompute_node_name);
            if (graph->num_users(recompute_node->id()) == 0)
            {
                for (Node* operand : graph->data_operands(recompute_node)) {
                    recompute_node_names.push_back(operand->name());
                }
                graph->remove_node(recompute_node);
            }
        }
    }

    for (const auto &[fwd_node, recompute_node_name] : forward_to_recompute_name) {
        if (graph->has_node_with_name(recompute_node_name)) {
            graph->add_edge(fwd_node, graph->get_node_by_name(recompute_node_name), graphlib::EdgeType::kAutogradFwdToRecompute);
        }
    }
    return;
}

void insert_user_defined_queues(
    graphlib::Graph *graph, const std::vector<std::tuple<std::string, std::string, int>> &insert_queues)
{
    int id = 0;
    for (auto const &[producer_name, consumer_name, input_port_id] : insert_queues)
    {
        log_trace(LogGraphCompiler, "insert_queues: {} -> {}[{}]", producer_name, consumer_name, input_port_id);
        graphlib::Node *producer = graph->get_node_by_name(producer_name);
        TT_LOG_ASSERT(producer, "Unable to find producer in insert_queues: {}", producer_name);
        graphlib::Node *consumer = graph->get_node_by_name(consumer_name);
        TT_LOG_ASSERT(consumer, "Unable to find consumer in insert_queues: {}", consumer_name);
        auto edges = graph->get_edges(producer, consumer);
        int capture_input_port_id = input_port_id;
        auto match = std::find_if(
            edges.begin(),
            edges.end(),
            [capture_input_port_id](graphlib::Edge const &e) {
                return e.edge_type == graphlib::EdgeType::kData and
                       (int) e.consumer_input_port_id == capture_input_port_id;
            });
        TT_LOG_ASSERT(
            match != edges.end(),
            "Unable to find specified edge in insert_queues: {} -> {}[{}]",
            producer_name,
            consumer_name,
            input_port_id);

        auto name = "insert_queue" + std::to_string(id++);
        // adding queue that has num_entries equal to microbatch size. This is enough to guarantee that queue will never
        // be full. microbatch size is the highest num_entires one buffering queue will ever need, since it is inside
        // one epoch
        auto *q = graph->add_node(
            graphlib::create_node<graphlib::BufferingQueueNode>(name, graph->get_microbatch() * 2),
            graph->get_subgraph_id_for_node(producer->id()));
        q->as<graphlib::TaggedNode>()->tag("inserted_queue");
        q->set_shape(producer->shape());
        q->set_output_df(producer->output_df());
        q->set_epoch_type(consumer->get_epoch_type());

        auto ublock_order = graph->get_edge_attributes(*match)->get_ublock_order();
        bool inherit_consumer_attrs = true;
        bool remove_edge = true;
        std::uint32_t consumer_index = 0;
        bool place_tms_on_outgoing = true;
        auto [producer_edge, consumer_edge] = insert_node_on_edge(
            graph, *match, q, inherit_consumer_attrs, remove_edge, consumer_index, place_tms_on_outgoing);
        graph->get_edge_attributes(producer_edge)->set_ublock_order(ublock_order);
    }
}

}  // namespace tt
