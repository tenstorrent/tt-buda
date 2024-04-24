// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/placer_buda_passes.hpp"

#include "balancer/balancer_cache_collection.hpp"
#include "balancer/policies/policies.hpp"
#include "passes/fracture.hpp"
#include "passes/padding_pass_placer.hpp"
#include "passes/passes_utils.hpp"
#include "passes/pre_placer_buda_passes.hpp"
#include "placer/lower_to_placer.hpp"
#include "placer/placer.hpp"
#include "reportify/reportify.hpp"
#include "utils/env.hpp"

namespace tt::passes
{

// Insert NOPs on queues that have a TM on their input.
//
static void fix_tms_on_queues(graphlib::Graph* graph)
{
    for (Node* n : graph->nodes_by_type(graphlib::NodeType::kQueue))
    {
        std::vector<Edge> edges = graph->operand_data_edges(n);
        TT_ASSERT(edges.size() == 1);
        std::vector<graphlib::OpType> tms = graph->get_edge_attributes(edges[0])->get_tms();

        if (tms.size() == 0)
            continue;

        graphlib::BudaOpNode* nop = graph->add_node(
            graphlib::create_node<graphlib::BudaOpNode>(n->name() + "_tm_nop", "nop"),
            graph->get_subgraph_id_for_node(n->id()));
        nop->copy_parent_op_attributes(graph->node_by_id(edges[0].producer_node_id)->as<graphlib::BudaOpNode>());
        graphlib::insert_node_on_edge(graph, edges[0], nop);
    }
}

static void graph_padding_pass(
    graphlib::Graph* graph,
    std::unordered_map<graphlib::Node*, const balancer::BudaOpNodeLegalizerFailureInfo>& nodes_to_pad,
    const balancer::BalancerConfig& balancer_config,
    std::shared_ptr<balancer::BalancerCacheCollection> balancer_cache_collection)
{
    // Pad graph before balancing.
    //
    if (!env_as<bool>("PYBUDA_DISABLE_PADDING_PASS") and !env_as<bool>("PYBUDA_PADDING_PASS_DISABLE_BUDA_OP"))
    {
        bool padded_flag =
            padding_placer::pad_pass_placer(graph, nodes_to_pad, balancer_config, balancer_cache_collection);
        if (padded_flag)
        {
            fix_tms_on_queues(graph);
            recalculate_shapes(graph);
            nodes_to_pad.clear();
        }
    }
}

static void graph_padding_override_pass(
    graphlib::Graph* graph,
    const py::dict& paddings_dict,
    const balancer::BalancerConfig& balancer_config,
    std::shared_ptr<balancer::BalancerCacheCollection> balancer_cache_collection)
{
    std::unordered_map<graphlib::Node*, const balancer::BudaOpNodeLegalizerFailureInfo> nodes_to_pad;

    // Convert paddings from pybind11, python py::dict to c++ std::map.
    //
    std::map<std::string, bool> paddings = paddings_dict.cast<std::map<std::string, bool>>();
    for (std::pair<std::string, bool> padding : paddings)
    {
        std::string name = padding.first;
        bool pad = padding.second;

        if (!pad)
            continue;

        Node* node = graph->get_node_by_name(name);
        balancer::BudaOpNodeLegalizerFailureInfo failure_info;
        failure_info.recordOpModelFailure(balancer::OpModelFailureReason::NoFailure);
        nodes_to_pad.emplace(node, failure_info);
    }

    graph_padding_pass(graph, nodes_to_pad, balancer_config, balancer_cache_collection);
}

// Insert nop after a node
static std::string insert_nop(graphlib::Graph* graph, const std::string& src_op)
{
    Node* src;
    TT_ASSERT(graph->has_node_with_name(src_op));
    src = graph->get_node_by_name(src_op);

    std::uint32_t buffer_index = 0;

    auto op_name = [](Node* src, std::uint32_t buffer_index)
    { return "dram_writer_" + std::to_string(buffer_index) + "_" + src->name(); };

    while (graph->has_node_with_name(op_name(src, buffer_index))) buffer_index++;

    graphlib::BudaOpNode* buffer_nop = nullptr;
    std::cout << "Insert NOP after " << src->name() << std::endl;

    for (graphlib::Edge e : graph->user_data_edges(src))
    {
        std::cout << " - edge" << std::endl;
        if (e.edge_type != graphlib::EdgeType::kData)
            continue;

        if (buffer_nop == nullptr)
        {
            std::cout << " - creating nop" << std::endl;
            buffer_nop = graph->add_node(
                graphlib::create_node<graphlib::BudaOpNode>(op_name(src, buffer_index), "nop"),
                graph->get_subgraph_id_for_node(src->id()));
            buffer_nop->set_shape(src->shape());
            buffer_nop->set_buffering_op(true);
        }

        auto [edge0, edge1] = graphlib::insert_node_on_edge(graph, e, buffer_nop);
        log_trace(
            LogGraphCompiler,
            "Inserted dram writer nop node {} between {} and {}",
            buffer_nop->name(),
            src->name(),
            graph->node_by_id(e.consumer_node_id)->name());

        // Move TMs to edge1
        auto& tms = graph->get_edge_attributes(edge0)->get_tms();
        if (true)  // not sure
        {
            // not hoisting tms, move them to edge1
            graph->get_edge_attributes(edge1)->set_tms(tms);
            graph->get_edge_attributes(edge0)->set_tms(std::vector<graphlib::OpType>{});
        }
    }

    TT_ASSERT(buffer_nop != nullptr);
    std::cout << " - created nop " << buffer_nop->name() << std::endl;
    return buffer_nop->name();
}

static void handle_node_exceeds_max_op_forks(
    graphlib::Graph* graph, balancer::BalancerError::NodeExceedsMaxOpForks type, int attempt)
{
    auto nodes = type.specific_node() ? std::vector<graphlib::Node*>{graph->node_by_id(type.node_id)} : graph->nodes();
    for (graphlib::Node* node : nodes)
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
            continue;

        std::vector<graphlib::Edge> users = graph->user_data_edges(node);
        if ((int)users.size() <= type.max_forks)
            continue;

        int num_nops = (int)ceil(users.size() / type.max_forks);

        for (int nop_i = 0; nop_i < num_nops; ++nop_i)
        {
            graphlib::OpType op_type("nop");
            graphlib::BudaOpNode* nop = graph->add_node(
                graphlib::create_node<graphlib::BudaOpNode>(
                    node->name() + "_attempt_" + std::to_string(attempt) + "_input_op_fork_nop" + std::to_string(nop_i),
                    op_type),
                graph->get_subgraph_id_for_node(node->id()));
            nop->set_shape(node->shape());
            nop->set_output_df(node->output_df());
            nop->set_epoch_type(node->get_epoch_type());
            if (is_integer_data_format(nop->output_df())) {
                // Set the math fidelity to HiFi4 for integer data formats
                nop->set_math_fidelity(MathFidelity::HiFi4);
                nop->set_accumulate_df(DataFormat::Int32);
                nop->set_intermediate_df(DataFormat::Int32);
            }

            graphlib::Edge input_nop_edge(node->id(), 0, nop->id(), 0, graphlib::EdgeType::kData);
            graph->add_edge(input_nop_edge);

            for (int edge_i = (nop_i * type.max_forks);
                 edge_i < std::min(((nop_i + 1) * type.max_forks), (int64_t)users.size());
                 ++edge_i)
            {
                graphlib::Edge edge = users[edge_i];
                auto edge_attrs = graph->get_edge_attributes(edge);
                graph->remove_edge(edge);

                graphlib::Edge output_nop_edge(
                    nop->id(), 0, edge.consumer_node_id, edge.consumer_input_port_id, edge.edge_type);
                graph->add_edge(output_nop_edge, edge_attrs);

                // Associate nop with one of the fwd to bwd edges (so it can belong to an op group)
                if (nop->get_epoch_type() != graphlib::NodeEpochType::Forward and edge_i % type.max_forks == 0)
                {
                    for (auto bwd_edge : graph->operand_edges(graph->node_by_id(edge.consumer_node_id)))
                    {
                        if (bwd_edge.edge_type != graphlib::EdgeType::kAutogradFwdToBwd and
                            bwd_edge.edge_type != graphlib::EdgeType::kAutogradFwdToGradient and
                            bwd_edge.edge_type != graphlib::EdgeType::kAutogradFwdToOptimizer and
                            bwd_edge.edge_type != graphlib::EdgeType::kAutogradFwdToRecompute)
                            continue;

                        graphlib::Edge nop_bwd_edge(
                            bwd_edge.producer_node_id,
                            bwd_edge.producer_output_port_id,
                            nop->id(),
                            0,
                            bwd_edge.edge_type);
                        graph->add_edge(nop_bwd_edge);
                    }
                }
            }
        }
    }
}

static void handle_input_exceeds_max_grid_forks(
    graphlib::Graph* graph, balancer::BalancerError::InputBroadcastExceedsMaxGridForks)
{
    split_broadcasts(graph);
}

static void handle_dram_writer_needs_nop(
    graphlib::Graph* graph,
    balancer::BalancerConfig& balancer_config,
    balancer::BalancerError::DRAMWriterNOPNeeded type)
{
    const std::string nop_name = insert_nop(graph, type.src);
    if (type.transpose)
        balancer_config.op_name_to_placer_overrides.emplace(nop_name, placer::PlacerOpOverride::force_op_transpose());
}

static bool handle_matmul_no_valid_grid(graphlib::Graph* graph, graphlib::BudaOpNode* op_node)
{
    TT_ASSERT(op_node != nullptr);
    TT_ASSERT(op_node->is_matmul());

    auto matmul_output_edges = graph->user_data_edges(op_node);
    if (matmul_output_edges.size() != 1)
        return false;

    auto edge_attr = graph->get_edge_attributes(matmul_output_edges[0]);
    auto consumer_op = dynamic_cast<graphlib::BudaOpNode*>(graph->node_by_id(matmul_output_edges[0].consumer_node_id));

    // Handle Matmul -> Eltwise Binary edge where Matmul has no valid grid
    if (consumer_op == nullptr or not graphlib::is_eltwise_binary(consumer_op))
        return false;
    if (not edge_attr->has_tm("transpose"))
        return false;

    // We have a matmul -> eltwise binary edge with transpose TM
    return try_insert_nop_on_transpose_edge(graph, matmul_output_edges[0]);
}

static bool handle_splice_no_valid_grid(graphlib::Graph* graph, graphlib::BudaOpNode* op_node)
{
    TT_ASSERT(op_node != nullptr);
    TT_ASSERT(op_node->is_splice());
    int dim = op_node->op_type().get_attr_as<int>("dim");
    if (dim != 2 and dim != 3)
        return false;

    auto [orig_dim, input_slices, output_stack] =
        op_node->py_attr<std::tuple<int, std::vector<int>, int>>("convert_mode_t");

    auto operand_edges = graph->operand_data_edges(op_node);
    TT_ASSERT(operand_edges.size() == input_slices.size());
    for (std::size_t i = 0; i < operand_edges.size(); ++i)
    {
        auto edge = operand_edges[i];
        auto slice_factor = input_slices[i];
        if (slice_factor > 1)
        {
            std::vector<graphlib::OpType>& tms = graph->get_edge_attributes(edge)->get_tms();
            tms.push_back(graphlib::OpType((orig_dim == 2) ? "vslice" : "hslice", {slice_factor}));
        }
    }

    if (output_stack > 1)
    {
        for (auto edge : graph->user_data_edges(op_node))
        {
            std::vector<graphlib::OpType>& tms = graph->get_edge_attributes(edge)->get_tms();
            tms.insert(tms.begin(), graphlib::OpType((orig_dim == 2) ? "vstack" : "hstack", {output_stack}));
        }
    }

    return true;
}

static void handle_no_valid_grid(
    graphlib::Graph* graph,
    std::unordered_map<graphlib::Node*, const balancer::BudaOpNodeLegalizerFailureInfo>& nodes_without_legal_op_model)
{
    std::vector<graphlib::Node*> fixed;
    for (const auto& node_fail_pair : nodes_without_legal_op_model)
    {
        const auto op_node = dynamic_cast<graphlib::BudaOpNode*>(node_fail_pair.first);

        if (op_node == nullptr)
            continue;

        if (op_node->is_matmul() and handle_matmul_no_valid_grid(graph, op_node))
        {
            fixed.push_back(op_node);
        }
        else if (op_node->is_splice() and handle_splice_no_valid_grid(graph, op_node))
        {
            fixed.push_back(op_node);
        }
    }

    for (auto op_node : fixed) nodes_without_legal_op_model.erase(op_node);
}

void insert_queues(
    graphlib::Graph* graph,
    std::unordered_map<graphlib::Node*, const balancer::BudaOpNodeLegalizerFailureInfo>& nodes_without_legal_op_model)
{
    std::vector<graphlib::Node*> fixed;

    for (auto const& [node, info] : nodes_without_legal_op_model)
    {
        auto* op_node = dynamic_cast<graphlib::BudaOpNode*>(node);
        TT_ASSERT(op_node);

        if (not (info.getOpModelFailureCountByType(balancer::OpModelFailureReason::UserAccessPreventsStreaming) || info.getOpModelFailureCountByType(balancer::OpModelFailureReason::OperandAndUserAccessPreventsStreaming)))
            continue;

        bool users_already_fixed = false;
        for (auto user : graph->user_data_edges(node))
        {
            auto* user_op_node = dynamic_cast<graphlib::BudaOpNode*>(graph->node_by_id(user.consumer_node_id));
            if (not user_op_node)
            {
                auto* user_queue_node = dynamic_cast<graphlib::QueueNode*>(graph->node_by_id(user.consumer_node_id));
                if (user_queue_node and user_queue_node->has_tag("no_valid_grids_queue"))
                    users_already_fixed = true;
                continue;
            }

            auto name = op_node->name() + "_no_valid_grids_queue";
            if (graph->has_node_with_name(name))
            {
                graphlib::QueueNode* queue = graph->get_node_by_name(name)->as<graphlib::QueueNode>();
                auto attr = graph->remove_edge(user);
                user.producer_node_id = queue->id();
                graph->add_edge(user, attr);
            }
            else
            {
                graphlib::QueueNode* queue = graph->add_node(
                    graphlib::create_node<graphlib::BufferingQueueNode>(name, 2),
                    graph->get_subgraph_id_for_node(op_node->id()));
                queue->tag("no_valid_grids_queue");

                bool constexpr inherit_consumer_attrs = false;
                bool constexpr remove_edge = true;
                bool constexpr place_tms_on_outgoing = true;
                graphlib::insert_node_on_edge(
                    graph,
                    user,
                    queue,
                    inherit_consumer_attrs,
                    remove_edge,
                    user.consumer_input_port_id,
                    place_tms_on_outgoing);
            }
        }

        if (not users_already_fixed)
            fixed.push_back(op_node);
    }

    for (auto op_node : fixed) nodes_without_legal_op_model.erase(op_node);
}

std::pair<std::shared_ptr<balancer::BalancerSolution>, bool> run_placer_buda_passes(
    graphlib::Graph* graph,
    balancer::BalancerConfig balancer_config,
    FractureChipIdAssignments const& fracture_chip_id_assignments,
    const py::dict& paddings_dict)
{
    int max_balancer_attempts = 30;
    int attempt = 0;
    int max_minor_attempts = 200;  // we expect a lot of these... really need to not have a limit, but a forward
                                   // progress indicator - keep going if the number of epochs placed is increasing.
    int minor_attempt = 0;

    std::shared_ptr<balancer::BalancerCacheCollection> balancer_cache_collection =
        std::make_shared<balancer::BalancerCacheCollection>();

    // Do padding if there are any overrides specified
    // in paddings_dict.
    //
    if (paddings_dict.size() > 0)
    {
        graph_padding_override_pass(graph, paddings_dict, balancer_config, balancer_cache_collection);
    }

    while (attempt < max_balancer_attempts)
    {
        check_unsupported_hw_ops(graph, true);

        try
        {
            // assign chips ids from fracture chip assignments, if any
            // iterate over the fracture chip assignments
            for (auto const& [node, chip_id] : fracture_chip_id_assignments)
            {
                // check if the balancer config already has this op name
                if (balancer_config.op_name_to_placer_overrides.find(node) !=
                    balancer_config.op_name_to_placer_overrides.end())
                {
                    // if it does, check if there is a chip id
                    if (balancer_config.op_name_to_placer_overrides[node].chip_id)
                    {
                        continue;
                    }
                    else
                    {
                        // if there is not, add the fracture chip id to the balancer config
                        balancer_config.op_name_to_placer_overrides[node].chip_id = chip_id;
                    }
                }
                else
                {
                    // if it does not, add it to the balancer config
                    auto placer_override = placer::PlacerOpOverride();
                    placer_override.chip_id = chip_id;
                    balancer_config.op_name_to_placer_overrides.emplace(node, placer_override);
                }
            }

            return std::make_pair(
                balancer::run_balancer_and_placer(graph, balancer_config, balancer_cache_collection), attempt > 0);
        }
        catch (balancer::BalancerError const& e)
        {
            log_debug(LogGraphCompiler, "Handle BalancerError: {}", e.what());

            attempt++;
            if (balancer::BalancerError::NodeExceedsMaxOpForks const* type =
                    std::get_if<balancer::BalancerError::NodeExceedsMaxOpForks>(&e.type))
            {
                handle_node_exceeds_max_op_forks(graph, *type, attempt);
            }
            else if (
                balancer::BalancerError::InputBroadcastExceedsMaxGridForks const* type =
                    std::get_if<balancer::BalancerError::InputBroadcastExceedsMaxGridForks>(&e.type))
            {
                handle_input_exceeds_max_grid_forks(graph, *type);
            }
            else if (
                balancer::BalancerError::DRAMWriterNOPNeeded const* type =
                    std::get_if<balancer::BalancerError::DRAMWriterNOPNeeded>(&e.type))
            {
                handle_dram_writer_needs_nop(graph, balancer_config, *type);
                attempt--;
                minor_attempt++;
                if (minor_attempt > max_minor_attempts)
                    break;
            }
            else if (
                balancer::BalancerError::NoValidGrid const* type =
                    std::get_if<balancer::BalancerError::NoValidGrid>(&e.type))
            {
                auto nodes_without_legal_op_model = type->nodes_without_legal_op_model;

                if (not nodes_without_legal_op_model.empty())
                    handle_no_valid_grid(graph, nodes_without_legal_op_model);

                if (not nodes_without_legal_op_model.empty())
                    graph_padding_pass(graph, nodes_without_legal_op_model, balancer_config, balancer_cache_collection);

                if (not nodes_without_legal_op_model.empty())
                    insert_queues(graph, nodes_without_legal_op_model);

                if (not nodes_without_legal_op_model.empty())
                    throw;
            }
            else if (balancer::BalancerError::Fatal const* type = std::get_if<balancer::BalancerError::Fatal>(&e.type))
            {
                log_fatal(LogGraphCompiler, "Fatal balancer error: {}", type->message);
            }
            else
            {
                throw;
            }
        }

        reportify::dump_graph(graph->name(), "balancer_error_handler_attempt" + std::to_string(attempt), graph);

        // We have to rerun the scheduler + some pre_placer graph passes after editing the graph
        placer::PlacerConfigUpdate updated_config = schedule_pre_placer_graph(
            graph,
            balancer_config.device_config,
            balancer_config.scheduler_config,
            balancer_config.chip_ids,
            balancer_config.op_names_to_chip_break,
            balancer_config.op_names_to_epoch_break,
            fracture_chip_id_assignments,
            "_attempt" + std::to_string(attempt) /* nops_remote_devices_postfix */,
            balancer_config.use_interactive_placer);
        balancer_config.op_to_chip_id_assignment = updated_config.op_to_chip_id_assignment;
    }

    log_fatal("Error: We failed to balance/place after {} attempts", max_balancer_attempts);
    // unreachable
    return std::make_pair(nullptr, false);
}
}  // namespace tt::passes
