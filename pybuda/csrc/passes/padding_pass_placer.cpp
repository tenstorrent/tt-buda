// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/padding_pass_placer.hpp"

#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include "balancer/balancer_utils.hpp"
#include "balancer/legalizer/legalizer.hpp"
#include "graph_lib/edge.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "graph_lib/utils.hpp"
#include "lower_to_buda/common.hpp"
#include "passes/eth_stream_reduction.hpp"
#include "utils/assert.hpp"
#include "shared_utils/sparse_matmul_utils.hpp"


using BudaOpAttrs = tt::BudaOpAttrs;

using Graph = tt::graphlib::Graph;
using NodeId = tt::graphlib::NodeId;
using Node = tt::graphlib::Node;
using NodeType = tt::graphlib::NodeType;
using TaggedNode = tt::graphlib::TaggedNode;
using Edge = tt::graphlib::Edge;
using EdgeType = tt::graphlib::EdgeType;
using PortId = tt::graphlib::PortId;
using BudaOpNode = tt::graphlib::BudaOpNode;
using OpType = tt::graphlib::OpType;
using Shape = tt::graphlib::Shape;
using ConstantInputNode = tt::graphlib::ConstantInputNode;

using Padding = tt::padding_placer::Padding;
using PaddingCriterion = tt::padding_placer::PaddingCriterion;
using PaddingOperation = tt::padding_placer::PaddingOperation;
using PaddingDimension = tt::padding_placer::PaddingDimension;

using SparseBUDA = tt::sparse::SparseBUDA;
using SparseCOO = tt::sparse::SparseCOO;

using OpModelFailureReason = tt::balancer::OpModelFailureReason;
using BudaOpNodeLegalizerFailureInfo = tt::balancer::BudaOpNodeLegalizerFailureInfo;
using BalancerConfig = tt::balancer::BalancerConfig;
using LegalOpModels = std::unordered_map<Node const *, std::vector<tt::balancer::OpModel>>;

namespace tt::padding_placer
{

// Inserts buffering queue instead of nop node.
void insert_queue_instead_of_nop(
    Graph *graph,
    Node *producer_node,
    Node *nop_node
)
{
    std::stringstream name_ss;
        name_ss << "queue_replacement_for_" << nop_node->name();

    graphlib::QueueNode *queue_node = graphlib::create_buffering_queue(graph, producer_node, name_ss.str(), graph->get_microbatch());
    log_debug(LogPadding, "\tCreating dram buffering queue node {} to replace nop {} ", name_ss.str(), nop_node->name());
    // After we have made queue_node, we now replace nop with it.
    replace_node(graph, /*original_node*/ nop_node, /*new_node*/ queue_node, /*skip_operands*/ false);
}

bool check_if_queue_fixes_failures(
    Graph *graph,
    Node *node,
    const BalancerConfig &balancer_config,
    std::shared_ptr<balancer::BalancerCacheCollection> balancer_cache_collection)
{
    // try adding queue
    std::vector<Node *> inserted_queues = insert_queue(graph, node);
    std::unordered_map<Node *, const BudaOpNodeLegalizerFailureInfo> failures =
        check_node_legality(graph, node, balancer_config, balancer_cache_collection);
    bool queue_fixes_failures = failures.size() == 0;
    // remove added queues from graph
    for (Node *queue : inserted_queues)
    {
        tt::graphlib::bypass_node(graph, queue, true /*remove_queue*/);
    }
    return queue_fixes_failures;
}

bool pad_pass_placer(
    Graph *graph,
    const std::unordered_map<graphlib::Node *, 
    const BudaOpNodeLegalizerFailureInfo> &nodes_to_pad,
    const balancer::BalancerConfig &balancer_config,
    std::shared_ptr<balancer::BalancerCacheCollection> balancer_cache_collection)
{
    const int PADDING_TRY_MAX = 10;
    bool padded = false;

    // We pass operations we want to pad, in other words if paddings map is not empty,
    // for each operations in our graph we check if it should be padded or not.
    // So, it should exist in the map and its flag should be TRUE, otherwise we skip the node.

    for (const auto &node_fail_pair : nodes_to_pad)
    {

        Node* node = node_fail_pair.first;
        const BudaOpNodeLegalizerFailureInfo failure_info = node_fail_pair.second;
        log_debug(LogPadding, "Padding node {} with {}", node->name(), failure_info.toString().c_str());

        // If the node has no valid grids and has padding_nop tag, we replace it with buffering queue.
        if (node->as<graphlib::TaggedNode>()->has_tag("padding_nop"))
        {
            std::vector<Node *> oprands = graph->data_operands(node);
            TT_ASSERT(oprands.size() == 1, "number of operands of padding_nop should be 1");
            Node *producer_node = oprands[0];
            insert_queue_instead_of_nop(graph, producer_node, /*nop_node*/ node);
            padded = true;
            continue;
        }

        if (node->as<graphlib::TaggedNode>()->has_tag("padding"))
            continue;

        std::uint32_t user_access_cnt = failure_info.getOpModelFailureCountByType(OpModelFailureReason::UserAccessPreventsStreaming);
        std::uint32_t buffer_alloc_cnt = failure_info.getOpModelFailureCountByType(OpModelFailureReason::InputBufferAllocationFailure);

        int padding_try_it = 0;
        bool padded_loop = false;

        Padding padding;
        // Preserve the original shape
        padding.orig_shape = node->shape();
        bool no_failures = false;

        // Check if adding queue after the node fixes failures. After padding loop is done, we will decide whether it is
        // better to pad the node or just to add queue.
        bool queue_fixes_failures =
            check_if_queue_fixes_failures(graph, node, balancer_config, balancer_cache_collection);

        while (padding_try_it++ < PADDING_TRY_MAX && buffer_alloc_cnt > 0)
        {
            padded_loop = pad_node(graph, node, padding);
            
            if (padded_loop) 
            {

                std::unordered_map<Node*, const BudaOpNodeLegalizerFailureInfo> failures = check_node_legality(graph, node, balancer_config, balancer_cache_collection);
                // user_access_cnt shouldn't be updated in padding loop because it is only used after the loop if there are still failures and graph is unpadded.
                buffer_alloc_cnt = (failures.size() > 0) ? failures[node].getOpModelFailureCountByType(OpModelFailureReason::InputBufferAllocationFailure) : 0;
                if (failures.size() > 0)
                {
                    // If we have tried to pad the node, but it failed, we remove padding and try again.
                    // Note, padding structure stays intact, and we resume padding in next iteration from where we have stopped.
                    remove_padding(graph, node, padding);
                    if (padded_loop)
                        padded_loop = false;
                    log_debug(LogPadding, "Node {} is illegal after padding: lhs_rt {} lhs_ct {} rhs_ct {}", node->name(), padding.pad_lhs_rt, padding.pad_lhs_ct, padding.pad_rhs_ct);
                }
                else
                {
                    no_failures = true;
                    log_debug(LogPadding, "Node {} is legal after padding: lhs_rt {} lhs_ct: {} rhs_ct: {}", node->name(), padding.pad_lhs_rt, padding.pad_lhs_ct, padding.pad_rhs_ct);
                    // If we added queue and also padded the node, we want to check if only adding the queue had solved
                    // the failures (queue_fixes_failures). If it did, we remove padding and keep the queue.
                    if (padding.added_queue && queue_fixes_failures)
                    {
                        remove_padding(graph, node, padding);
                        insert_queue(graph, node);
                        std::unordered_map<Node *, const BudaOpNodeLegalizerFailureInfo> failures =
                            check_node_legality(graph, node, balancer_config, balancer_cache_collection);
                        TT_ASSERT(
                            failures.size() == 0, "Adding queue is expected to fix all failures in this situation");
                    }
                    padded |= padded_loop;
                    break;
                }
            }

        }


        if (!no_failures)
        {
            if (buffer_alloc_cnt > 0)
            {
                // After padding loop we still have input buffer allocation count issues.
                log_warning(LogPadding, "Couldn't find padding for node: {} after {} iterations.", node->name(), padding_try_it);
            }

            if (user_access_cnt > 0)
            {
                // Inserting queue helps with user access failures but can also solve input buffer allocation issues.
                insert_queue(graph, node);
                padded = true;
            }
        }

    }

    return padded;
}

std::unordered_map<Node *, const BudaOpNodeLegalizerFailureInfo> check_node_legality(
    Graph *graph,
    Node *node,
    const BalancerConfig &balancer_config,
    std::shared_ptr<balancer::BalancerCacheCollection> balancer_cache_collection)
{
    // We use this functions to check if the particular node has legal op models.
    // This function is intended to be used in the padding pass,
    // but it can also be a general purpose function.

    std::unordered_set<graphlib::Node *> nodes_to_legalize = {node};

    try
    {
        LegalOpModels legal_op_models = tt::balancer::legalizer::get_legal_op_models(
            graph, balancer_config, balancer_cache_collection, &nodes_to_legalize);
    }
    catch (const balancer::BalancerError &e)
    {
        balancer::BalancerError::NoValidGrid const *type = std::get_if<balancer::BalancerError::NoValidGrid>(&e.type);
        if (type)
        {
            return type->nodes_without_legal_op_model;
        }
    }

    return {};
}


void remove_padding(Graph *graph, Node *node, const Padding &padding)
{
    if (node->node_type() != NodeType::kBudaOp)
        return;

    remove_pad(graph, node, padding);
    remove_unpad(graph, node /*, padding */);

    // Reset the shape of the node.
    node->set_shape(padding.orig_shape);
}

void remove_pad(Graph *graph, Node *node, const Padding &padding)
{
    if (node->as<BudaOpNode>()->is_sparse_matmul())
        restore_smm(graph, node, padding);
    remove_buda_pad(graph, node);
}

void restore_smm(Graph *graph, Node *node, const Padding &padding)
{

    std::vector<Edge> incoming_edges = graph->operand_data_edges(node);
    Node *incoming_node = nullptr;

    for (Edge incoming_edge : incoming_edges)
    {
        if (incoming_edge.consumer_input_port_id == 0)
        {
            NodeId incoming_node_id = incoming_edge.producer_node_id;
            incoming_node = graph->node_by_id(incoming_node_id);
        }
    }

    if (incoming_node == nullptr)
        return;

    // Create constant input node using incoming node.
    ConstantInputNode *pad_node = incoming_node->as<ConstantInputNode>();

    // Get sparse buda object that keeps information about the sparse tensor we want to pad
    SparseBUDA sparse_pad_node = pad_node->get_sparse_buda();

    // Change shape of the sparse buda tensor
    std::vector<std::int64_t> sparse_shape = sparse_pad_node.sparse_shape;
    std::uint32_t sparse_shape_size = sparse_shape.size();
    sparse_shape[sparse_shape_size - 2] -= padding.pad_lhs_rt * Shape::BUDA_TILE_DIM;
    sparse_shape[sparse_shape_size - 1] -= padding.pad_lhs_ct * Shape::BUDA_TILE_DIM;

    // Change shape of the sparse_zs tensors
    std::vector<SparseCOO> sparse_zs = sparse_pad_node.sparse_zs;
    for (SparseCOO& sparse_z : sparse_zs) {
        std::vector<std::int64_t> sparse_z_shape = sparse_z.shape;
        std::uint32_t sparse_z_shape_size = sparse_z_shape.size();
        sparse_z_shape[sparse_z_shape_size - 2] -= padding.pad_lhs_rt * Shape::BUDA_TILE_DIM;
        sparse_z_shape[sparse_z_shape_size - 1] -= padding.pad_lhs_ct * Shape::BUDA_TILE_DIM;
        sparse_z.shape = sparse_z_shape;
    }

    // Set the sparse buda tensor to pad node with the new shapes
    sparse_pad_node.sparse_shape = sparse_shape;
    sparse_pad_node.sparse_zs = sparse_zs;
    pad_node->set_sparse_buda(sparse_pad_node);

    if (padding.pad_lhs_rt > 0) {
        graphlib::OpNode* op = node->as<graphlib::OpNode>();
        auto op_attrs = op->op_attrs();
        op_attrs[5] = padding.sparse_r_attr;
        op->overwrite_op_attrs(op_attrs);
    }
}

void remove_unpad(Graph *graph, Node *node /*, Padding &padding*/)
{
    remove_buda_unpad(graph, node);
}

void remove_buda_pad(Graph *graph, Node *node)
{
    std::vector<Edge> incoming_edges = graph->operand_data_edges(node);
    for (Edge incoming_edge : incoming_edges)
    {
        std::vector<OpType> tms = graph->get_edge_attributes(incoming_edge)->get_tms();
        // Buda Pad operation is always the last TM on the edge in this phase.
        if (tms.size() > 0 && tms.back().op == "buda_pad")
            // Remove certain TM.
            tms.pop_back();
        // Set the new TMs without buda pad.
        graph->get_edge_attributes(incoming_edge)->set_tms(tms);
    }
}


void remove_buda_unpad(Graph *graph, Node *node)
{
    std::vector<Edge> outgoing_edges = graph->user_data_edges(node);
    for (Edge outgoing_edge : outgoing_edges)
    {

        NodeId outgoing_node_id = outgoing_edge.consumer_node_id;
        Node* outgoing_node = graph->node_by_id(outgoing_node_id);

        // Remove unpad node, nop, queue. In the future, we can have a few combinations.
        // In buda space unpad is an attribute.
        // Combination #1, nop, queue, buda_unpad
        // Combination #2, only buda_unpad
        // Combination #3, nop and buda_unpad
        // Combination #4, queue and buda_unpad 
        if (outgoing_node->node_type() == NodeType::kBudaOp)
        {
            // If the combination starts we possibly have combinations #1, #2 or #3.
            BudaOpNode *buda_op_node = outgoing_node->as<BudaOpNode>();
            // Get type of the operation
            std::string op_type = buda_op_node->as<BudaOpNode>()->op_type().op;
            if (op_type == "nop" && buda_op_node->as<graphlib::TaggedNode>()->has_tag("padding_nop")) 
            {
                // Potential combinations #1 and #3.
                std::vector<Edge> nop_outgoing_edges = graph->user_data_edges(outgoing_node);
                NodeId nop_outgoing_node_id = nop_outgoing_edges[0].consumer_node_id;
                Node* nop_outgoing_node = graph->node_by_id(nop_outgoing_node_id);
                if (nop_outgoing_edges.size() != 1)
                    break;
                std::vector<OpType> tms = graph->get_edge_attributes(nop_outgoing_edges[0])->get_tms();
                // Buda Unpad operation is always first TM on the edge in this phase
                if (tms.size() > 0 && tms[0].op == "buda_unpad") {
                    // Potential combination #3.
                    // Remove buda_unpad.
                    tms.erase(tms.begin());
                    graph->get_edge_attributes(nop_outgoing_edges[0])->set_tms(tms);
                    // Remove nop.
                    bypass_node(graph, outgoing_node, /* remove node */ true);
                }
                else if (nop_outgoing_node->node_type() == NodeType::kQueue)
                {
                    // Potential combination #1.
                    std::vector<Edge> queue_outgoing_edges = graph->user_data_edges(nop_outgoing_node);
                    if (queue_outgoing_edges.size() != 1)
                        break;
                    std::vector<OpType> tms = graph->get_edge_attributes(queue_outgoing_edges[0])->get_tms();
                    // Buda Unpad operation is always first TM on the edge in this phase
                    if (tms.size() > 0 && tms[0].op == "buda_unpad")
                    {
                        // Remove buda_unpad.
                        tms.erase(tms.begin());
                        graph->get_edge_attributes(queue_outgoing_edges[0])->set_tms(tms);
                        // Remove queue.
                        bypass_node(graph, nop_outgoing_node, /* remove node */ true);
                        // Remove nop.
                        bypass_node(graph, outgoing_node, /* remove node */ true);
                    }
                }
                
            }
            else 
            {
                // Potential combination #2.
                std::vector<OpType> tms = graph->get_edge_attributes(outgoing_edge)->get_tms();
                // Buda Unpad operation is always first TM on the edge in this phase
                if (tms.size() > 0 && tms[0].op == "buda_unpad")
                {
                    // Remove buda_unpad.
                    tms.erase(tms.begin());
                    graph->get_edge_attributes(outgoing_edge)->set_tms(tms);
                }
            }
        }
        else if (outgoing_node->node_type() == NodeType::kQueue)
        {
            // We possibly have combination #4.
            std::vector<Edge> queue_outgoing_edges = graph->user_data_edges(outgoing_node);
            if (queue_outgoing_edges.size() != 1)
                break;
            std::vector<OpType> tms = graph->get_edge_attributes(queue_outgoing_edges[0])->get_tms();
            // Buda Unpad operation is always first TM on the edge in this phase
            if (tms.size() > 0 && tms[0].op == "buda_unpad")
            {
                // Remove buda_unpad.
                tms.erase(tms.begin());
                graph->get_edge_attributes(queue_outgoing_edges[0])->set_tms(tms);
                // Remove queue.
                bypass_node(graph, outgoing_node, /* remove node */ true);
            }
        }

    }
}


bool pad_node(
    Graph *graph, 
    Node *node,
    Padding &padding
)
{

    // Get environment variables that tell us if we should pad matmul and elemnt-wise operations.
    bool element_wise_flag = env_as<bool>("PYBUDA_PADDING_PASS_ELEMENT_WISE", 1);
    bool matmul_flag = env_as<bool>("PYBUDA_PADDING_PASS_MATMUL", 1);
    bool sparse_matmul_flag = env_as<bool>("PYBUDA_PADDING_PASS_SPARSE_MATMUL", 1);
    // TODO: Should be enabled or removed.
    // bool splice_flag = env_as<bool>("PYBUDA_PADDING_PASS_SPLICE");

    // Padding criterion for each type of operations
    PaddingCriterion criterion = PaddingCriterion::BIGGEST_FACTOR_PRIME_10_INCREMENT;

    // If the node is not operation it's not element-wise and matmul, too.
    // If it is an operation, it can be for example "multiply", "add", "exp", etc.
    if (node->node_type() != NodeType::kBudaOp)
        return false;

    BudaOpNode *buda_op_node = node->as<BudaOpNode>();
    // Get type of the operation
    std::string op_type = node->as<BudaOpNode>()->op_type().op;

    if (!is_irregular(graph, node, padding, criterion))
    {
        padding.pad_lhs_rt++;
        // TODO: For now we increment only R dimension.
        // padding.pad_lhs_ct++;
        // padding.pad_rhs_ct++;
    }

    if (graphlib::is_eltwise(buda_op_node))
    {

        if (element_wise_flag && op_type != "splice")
        {
            compute_pad_eltwise(node, padding, criterion);
            return pad_eltwise(graph, node, padding);
        }

        /* TODO: Should be enabled.
        if (splice_flag && op_type == "splice")
            return pad_splice(graph, node);
        */

    }  // end if, is element-wise

    if (buda_op_node->is_matmul())
    {
        // Pad sparse matmul
        if (buda_op_node->is_sparse_matmul() && sparse_matmul_flag)
        {
            compute_pad_smm(graph, node, padding, criterion);
            return pad_smm(graph, node, padding);
        }

        // Pad matmul
        if (buda_op_node->is_matmul() && matmul_flag)
        {
            compute_pad_matmul(graph, node, padding, criterion);
            return pad_matmul(graph, node, padding);
        }

    }  // end if, matmul

    /* TODO: Should be enabled.
    if (buda_op_node->is_fused_op())
        return pad_fused_op(graph, node);
    */

    return false;
}

void set_padded_node_out_shape(Node* padded_node, Padding &padding)
{
    // Set shape
    std::vector<std::uint32_t> shape = padded_node->shape().as_vector();
    std::uint32_t shape_size = shape.size();
    shape[shape_size - 2] += padding.pad_lhs_rt * Shape::BUDA_TILE_DIM;
    shape[shape_size - 1] += padding.pad_rhs_ct * Shape::BUDA_TILE_DIM;
    padded_node->set_shape(Shape::create_buda(shape));
    padded_node->as<TaggedNode>()->add_tags({ { "padding", true } });
}

bool pad_eltwise(
    Graph *graph, 
    Node *node,
    Padding &padding
)
{

    bool padded = padding.pad_lhs_rt > 0 || padding.pad_lhs_ct > 0;

    // Both dimensions are regular, so we skip padding
    if (!padded)
        return padded;

    // Now, when we have figured out that shape is not irregular, we get incoming and outgoing edges.
    // The idea is to pad incoming nodes and unpad outgoing nodes, we get these using the edges of the given node.
    std::vector<Edge> incoming_edges = graph->operand_data_edges(node);
    std::vector<Edge> outgoing_edges = graph->user_data_edges(node);

    // Insert pad node for each incoming edge
    for (Edge incoming_edge : incoming_edges)
    {
        NodeId incoming_node_id = incoming_edge.producer_node_id;
        Node *incoming_node = graph->node_by_id(incoming_node_id);

        if (check_shape_size(incoming_node->shape()) || check_shape_ones(incoming_node->shape()))
            update_broadcast_op_with_pad(
                graph, 
                incoming_edge, 
                padding.pad_lhs_rt * Shape::BUDA_TILE_DIM, 
                padding.pad_lhs_ct * Shape::BUDA_TILE_DIM
            );
        else
            insert_pad_buda(
                graph,
                incoming_edge,
                padding.pad_lhs_rt,
                padding.pad_lhs_ct,
                // Padding value, used only in case
                // when we use buda implmentation for padding
                0.0
            );

    }  // end for, incoming edges

    set_padded_node_out_shape(node, padding);

    // Insert unpad for each outgoing edge
    for (Edge outgoing_edge : outgoing_edges)
    {

        insert_unpad(
            graph,
            node,
            outgoing_edge,
            padding,
            /* Insert nop and queue. */
            false
        );

    }  // end for, outgoing edges

    return padded;
}

bool pad_matmul(
    Graph *graph, 
    Node *node,
    Padding &padding
)
{

    std::vector<Edge> incoming_edges = graph->operand_data_edges(node);
    std::vector<Edge> outgoing_edges = graph->user_data_edges(node);

    // Get the operands of the matmul
    Edge lhs_edge;
    Edge rhs_edge;
    Edge bias_edge;
    for (Edge incoming_edge : incoming_edges)
    {
        if (incoming_edge.consumer_input_port_id == 0)
            lhs_edge = incoming_edge;
        else if (incoming_edge.consumer_input_port_id == 1)
            rhs_edge = incoming_edge;
        else if (incoming_edges.size() > 2 && incoming_edge.consumer_input_port_id == 2)
            bias_edge = incoming_edge;
    }

    // All operands have regular shape, nothing to do
    bool padded_lhs = padding.pad_lhs_rt > 0 || padding.pad_lhs_ct > 0;
    bool padded_rhs = padding.pad_lhs_ct > 0 || padding.pad_rhs_ct > 0;

    if (!padded_lhs && !padded_rhs)
        return false;

    // Insert pad for the left operand
    if (padded_lhs)
    {
        insert_pad_buda(
            graph,
            lhs_edge,
            padding.pad_lhs_rt,
            padding.pad_lhs_ct,
            // Padding value, used only in case
            // when we use buda implmentation for padding
            0.0);
    }

    // Insert pad for the right operand
    if (padded_rhs)
    {
        insert_pad_buda(
            graph,
            rhs_edge,
            // R dimension for right operand is the same as C dimension for left operand
            padding.pad_lhs_ct,
            padding.pad_rhs_ct,
            // Padding value, used only in case
            // when we use buda implmentation for padding
            0.0);
    }

    // If matmul has bias with broadcast, align with proper padding.
    if ((incoming_edges.size() > 2))
    {
        update_broadcast_op_with_pad(graph, bias_edge, padding.pad_lhs_rt, padding.pad_rhs_ct);
    }

    set_padded_node_out_shape(node, padding);

    // Insert unpad for each output node,
    for (Edge outgoing_edge : outgoing_edges)
    {
        insert_unpad(
            graph,
            node,
            outgoing_edge,
            padding,
            /* Insert nop and queue. */
            false
        );

    }  // end for, outgoing edges

    return true;
}

bool pad_smm(
    Graph *graph, 
    Node *node, 
    Padding &padding
)
{
    bool padded = false;

    std::vector<Edge> incoming_edges = graph->operand_data_edges(node);
    std::vector<Edge> outgoing_edges = graph->user_data_edges(node);

    bool unpad_flag = false;

    // Insert pad for each incoming edge.
    for (Edge incoming_edge : incoming_edges)
    {
        NodeId incoming_node_id = incoming_edge.producer_node_id;
        Node *incoming_node = graph->node_by_id(incoming_node_id);

        // Pad the LHS operand.
        if (incoming_edge.consumer_input_port_id == 0)
        {
            if (padding.pad_lhs_rt > 0 || padding.pad_lhs_ct > 0)
            {

                padded = true;

                // If the operation is sparse matmul we do not add buda_pad 
                // operation, we change the existing constant node
                insert_pad_smm(
                    incoming_node, 
                    /* pad R dimension */ 
                    padding.pad_lhs_rt * Shape::BUDA_TILE_DIM, 
                    /* pad C dimension */ 
                    padding.pad_lhs_ct * Shape::BUDA_TILE_DIM
                );

                // We unpad only if outer dimensions are padded, in the case of LHS operand that's R dimension
                if (padding.pad_lhs_rt > 0) {
                    unpad_flag = true;
                    graphlib::OpNode* op = node->as<graphlib::OpNode>();
                    auto op_attrs = op->op_attrs();
                    padding.sparse_r_attr = std::get<int>(op_attrs[5]);
                    op_attrs[5] = ((std::get<int>(op_attrs[5]) - 1) / Shape::BUDA_TILE_DIM + 1) * Shape::BUDA_TILE_DIM + (int) (padding.pad_lhs_rt * Shape::BUDA_TILE_DIM);
                    op->overwrite_op_attrs(op_attrs);
                }

            }

        } // end if, LHS operand

        // Pad the RHS operand.
        else if (incoming_edge.consumer_input_port_id == 1)
        {

            if (padding.pad_lhs_ct > 0 || padding.pad_rhs_ct > 0)
            {

                insert_pad_buda(
                    graph,
                    incoming_edge,
                    // R dimension for right operand is the same as C dimension for left operand
                    padding.pad_lhs_ct,
                    padding.pad_rhs_ct,
                    // Padding value, used only in case
                    // when we use buda implmentation for padding
                    0.0
                );

                padded = true;
                if (!unpad_flag && padding.pad_rhs_ct > 0)
                    unpad_flag = true;

            }

        } // end if, RHS operand

    }

    set_padded_node_out_shape(node, padding);

    if (unpad_flag)
    {

        for (Edge outgoing_edge : outgoing_edges)
        {

            insert_unpad(
                graph,
                node,
                outgoing_edge,
                padding,
                /* Insert nop and queue. */
                false
            );

        }  // end for, outgoing edges

    }

    return padded;
}

/* TODO: Should be implemented in the future.
bool pad_splice(
    Graph *graph, 
    Node *node
)
{
    bool padded = false;

    return padded;
e
}
*/

/* TODO: Should be implemented in the future.
bool pad_fused_op(
    Graph *graph, 
    Node *node
)
{

    bool padded = false;

    return padded;

}
*/


// TODO
// void remove_redundant_pad(Graph *graph)
// {
//     // In padding pass we pad particular operations,
//     // in such a way that we put padding before and
//     // unpadding after the operation.

//     // Sometimes we will have successive unpad and pad,
//     // that's uneccessary and expensive, so we want to
//     // reduce those situations. Also, we can have for the same
//     // operation a few the same paddings. In that case we want to
//     // replace all the same paddings with only one.

//     // This pass is divided into two passes, the first one is
//     // DFS removing and the second one is BFS removing,

//     // DFS
//     // remove_redudant_pad_dfs(graph);

//     // BFS
//     // remove_redudant_pad_bfs(graph);
// }

// TODO Needs to be reimplemented for TMs because padding pass was moved to lowered graph.
//
// void remove_redudant_pad_dfs(Graph *graph)
// {
//     // In this "sub-pass", we remove blocks with the same
//     // unpadding/padding operations.

//     std::vector<Node *> removing_nodes;

//     std::vector<Node *> nodes = tt::graphlib::topological_sort(*graph);

//     for (Node *node : nodes)
//     {
//         if (node->as<TaggedNode>()->has_tag("padding"))
//         {
//             std::string op_type = node->as<BudaOpNode>()->op_type().op;
//             // Check if it is unpad operator
//             if (op_type == "buda_unpad")
//             {
//                 // Check if it has corresponding pad operator
//                 Edge pad_edge = graph->user_data_edges(node)[0];
//                 NodeId pad_node_id = pad_edge.consumer_node_id;
//                 Node *pad_node = graph->node_by_id(pad_node_id);

//                 // Check if the operator has padding tag
//                 if (!pad_node->as<TaggedNode>()->has_tag("padding"))
//                     continue;
//                 if (pad_node->node_type() != NodeType::kPyOp)
//                     continue;
//                 std::string pad_op_type = pad_node->as<BudaOpNode>()->op_type().op;
//                 if (pad_op_type != "buda_pad")
//                     continue;

//                 // Check if the previous operation can change the result
//                 Edge previous_edge = graph->operand_data_edges(node)[0];
//                 NodeId previous_node_id = previous_edge.producer_node_id;
//                 Node *previous_node = graph->node_by_id(previous_node_id);
//                 if (change_result(previous_node))
//                     continue;

//                 // TODO: In Progress
//                 // // Check if the next operation can change the result
//                 // Edge next_edge = graph->user_data_edges(pad_node)[0];
//                 // NodeId next_node_id = next_edge.consumer_node_id;
//                 // Node *next_node = graph->node_by_id(next_node_id);
//                 // if (change_result(next_node))
//                 //     continue;

//                 // Get padding and unpadding attributes and compare them
//                 std::vector<BudaOpAttr> pad_attr = pad_node->as<BudaOpNode>()->op_type().attr;
//                 BudaOpAttr pad_attr_rt = pad_attr[0];
//                 BudaOpAttr pad_attr_ct = pad_attr[1];

//                 std::vector<BudaOpAttr> unpad_attr = node->as<BudaOpNode>()->op_type().attr;
//                 BudaOpAttr unpad_attr_rt = unpad_attr[0];
//                 BudaOpAttr unpad_attr_ct = unpad_attr[1];

//                 // If the padding and unpadding attributes are the same, remove particular nodes
//                 // and particular edges.
//                 if (pad_attr_rt == unpad_attr_rt && pad_attr_ct == unpad_attr_ct)
//                 {
//                     // This removal will be done in two steps.
//                     // Preserve nodes for removing, then remove them.

//                     // This is done in this way, because we can't remove nodes immediately.
//                     // We are iterating over already fetched nodes, and if we remove
//                     // some node, we will have a potential problem to access the same node.

//                     removing_nodes.push_back(node);
//                     removing_nodes.push_back(pad_node);

//                 }  // end if, remove block

//             }  // end if, buda unpad

//         }  // end if, padding node

//     }  // end for, graph traversal

//     // Remove nodes
//     for (Node *node : removing_nodes)
//     {
//         bypass_node(graph, node, /* remove node */ true);
//     }
// }

// TODO: In Progress
// void remove_redudant_pad_bfs(Graph *graph)
// {
//     // In this "sub-pass", we remove the same paddings and
//     // the same unpaddings used a few times for the same operation.

//     // std::vector<Node *> nodes = tt::graphlib::topological_sort(*graph);
//     // for (Node *node : nodes) {

//     // }

// }

void insert_pad_smm(Node *incoming_node, std::uint32_t pad_r, std::uint32_t pad_c)
{
    // In this case incoming node is our node we want to pad,
    // but we can't, because sparse matmul takes for left hand operand
    // only ConstantInputNode type, so we need to pad it in the other way.
    // That way is to change input node, because these kinds of nodes
    // are created only in python and cpp part of compiler has only pointer
    // to them. So, we make python function that pads the node and call it here.

    // Check if we need to pad the node.
    // For sparse matmul, we will have padding for R or C, or R and C dimensions,
    // case where both of dimensions are not padded is almost impossible, but we want to discard that possibility.
    if (pad_r <= 0 && pad_c <= 0)
        return;

    // Create constant input node using incoming node.
    ConstantInputNode *pad_node = incoming_node->as<ConstantInputNode>();

    // Get sparse buda object that keeps information about the sparse tensor we want to pad
    SparseBUDA sparse_pad_node = pad_node->get_sparse_buda();

    // Change shape of the sparse buda tensor
    std::vector<std::int64_t> sparse_shape = sparse_pad_node.sparse_shape;
    std::uint32_t sparse_shape_size = sparse_shape.size();
    sparse_shape[sparse_shape_size - 2] += pad_r;
    sparse_shape[sparse_shape_size - 1] += pad_c;

    // Change shape of the sparse_zs tensors
    std::vector<SparseCOO> sparse_zs = sparse_pad_node.sparse_zs;
    for (SparseCOO& sparse_z : sparse_zs) {
        std::vector<std::int64_t> sparse_z_shape = sparse_z.shape;
        std::uint32_t sparse_z_shape_size = sparse_z_shape.size();
        sparse_z_shape[sparse_z_shape_size - 2] += pad_r;
        sparse_z_shape[sparse_z_shape_size - 1] += pad_c;
        sparse_z.shape = sparse_z_shape;
    }

    // Set the sparse buda tensor to pad node with the new shapes
    sparse_pad_node.sparse_shape = sparse_shape;
    sparse_pad_node.sparse_zs = sparse_zs;
    pad_node->set_sparse_buda(sparse_pad_node);
}

void insert_pad_buda(Graph *graph, Edge incoming_edge, std::uint32_t pad_r, std::uint32_t pad_c, float value)
{
    log_trace(LogPadding, "Padding node with pad_r {} pad_c {} value {}.", pad_r, pad_c, value);
    std::vector<OpType::Attr> buda_pad_attrs(3, 0);
    buda_pad_attrs[0] = (int)pad_r;
    buda_pad_attrs[1] = (int)pad_c;
    buda_pad_attrs[2] = value;
    tt::BudaOpAttrs buda_attrs = tt::BudaOpAttrs{};
    buda_attrs["rt"] = buda_pad_attrs[0];
    buda_attrs["ct"] = buda_pad_attrs[1];
    buda_attrs["pad_value"] = buda_pad_attrs[2];

    graphlib::OpType tm_op_type = graphlib::OpType("buda_pad", buda_pad_attrs, buda_attrs);
    graph->get_edge_attributes(incoming_edge)->append_tm(tm_op_type);
}

void insert_unpad_buda(
    Graph *graph, 
    Node *node, 
    Edge edge, 
    std::uint32_t pad_r, 
    std::uint32_t pad_c,
    std::uint32_t orig_r,
    std::uint32_t orig_c
)
{
    log_trace(LogPadding, "Unpadding node with pad_r {} pad_c {}.", pad_r, pad_c);
    std::vector<std::uint32_t> shape_vect = node->shape().as_vector();
    std::uint32_t shape_size = shape_vect.size();

    std::vector<OpType::Attr> buda_unpad_attrs(4, 0);
    buda_unpad_attrs[0] = (int)pad_r;
    buda_unpad_attrs[1] = (int)pad_c;
    if (orig_r == 0) 
        buda_unpad_attrs[2] = (int)shape_vect[shape_size - 2];
    else
        buda_unpad_attrs[2] = (int)orig_r;
    if (orig_c == 0)
        buda_unpad_attrs[3] = (int)shape_vect[shape_size - 1];
    else
        buda_unpad_attrs[3] = (int)orig_c;

    tt::BudaOpAttrs buda_attrs = tt::BudaOpAttrs{};
    buda_attrs["rt"] = buda_unpad_attrs[0];
    buda_attrs["ct"] = buda_unpad_attrs[1];
    buda_attrs["orig_r"] = buda_unpad_attrs[2];
    buda_attrs["orig_c"] = buda_unpad_attrs[3];

    graphlib::OpType tm_op_type = graphlib::OpType("buda_unpad", buda_unpad_attrs, buda_attrs);
    graph->get_edge_attributes(edge)->prepend_tm(tm_op_type);
}

void insert_unpad(
    Graph *graph, 
    Node *node, 
    Edge edge,
    Padding &padding,
    bool insert_nop_queue
)
{

    std::vector<std::uint32_t> orig_shape = padding.orig_shape.as_vector();
    std::uint32_t orig_shape_size = orig_shape.size();
    std::uint32_t orig_r = orig_shape[orig_shape_size - 2];
    std::uint32_t orig_c = orig_shape[orig_shape_size - 1];

    NodeId outgoing_node_id = edge.consumer_node_id;
    Node* outgoing_node = graph->node_by_id(outgoing_node_id);

    if (insert_nop_queue)
    {
        BudaOpNode *nop_node = create_nop(graph, node, "unpadding");
        nop_node->as<TaggedNode>()->add_tags({ { "padding_nop", true } });
        insert_node_on_edge(graph, edge, nop_node, true, true, 0, true);

        Edge nop_edge = retrieve_between_edge(graph, nop_node, outgoing_node);

        insert_unpad_buda(
            graph,
            nop_node,
            nop_edge,
            // With buda implmentation we pad R in tiles
            padding.pad_lhs_rt,
            // With buda implmentation we pad C in tiles
            padding.pad_rhs_ct,
            // Original shape R dimension
            orig_r,
            // Original shape C dimension
            orig_c
        );

        padding.added_queue = true;
        insert_serialized_dram_queue_between_ops(
            // graph
            graph,
            // producer name
            nop_node->name(),
            // consumer name
            outgoing_node->name(),
            // operand index is always zero,
            // because vstack has only one operand
            (PortId) edge.consumer_input_port_id
        );
    }
    else if (padding.added_nop)
    {
        insert_unpad_buda(
            graph,
            node,
            edge,
            // With buda implmentation we pad R in tiles
            padding.pad_lhs_rt,
            // With buda implmentation we pad C in tiles
            padding.pad_rhs_ct,
            // Original shape R dimension
            orig_r,
            // Original shape C dimension
            orig_c
        );

        padding.added_queue = true;
        insert_serialized_dram_queue_between_ops(
            // graph
            graph,
            // producer name
            node->name(),
            // consumer name
            outgoing_node->name(),
            // operand index is always zero,
            // because vstack has only one operand
            (PortId) edge.consumer_input_port_id
        );
    }
    else
    {
        BudaOpNode *nop_node = create_nop(graph, node, "unpadding");
        nop_node->as<TaggedNode>()->add_tags({ { "padding_nop", true } });
        insert_node_on_edge(graph, edge, nop_node, true, true, 0, true);

        Edge nop_edge = retrieve_between_edge(graph, nop_node, outgoing_node);

        insert_unpad_buda(
            graph,
            nop_node,
            nop_edge,
            // With buda implmentation we pad R in tiles
            padding.pad_lhs_rt,
            // With buda implmentation we pad C in tiles
            padding.pad_rhs_ct,
            // Original shape R dimension
            orig_r,
            // Original shape C dimension
            orig_c
        );

        // Set the NOP indicator.
        padding.added_nop = true;
    }

}

std::vector<Node*> insert_queue(Graph *graph, Node *node)
{
    std::vector<Edge> outgoing_edges = graph->user_data_edges(node);
    std::vector<Node*> queue_nodes;
    // Insert unpad for each outgoing node.
    for (Edge outgoing_edge : outgoing_edges)
    {
        NodeId outgoing_node_id = outgoing_edge.consumer_node_id;
        Node* outgoing_node = graph->node_by_id(outgoing_node_id);

        if (outgoing_node->node_type() == NodeType::kQueue)
            continue;

        std::tuple<Edge, graphlib::Node *, Edge> new_queue_info = insert_serialized_dram_queue_between_ops(
                // graph
                graph,
                // producer name
                node->name(),
                // consumer name
                outgoing_node->name(),
                // operand index is always zero,
                // because vstack has only one operand
                (PortId)outgoing_edge.consumer_input_port_id);
        queue_nodes.push_back(std::get<1>(new_queue_info));
    }  // end for, outgoing edges
    return queue_nodes;
}

BudaOpNode* create_op(
    Graph *graph,
    Node *node,
    Shape shape,
    std::vector<OpType::Attr> attrs,
    std::string name,
    std::string op_name
)
{

    // This function creates new operation node based on given
    // graph, operation shape, attributes, operation name, 
    // name of a new node epoch type and output data format.

    OpType op_type = OpType(op_name, attrs);
    BudaOpNode *op_node = graph->add_node(
        tt::graphlib::create_node<BudaOpNode>(name, op_type),
        graph->get_subgraph_id_for_node(node->id())
    );
    op_node->set_epoch_type(node->get_epoch_type());
    op_node->set_output_df(node->output_df());
    op_node->set_shape(shape);
    op_node->as<TaggedNode>()->tag("original_op_name", name);
    op_node->as<TaggedNode>()->tag("original_op_type", op_name);
    op_node->as<TaggedNode>()->add_tags({ { "padding", true } });

    return op_node;
    
}

BudaOpNode* create_nop(
    Graph *graph,
    Node *node,
    std::string padding_type
)
{
    // Nop Attributes
    std::vector<OpType::Attr> nop_attrs = {};
    // Nop Names
    std::string op_name = "nop";
    std::string name = node->name() + "." + padding_type + "." + op_name + "_" + std::to_string(node->get_padding_id());
    node->increment_padding_id();
    // Nop Shape, the shape is the same as shape of the previous node
    BudaOpNode *nop = create_op(graph, node, node->shape(), nop_attrs, name, op_name);

    return nop;
}

bool is_irregular(Graph *graph, Node *node, Padding &padding, PaddingCriterion criterion)
{
    if (node->node_type() != NodeType::kBudaOp)
        return false;

    BudaOpNode *buda_op_node = node->as<BudaOpNode>();

    if (graphlib::is_eltwise(buda_op_node))
        return is_irregular_element_wise(node, padding, criterion);

    if (buda_op_node->is_matmul())
    {
        // Check only sparse matmul.
        if (buda_op_node->is_sparse_matmul())
            return is_irregular_smm(graph, node, padding, criterion);

        // Check only matmul.
        if (buda_op_node->is_matmul())
            return is_irregular_matmul(graph, node, padding, criterion);
    }

    return false;
}

bool is_irregular_element_wise(Node *node, Padding &padding, PaddingCriterion criterion)
{

    auto [row_size, column_size] = extract_dimensions_eltwise(node);

    // // Now, when we know that the operation is element-wise, we get its shape.
    // // Based on shape we get R dimension, its index and its value.
    // std::vector<std::uint32_t> shape = node->shape().as_vector();
    // std::uint32_t shape_size = shape.size();
    // std::uint32_t row_dim = shape_size - 2;
    // std::uint32_t row_size = shape[row_dim];
    // std::uint32_t column_dim = shape_size - 1;
    // std::uint32_t column_size = shape[column_dim];

    bool irregular = false;

    if (is_irregular(row_size + padding.pad_lhs_rt * Shape::BUDA_TILE_DIM, criterion))
        irregular |= true;

    if (is_irregular(column_size + padding.pad_lhs_ct * Shape::BUDA_TILE_DIM, criterion))
        irregular |= true;

    return irregular;
}

bool is_irregular_matmul(Graph *graph, Node *node, Padding &padding, PaddingCriterion criterion)
{

    auto [lhs_row_size, lhs_col_size, rhs_col_size] = extract_dimensions_matmul(graph, node);

    bool irregular = false;
    
    // Check if the operation has regular/irregular shape
    // Left operand, R dimension
    if (is_irregular(lhs_row_size + padding.pad_lhs_rt * Shape::BUDA_TILE_DIM, criterion))
        irregular |= true;

    // Left operand, C dimension
    if (is_irregular(lhs_col_size + padding.pad_lhs_ct * Shape::BUDA_TILE_DIM, criterion))
        irregular |= true;

    // Right operand, C dimension
    if (is_irregular(rhs_col_size + padding.pad_rhs_ct * Shape::BUDA_TILE_DIM, criterion))
        irregular |= true;

    return irregular;

}

bool is_irregular_smm(Graph *graph, Node *node, Padding &padding, PaddingCriterion criterion)
{
    
    auto [lhs_row_size, lhs_col_size, rhs_col_size] = extract_dimensions_smm(graph, node);

    bool irregular = false;

    // Check if the operation has regular/irregular shape
    // Left operand, R dimension.
    if (is_irregular(lhs_row_size + padding.pad_lhs_rt * Shape::BUDA_TILE_DIM, criterion))
        irregular |= true;

    // Left operand, C dimension, and right operand R dimension.
    if (is_irregular(lhs_col_size + padding.pad_lhs_ct * Shape::BUDA_TILE_DIM, criterion))
        irregular |= true;

    // Right operand, C dimension.
    if (is_irregular(rhs_col_size + padding.pad_rhs_ct * Shape::BUDA_TILE_DIM, criterion))
        irregular |= true;

    return irregular;

}


bool is_irregular(std::uint32_t dimension, PaddingCriterion criterion)
{
    if (criterion == PaddingCriterion::PRIME_NUMBER)
        return is_prime(dimension);

    if (criterion == PaddingCriterion::POWER_OF_TWO)
        return !is_power_of_2(dimension);

    if (criterion == PaddingCriterion::MULTIPLE_12_OF_TILE)
        return !is_multiple_12_of_tile(dimension);

    if (criterion == PaddingCriterion::MULTIPLE_10_OF_TILE)
        return !is_multiple_10_of_tile(dimension);

    if (criterion == PaddingCriterion::PRIME_TILE)
        return !is_tile_prime(dimension);

    if (criterion == PaddingCriterion::BIGGEST_FACTOR_PRIME_10 ||
        criterion == PaddingCriterion::BIGGEST_FACTOR_PRIME_10_INCREMENT)
        return is_biggest_factor_prime(10, dimension);

    return false;
}

bool is_sparse_irregular(std::uint32_t r_dim, std::uint32_t c_dim, PaddingCriterion criterion)
{
    // This method checks if given sparse matrix multiplication is irregular.

    if (criterion == PaddingCriterion::SPARSE_MATMUL_BASIC)
    {
        // If C dimension is greater than R dimension, then we skip the padding.
        if (c_dim > r_dim)
            return false;

        std::uint32_t r_tiles = get_tiles_num(r_dim);
        std::uint32_t c_tiles = get_tiles_num(c_dim);

        return is_sparse_irregular_tiles(r_tiles, c_tiles);
    }

    return false;
}

bool is_sparse_irregular_tiles(std::uint32_t r_tiles, std::uint32_t c_tiles)
{
    // This method checks if given sparse matrix multiplication is irregular.
    // First we get GCD of R and C tiles, then we multiply it with input tiles.
    // Input tiles is constant empirically determined.
    // If this multiplication biggee than r_tiles and c_tiles the sparse matmul is regular.

    std::uint32_t input_tiles = 10;
    std::uint32_t gcd_num = tt::balancer::gcd(r_tiles, c_tiles);
    std::uint32_t factor = input_tiles * gcd_num;

    if (factor > r_tiles && factor > c_tiles)
        return false;

    return true;
}

// Get number of tiles for given dimension.
std::uint32_t get_tiles_num(std::uint32_t dimension) { return (dimension - 1) / Shape::BUDA_TILE_DIM + 1; }

bool is_prime(std::uint32_t n)
{
    // Primality test

    if (n == 2 || n == 3)
        return true;

    if (n <= 1 || n % 2 == 0 || n % 3 == 0)
        return false;

    for (std::uint32_t i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return false;

    return true;
}

bool is_tile_prime(std::uint32_t n)
{
    std::uint32_t tile_prime = (n - 1) / Shape::BUDA_TILE_DIM + 1;
    if (is_prime(tile_prime))
        return true;
    return false;
}

bool is_power_of_2(std::uint32_t n)
{
    TT_ASSERT(n > 0, "Number must be strictly greater than 0.");
    float result_log = log2(n);
    if (ceil(result_log) == floor(result_log))
        return true;
    return false;
}

bool is_multiple_12_of_tile(std::uint32_t n)
{
    if (((n - 1) / Shape::BUDA_TILE_DIM + 1) % 12 == 0)
        return true;
    return false;
}

bool is_multiple_10_of_tile(std::uint32_t n)
{
    if (((n - 1) / Shape::BUDA_TILE_DIM + 1) % 10 == 0)
        return true;
    return false;
}

bool is_biggest_factor_prime(std::uint32_t threshold, std::uint32_t dimension)
{
    std::uint32_t tile_size = (dimension - 1) / Shape::BUDA_TILE_DIM + 1;
    std::vector<std::uint32_t> factors = prime_factorize(tile_size);

    if (factors.size() > 0 && factors[factors.size() - 1] > threshold)
        return true;
    return false;
}

std::vector<std::uint32_t> prime_factorize(std::uint32_t dimension)
{
    std::vector<std::uint32_t> factors;

    while (dimension % 2 == 0)
    {
        dimension /= 2;
        factors.push_back(2);
    }

    for (int divisor = 3; divisor < (int)std::sqrt(dimension) + 1; divisor += 2)
    {
        while (dimension % divisor == 0)
        {
            dimension /= divisor;
            factors.push_back(divisor);
        }
    }

    if (dimension > 2)
        factors.push_back(dimension);

    return factors;
}


void compute_pad_eltwise(Node *node, Padding &padding, PaddingCriterion criterion)
{

    auto [row_size, column_size] = extract_dimensions_eltwise(node);

    std::uint32_t pad_lhs_r = compute_pad(row_size + padding.pad_lhs_rt * Shape::BUDA_TILE_DIM, criterion);
    std::uint32_t pad_lhs_c = compute_pad(column_size + padding.pad_lhs_ct * Shape::BUDA_TILE_DIM, criterion);

    padding.pad_lhs_rt += pad_lhs_r / Shape::BUDA_TILE_DIM;
    padding.pad_lhs_ct += pad_lhs_c / Shape::BUDA_TILE_DIM;
    padding.pad_rhs_ct += pad_lhs_c / Shape::BUDA_TILE_DIM;

}

void compute_pad_matmul(Graph *graph, Node *node, Padding &padding, PaddingCriterion criterion)
{

    auto [lhs_row_size, lhs_col_size, rhs_col_size] = extract_dimensions_matmul(graph, node);

    std::uint32_t pad_lhs_r = compute_pad(lhs_row_size + padding.pad_lhs_rt * Shape::BUDA_TILE_DIM, criterion);
    std::uint32_t pad_lhs_c = compute_pad(lhs_col_size + padding.pad_lhs_ct * Shape::BUDA_TILE_DIM, criterion);
    std::uint32_t pad_rhs_c = compute_pad(rhs_col_size + padding.pad_rhs_ct * Shape::BUDA_TILE_DIM, criterion);

    padding.pad_lhs_rt += pad_lhs_r / Shape::BUDA_TILE_DIM;
    padding.pad_lhs_ct += pad_lhs_c / Shape::BUDA_TILE_DIM;
    padding.pad_rhs_ct += pad_rhs_c / Shape::BUDA_TILE_DIM;

}

void compute_pad_smm(Graph *graph, Node *node, Padding &padding, PaddingCriterion criterion)
{

    auto [lhs_row_size, lhs_col_size, rhs_col_size] = extract_dimensions_smm(graph, node);

    std::uint32_t pad_lhs_r = compute_pad(lhs_row_size + padding.pad_lhs_rt * Shape::BUDA_TILE_DIM, criterion);
    std::uint32_t pad_lhs_c = compute_pad(lhs_col_size + padding.pad_lhs_ct * Shape::BUDA_TILE_DIM, criterion);
    std::uint32_t pad_rhs_c = compute_pad(rhs_col_size + padding.pad_rhs_ct * Shape::BUDA_TILE_DIM, criterion);

    padding.pad_lhs_rt += pad_lhs_r / Shape::BUDA_TILE_DIM;
    padding.pad_lhs_ct += pad_lhs_c / Shape::BUDA_TILE_DIM;
    padding.pad_rhs_ct += pad_rhs_c / Shape::BUDA_TILE_DIM;

}

std::uint32_t compute_pad(std::uint32_t dimension, PaddingCriterion criterion)
{

    if (criterion == PaddingCriterion::POWER_OF_TWO)
        return round_power_of_2(dimension) - dimension;

    if (criterion == PaddingCriterion::PRIME_TILE)
        return round_tile_prime(dimension) - dimension;

    if (criterion == PaddingCriterion::MULTIPLE_12_OF_TILE)
        return round_multiple_12_of_tile(dimension) - dimension;

    if (criterion == PaddingCriterion::MULTIPLE_10_OF_TILE)
        return round_multiple_10_of_tile(dimension) - dimension;

    if (criterion == PaddingCriterion::BIGGEST_FACTOR_PRIME_10)
        return round_biggest_factor_prime(10, dimension) - dimension;

    if (criterion == PaddingCriterion::BIGGEST_FACTOR_PRIME_10_INCREMENT)
        return increment_until_valid(dimension, criterion) - dimension;

    // TODO: It should be assertion ???
    return 0;
}

std::uint32_t compute_sparse_pad(std::uint32_t r_dim, std::uint32_t c_dim, PaddingCriterion criterion)
{
    if (criterion == PaddingCriterion::SPARSE_MATMUL_BASIC)
        return compute_sparse_pad_basic(r_dim, c_dim);
    if (criterion == PaddingCriterion::SPARSE_MATMUL_FACTORS)
        return compute_sparse_pad_factors(r_dim, c_dim);

    // TODO: It should be assertion ???
    return 0;
}

std::uint32_t compute_sparse_pad_basic(std::uint32_t r_dim, std::uint32_t c_dim)
{
    std::uint32_t r_tiles = get_tiles_num(r_dim);
    std::uint32_t c_tiles = get_tiles_num(c_dim);

    while (is_sparse_irregular_tiles(r_tiles, c_tiles)) c_tiles++;

    return c_tiles * Shape::BUDA_TILE_DIM - c_dim;
}

std::uint32_t compute_sparse_pad_factors(std::uint32_t r_dim, std::uint32_t c_dim)
{
    std::uint32_t r_tiles = get_tiles_num(r_dim);
    std::uint32_t c_tiles = get_tiles_num(c_dim);
    std::vector<std::uint32_t> factors_r = prime_factorize(r_tiles);
    std::uint32_t new_c_tiles = 1;
    for (int factor_it = factors_r.size() - 1; factor_it >= 0; factor_it--)
    {
        if (new_c_tiles >= c_tiles)
            break;
        new_c_tiles *= factors_r[factor_it];
    }
    return new_c_tiles * Shape::BUDA_TILE_DIM - c_dim;
}

std::uint32_t round_power_of_2(std::uint32_t n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;

    return n;
}

std::uint32_t round_tile_prime(std::uint32_t n) { return ((n - 1) / Shape::BUDA_TILE_DIM + 2) * Shape::BUDA_TILE_DIM; }

std::uint32_t round_multiple_12_of_tile(std::uint32_t n)
{
    return ((n - 1) / Shape::BUDA_TILE_DIM / 12 + 1) * Shape::BUDA_TILE_DIM * 12;
}

std::uint32_t round_multiple_10_of_tile(std::uint32_t n)
{
    return ((n - 1) / Shape::BUDA_TILE_DIM / 10 + 1) * Shape::BUDA_TILE_DIM * 10;
}

std::uint32_t round_biggest_factor_prime(std::uint32_t threshold, std::uint32_t dimension)
{
    std::uint32_t result = 1;
    std::uint32_t tile_size = (dimension - 1) / Shape::BUDA_TILE_DIM + 1;
    std::vector<std::uint32_t> factors = round_biggest_factor_prime_inner(threshold, tile_size);
    for (std::uint32_t item : factors) result *= item;
    return result * Shape::BUDA_TILE_DIM;
}

std::vector<std::uint32_t> round_biggest_factor_prime_inner(std::uint32_t threshold, std::uint32_t dimension)
{
    std::uint32_t result = 1;
    std::vector<std::uint32_t> factors = prime_factorize(dimension);
    if (factors[factors.size() - 1] > threshold)
    {
        for (int it = factors.size() - 1; it >= 0; it--)
        {
            if (factors[it] > threshold)
                factors[it] += 1;
            result *= factors[it];
        }
        return round_biggest_factor_prime_inner(threshold, result);
    }
    else
        return factors;
}

bool check_shape_dims(Shape shape)
{
    // This function is because of imperfection of padding algorithm.
    // Padding doesn't work for operations with shape that have dimensions
    // Z, W, etc. bigger than zero.

    // e.g. these are allowed shapes: [32, 192], [1, 1, 48, 256]
    // e.g. these are not allowed shapes: [1, 6, 32, 192], [32, 1, 48, 256], [8, 192, 192]

    std::vector<std::uint32_t> shape_vect = shape.as_vector();
    std::uint32_t shape_size = shape_vect.size();
    if (shape_size > 2)
    {
        for (std::uint32_t it = 0; it < shape_size - 2; it++)
        {
            if (shape_vect[it] > 1)
                return false;
        }
    }
    return true;
}

std::uint32_t increment_until_valid(std::uint32_t dimension, PaddingCriterion criterion)
{
    std::uint32_t tile_num = get_tiles_num(dimension);

    while (is_irregular(tile_num * Shape::BUDA_TILE_DIM, criterion))
    {
        tile_num++;
    }

    return tile_num * Shape::BUDA_TILE_DIM;
}

bool check_shape_size(tt::graphlib::Shape shape)
{
    // Check if shape size smaller than 2
    if (shape.as_vector().size() < 2)
        return true;
    return false;
}

bool check_shape_ones(tt::graphlib::Shape shape)
{
    // Check if all dimensions in tesnor are 1,
    // if that's true we don't need to pad,
    // the tensor will be broadcasted.
    bool is_one = true;

    std::vector<std::uint32_t> shape_vect = shape.as_vector();
    std::uint32_t length = shape_vect.size();
    for (std::uint32_t index = 0; index < length - 1; index++)
    {
        if (shape_vect[index] != 1)
        {
            is_one = false;
            break;
        }
    }

    return is_one;
}

void update_broadcast_op_with_pad(Graph *graph, Edge edge, std::uint32_t pad_r, std::uint32_t pad_c)
{
    // If padded node has broadcast value on input, that broadcast needs to be updated too.

    std::vector<graphlib::OpType> tms = graph->get_edge_attributes(edge)->get_tms();
    for (OpType tm : tms)
    {
        std::string tm_op = tm.op;
        if (tm_op == "broadcast")
        {
            std::vector<BudaOpAttr> attrs = tm.attr;
            // Broadcast parameters are computed and added before pre-lowering pass where we actually add padding pass,
            // so, if we only change the shape broadcast on the will not be affected, because of that we change it
            // manually.
            int broadcast_dim = std::get<int>(attrs[0]);
            if (broadcast_dim == -2)
            {
                if (pad_r > 0)
                {
                    int broadcast_size = std::get<int>(attrs[1]);
                    graph->get_edge_attributes(edge)->remove_broadcast_dim(-2);
                    graph->get_edge_attributes(edge)->set_broadcast_dim(-2, broadcast_size + (int)pad_r);
                }
            }
            if (broadcast_dim == -1)
            {
                if (pad_c > 0)
                {
                    int broadcast_size = std::get<int>(attrs[1]);
                    graph->get_edge_attributes(edge)->remove_broadcast_dim(-1);
                    graph->get_edge_attributes(edge)->set_broadcast_dim(-1, broadcast_size + (int)pad_c);
                }
            }
        }
    }
}

bool change_result(Node *operation)
{
    // Check if we have operations that can change zero to something else
    std::string op_type = operation->as<BudaOpNode>()->op_type().op;

    if (op_type == "exp")
        return true;
    if (op_type == "log")
        return true;
    if (op_type == "cosine")
        return true;
    if (op_type == "sigmoid")
        return true;
    if (op_type == "gelu_derivative")
        return true;
    if (op_type == "reciprocal")
        return true;
    return false;
}

std::tuple<std::uint32_t, std::uint32_t, std::uint32_t> extract_dimensions_smm(Graph *graph, Node *node)
{

    // This function extracts all necessary dimensions for sparse matmul padding.

    Node *lhs_node = nullptr;
    Node *rhs_node = nullptr;
    std::vector<Edge> incoming_edges = graph->operand_data_edges(node);
    for (Edge incoming_edge : incoming_edges)
    {
        NodeId incoming_node_id = incoming_edge.producer_node_id;
        Node *incoming_node = graph->node_by_id(incoming_node_id);
        // Get the first operand. Unique sparse tiles.
        if (incoming_edge.consumer_input_port_id == 0)
            lhs_node = incoming_node;

        // Get the second operand. Activations.
        else if (incoming_edge.consumer_input_port_id == 1)
            rhs_node = incoming_node;
    }

    TT_ASSERT(lhs_node != nullptr && rhs_node != nullptr, "Pointers of sparse matmul operands must be non-null value.");

    ConstantInputNode *lhs_node_const = lhs_node->as<ConstantInputNode>();
    SparseBUDA lhs_node_sparse = lhs_node_const->get_sparse_buda();
    std::vector<std::int64_t> lhs_shape = lhs_node_sparse.sparse_shape;
    std::uint32_t lhs_shape_size = lhs_shape.size();
    std::uint32_t lhs_row_dim = lhs_shape_size - 2;
    std::uint32_t lhs_col_dim = lhs_shape_size - 1;
    std::uint32_t lhs_row_size = lhs_shape[lhs_row_dim];
    std::uint32_t lhs_col_size = lhs_shape[lhs_col_dim];

    std::vector<std::uint32_t> rhs_shape = rhs_node->shape().as_vector();
    std::uint32_t rhs_shape_size = rhs_shape.size();
    std::uint32_t rhs_col_dim = rhs_shape_size - 1;
    std::uint32_t rhs_col_size = rhs_shape[rhs_col_dim];

    return {
        lhs_row_size,
        lhs_col_size,
        rhs_col_size
    };

}

std::tuple<std::uint32_t, std::uint32_t, std::uint32_t> extract_dimensions_matmul(Graph *graph, Node *node)
{

    std::vector<Edge> incoming_edges = graph->operand_data_edges(node);

    Node *lhs_operand = nullptr;
    Node *rhs_operand = nullptr;
    for (Edge incoming_edge : incoming_edges)
    {
        // Left operand has input port id 0
        if (incoming_edge.consumer_input_port_id == 0)
            lhs_operand = graph->node_by_id(incoming_edge.producer_node_id);
        // Right operand has input port id 1
        else if (incoming_edge.consumer_input_port_id == 1)
            rhs_operand = graph->node_by_id(incoming_edge.producer_node_id);
    }

    // Get shapes and R/C dimensions of the operands
    // Left operand
    std::vector<std::uint32_t> lhs_shape = lhs_operand->shape().as_vector();
    std::uint32_t lhs_shape_size = lhs_shape.size();
    std::uint32_t lhs_row_dim = lhs_shape_size - 2;
    std::uint32_t lhs_row_size = lhs_shape[lhs_row_dim];
    std::uint32_t lhs_col_dim = lhs_shape_size - 1;
    std::uint32_t lhs_col_size = lhs_shape[lhs_col_dim];

    // Right operand
    // In case for right hand we take only C dimension,
    // because inner dimensions are equal for both operands in matmul
    std::vector<std::uint32_t> rhs_shape = rhs_operand->shape().as_vector();
    std::uint32_t rhs_shape_size = rhs_shape.size();
    std::uint32_t rhs_col_dim = rhs_shape_size - 1;
    std::uint32_t rhs_col_size = rhs_shape[rhs_col_dim];

    return {
        lhs_row_size,
        lhs_col_size,
        rhs_col_size
    };

}

std::tuple<std::uint32_t, std::uint32_t> extract_dimensions_eltwise(Node *node)
{

    // Now, when we know that the operation is element-wise, we get its shape.
    // Based on shape we get R and C dimensions, their index and value.
    std::vector<std::uint32_t> shape = node->shape().as_vector();
    std::uint32_t shape_size = shape.size();
    std::uint32_t row_dim = shape_size - 2;
    std::uint32_t row_size = shape[row_dim];
    std::uint32_t column_dim = shape_size - 1;
    std::uint32_t column_size = shape[column_dim];

    return { row_size, column_size };

}

// void update_padding_matmul()
// {

// }

// void update_padding_eltwise()
// {

// }


}  // namespace tt::padding_placer
