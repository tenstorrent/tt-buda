// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/fracture.hpp"

#include "graph_lib/node_types.hpp"
#include "graph_lib/query.hpp"
#include "graph_lib/utils.hpp"
#include "passes/nd_slice.hpp"
#include "utils/assert.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{
using FracturedNodes = std::pair<NDSlice, std::vector<graphlib::NodeId>>;

static std::string fractured_name(graphlib::Node* node, int index, int dim = 0, std::string const& tag = "")

{
    std::string dim_str;
    switch (dim)
    {
        case -1: dim_str = "n"; break;
        case -2: dim_str = "m"; break;
        case -3: dim_str = "t"; break;
        case NDSlice::k_dim: dim_str = "k"; break;
        default: dim_str = std::to_string(dim); break;
    }
    return "fractured_" + tag + (tag.empty() ? "" : "_") + (dim ? dim_str : "") + std::to_string(index) + "_" +
           node->name();
}

static int dim_tiles(int d) { return (d + graphlib::Shape::BUDA_TILE_DIM - 1) / graphlib::Shape::BUDA_TILE_DIM; }

static bool is_parameter(graphlib::InputNode* input)
{
    return input->is_parameter() or input->is_constant() or input->is_optimizer_parameter();
};

static std::vector<graphlib::Node*> get_group_nodes(graphlib::Graph* graph, FractureGroup const& group)
{
    std::vector<graphlib::Node*> nodes;
    nodes.reserve(group.size());

    for (auto const& [name, dims, factors] : group)
    {
        auto* n = graph->get_node_by_name(name);
        TT_ASSERT(n, "Op not found in graph");
        nodes.push_back(n);
    }
    return nodes;
}

static bool node_in_group(graphlib::Node const* node, FractureGroup const& group)
{
    for (auto const& [name, dims, factors] : group)
    {
        if (node->name() == name)
            return true;
    }
    return false;
}

static FractureGroup expand_query_fracture_group(graphlib::Graph* graph, FractureGroup group)
{
    using FractureGroup = std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>>>;
    FractureGroup expanded_group;
    expanded_group.reserve(group.size());
    auto nodes = graph->nodes();
    for (auto const& [name_regex, dims, factors] : group)
    {
        for (graphlib::Node* node : graphlib::query::filter_nodes_by_name(graph, name_regex))
        {
            expanded_group.emplace_back(node->name(), dims, factors);
        }
    }
    return expanded_group;
}

static FractureGroup infer_op_group(graphlib::Graph* graph, FractureGroup group)
{
    auto contains = [](FractureGroup const& group, graphlib::Node* node) -> bool
    {
        return std::find_if(
                   group.begin(), group.end(), [node](auto const& f) { return node->name() == std::get<0>(f); }) !=
               group.end();
    };

    // Infer ops that are directly connected to fractured parameters
    for (graphlib::InputNode* param :
         graphlib::sorted<graphlib::InputNode>(graph, get_group_nodes(graph, group), is_parameter))
    {
        for (graphlib::Node* node : graph->data_users(param))
        {
            if (not contains(group, node))
            {
                group.push_back(std::make_tuple(node->name(), std::vector<int>{}, std::vector<int>{}));
            }
        }

        // Implicitly bring in write back operands
        for (graphlib::Edge edge : graph->operand_edges(param))
        {
            TT_ASSERT(edge.edge_type == graphlib::EdgeType::kPartialDataCopy, "Only support partial datacopy for now");
            graphlib::Node* output = graph->node_by_id(edge.producer_node_id);
            auto operands = graph->data_operands(output);
            TT_ASSERT(operands.size() == 1);
            group.push_back(std::make_tuple(operands.front()->name(), std::vector<int>{}, std::vector<int>{}));
            group.push_back(std::make_tuple(output->name(), std::vector<int>{}, std::vector<int>{}));
        }
    }

    std::vector<graphlib::Node*> node_ptrs = graphlib::sorted<graphlib::Node>(graph, get_group_nodes(graph, group));
    auto order = [&node_ptrs](graphlib::Node* n)
    {
        auto iter = std::find_if(node_ptrs.begin(), node_ptrs.end(), [n](auto const* n0) { return n0 == n; });
        TT_LOG_ASSERT(iter != node_ptrs.end(), "Specified fracture op not found in graph: {}", n->name());
        return iter;
    };

    // Infer ops that are contained within bounded subgraph
    FractureGroup inferred_group = group;
    for (auto const& [producer, pdims, pfactors] : group)
    {
        for (auto const& [consumer, cdims, cfactors] : group)
        {
            if (producer == consumer)
                continue;

            graphlib::Node* producer_node = graph->get_node_by_name(producer);
            graphlib::Node* consumer_node = graph->get_node_by_name(consumer);

            if (order(producer_node) > order(consumer_node))
                std::swap(producer_node, consumer_node);

            for (graphlib::Node* node : subgraph(graph, producer_node, consumer_node))
            {
                if (not contains(inferred_group, node))
                {
                    inferred_group.push_back(std::make_tuple(node->name(), std::vector<int>{}, std::vector<int>{}));
                }
            }
        }
    }

    return inferred_group;
}

static void assign_writeback_output_order(graphlib::Graph* graph)
{
    int offset = (int)graph->ordered_module_outputs().size();
    auto partial_datacopy_outputs = graph->ordered_partial_datacopy_outputs();
    for (int index = 0; index < (int)partial_datacopy_outputs.size(); ++index)
    {
        partial_datacopy_outputs[index]->as<graphlib::OutputNode>()->set_partial_datacopy_golden_output_index(
            offset + index);
    }
}

static void handle_writeback_golden_transforms(
    graphlib::OutputNode* output, graphlib::OutputNode* fractured_output, NDSlice::Slice slice)
{
    constexpr bool channel_last = false;
    std::vector<graphlib::OpType::Attr> pad = {
        0 /*padding_left*/, 0 /*padding_right*/, 0 /*padding_top*/, 0 /*padding_bottom*/, channel_last};

    for (auto [dim, index, factor] : slice.indices)
    {
        TT_ASSERT(dim == -2 or dim == -1, "Pad only supports 2d");
        int dim_size = (int)output->shape()[dim];
        int chunk_size = dim_size / factor;
        int pad_idx = dim == -1 ? 0 : 2;
        pad[pad_idx + 0] = index * chunk_size;
        pad[pad_idx + 1] = dim_size - (index + 1) * chunk_size;
    }

    fractured_output->add_golden_transform(graphlib::OpType("pad", pad));
    fractured_output->set_partial_datacopy_golden_output_index(*output->get_partial_datacopy_golden_output_index());
}

static std::unique_ptr<graphlib::PyOpNode> create_slice(
    graphlib::Node* op, graphlib::Shape operand_shape, int dim, int i, int factor, std::string new_op_name)
{
    graphlib::Shape shape = operand_shape;

    // If the dim is 1 along the fractured dim, this is a bcast special case
    if (not shape.index_in_bounds(dim) or shape[dim] == 1)
        return nullptr;

    TT_ASSERT(dim < -2 or dim_tiles(shape[dim]) % factor == 0, shape, dim, factor, op->name());
    TT_ASSERT(dim >= -2 or shape[dim] % factor == 0, shape, dim, factor, op->name());

    int stride = (int)shape[dim];
    shape[dim] /= factor;

    int start = i * (int)shape[dim];
    int length = (int)shape[dim];
    graphlib::OpType select("select", {dim, start, length, stride});
    auto new_op = graphlib::create_node<graphlib::PyOpNode>(new_op_name, select);
    new_op->set_shape(shape);
    new_op->set_epoch_type(op->get_epoch_type());
    new_op->set_output_df(op->output_df());
    return new_op;
}

static graphlib::Node* create_slice(
    graphlib::Graph* graph,
    graphlib::Node* op,
    graphlib::Shape operand_shape,
    int dim,
    int i,
    int factor,
    std::string new_op_name)
{
    auto slice = create_slice(op, operand_shape, dim, i, factor, new_op_name);
    if (not slice)
        return op;
    return graph->add_node(std::move(slice), graph->get_subgraph_id_for_node(op->id()));
}

static std::unique_ptr<graphlib::PyOpNode> create_gather(
    graphlib::Node* op, graphlib::Shape operand_shape, int dim, int factor, std::string new_op_name, std::uint32_t fracture_group_id)
{
    graphlib::OpType gather_op = (dim == NDSlice::k_dim)
                                     ? graphlib::OpType("add", {}, {})
                                     : graphlib::OpType("concatenate", {dim}, {});

    graphlib::Shape shape = operand_shape;
    if (gather_op.op == "concatenate")
    {
        shape[dim] *= factor;
    }

    auto gather = graphlib::create_node<graphlib::PyOpNode>(new_op_name, gather_op);
    gather->set_epoch_type(op->get_epoch_type());
    gather->set_output_df(op->output_df());
    gather->change_op_type(gather_op);
    gather->set_shape(shape);
    // tag the gather node with the fracture group id
    gather->as<graphlib::TaggedNode>()->tag("fracture_group_id", fracture_group_id);
    gather->as<graphlib::TaggedNode>()->tag("fracture_gather", true);
    gather->as<graphlib::TaggedNode>()->tag("dont_fuse", true);
    return gather;
}

// define a function to tag the top and bottom nodes of a fracture group
static void tag_top_bottom_nodes(graphlib::Graph* graph, std::vector<graphlib::NodeId> const& fractured_op_ids)
{
    // get a vector of nodes of interest
    std::vector<graphlib::Node*> nodes_of_interest;
    for (auto node_id : fractured_op_ids)
    {
        nodes_of_interest.push_back(graph->node_by_id(node_id));
    }

    // get the top row of the fractured nodes
    auto top_row = graphlib::top_row(graph, nodes_of_interest);

    // get the bottom row of the fractured nodes
    auto bottom_row = graphlib::bot_row(graph, nodes_of_interest);

    // tag each node in the top row as "top"
    for (auto node : top_row)
    {
        node->as<graphlib::TaggedNode>()->tag("fracture_top", true);
    }

    // tag each node in the bottom row as "bottom"
    for (auto node : bottom_row)
    {
       node->as<graphlib::TaggedNode>()->tag("fracture_bottom", true);
    }

}

static void assign_chip_ids(
    graphlib::Graph* graph,
    std::string const& op_name,
    FractureChipIdAssignments& chip_id_assignments,
    FractureChipIds const& chip_ids,
    FracturedNodes const& fractured_nodes)
{
    auto match = chip_ids.find(op_name);
    if (match == chip_ids.end())
        return;

    auto const& op_chip_ids = match->second;
    if (op_chip_ids.empty())
        return;

    auto const& [nd_slice, node_ids] = fractured_nodes;

    // gathers need to be on the same chip as their last operand,
    // for parity with scheduling
    if ((node_ids.size() == 1) and (graph->node_by_id(node_ids[0])->as<graphlib::TaggedNode>()->has_tag("fracture_gather")))
    {
        chip_id_assignments[graph->node_by_id(node_ids[0])->name()] = op_chip_ids.back();
        return;
    }


    for (std::size_t i = 0; i < node_ids.size(); ++i)
    {
        auto* op = graph->node_by_id(node_ids[i]);
        TT_ASSERT(chip_id_assignments.find(op->name()) == chip_id_assignments.end());
        int chip_id = op_chip_ids[i % op_chip_ids.size()];
        log_debug(LogFracture, "Assigning fractured node: {} chip_id: {}", op->name(), chip_id);
        chip_id_assignments[op->name()] = chip_id;
    }
}

static void assign_chip_ids(
    graphlib::Graph* graph,
    std::vector<graphlib::PyOpNode*> const& ops,
    FractureChipIdAssignments& chip_id_assignments,
    FractureChipIds const& chip_ids,
    std::unordered_map<graphlib::NodeId, FracturedNodes> const& node_to_fractured_nodes)
{
    if (chip_ids.empty())
        return;

    std::vector<graphlib::NodeId> fractured_op_ids;
    //
    // Assign all op's chip ids
    //
    for (auto const* op : ops)
    {
        auto const& fractured_nodes = node_to_fractured_nodes.at(op->id());
        assign_chip_ids(graph, op->name(), chip_id_assignments, chip_ids, fractured_nodes);
        fractured_op_ids.insert(fractured_op_ids.end(), fractured_nodes.second.begin(), fractured_nodes.second.end());
    }

    // tag the top and bottom rows of the fractured region
    tag_top_bottom_nodes(graph, fractured_op_ids);

    //
    // Generated scatter / gather ops inherit chip ids from original ops
    //
    for (bool visit_operands : {false, true})
    {
        std::vector<graphlib::NodeId> needs_visit = fractured_op_ids;
        while (not needs_visit.empty())
        {
            auto node_id = needs_visit.back();
            needs_visit.pop_back();

            auto* node = graph->node_by_id(node_id);
            auto match = chip_id_assignments.find(node->name());
            if (match == chip_id_assignments.end())
                continue;

            auto inherit_chip_id = match->second;

            // if visiting operands, no need to visit relations of top row
            if (visit_operands and node->as<graphlib::TaggedNode>()->has_tag("fracture_top"))
                continue;

            // if visiting users, no need to visit relations of bottom row
            if (not visit_operands and node->as<graphlib::TaggedNode>()->has_tag("fracture_bottom"))
                continue;

            auto relation = visit_operands ? graph->data_operands(node) : graph->data_users(node);
            for (auto* related : relation)
            {
                if (not dynamic_cast<graphlib::OpNode*>(related))
                    continue;
                if (chip_id_assignments.find(related->name()) != chip_id_assignments.end())
                    continue;

                // make sure all the operand nodes have chip assignments already
                // this is in line with scheduling, where an op receives a schedule only
                // when all its operands have been scheduled
                auto operands = graph->data_operands(related);
                bool all_operands_assigned = true;
                for (auto* operand : operands)
                {
                    // if operand is not in fractured op ids, no need to check it
                    if (std::find(fractured_op_ids.begin(), fractured_op_ids.end(), operand->id()) ==
                        fractured_op_ids.end())
                        continue;
                    if (chip_id_assignments.find(operand->name()) == chip_id_assignments.end())
                    {
                        log_debug(LogFracture, "Skipping chip_id assignment for fractured node: {}",
                                    related->name());
                        all_operands_assigned = false;
                        break;
                    }
                }
                if (not all_operands_assigned)
                    continue;

                log_debug(LogFracture, "Inherit chip_id[{}]: {} -> {}", inherit_chip_id, node->name(), related->name());
                chip_id_assignments[related->name()] = inherit_chip_id;
                needs_visit.push_back(related->id());
            }
        }
    }
}

static void assign_fracture_group_ids(
    graphlib::Graph* graph, std::uint32_t fracture_group_id, FracturedNodes const& fractured_nodes)
{
    auto const& [nd_slice, fractured_node_ids] = fractured_nodes;
    for (auto node_id : fractured_node_ids)
    {
        graph->node_by_id(node_id)->as<graphlib::TaggedNode>()->tag("fracture_group_id", fracture_group_id);
    }
}

static NDSlice create_matmul_nd_slice(NDSlice const& operand, int input_port_id)
{
    int inner_dim = input_port_id == 0 ? -1 : -2;
    return operand.replace_dim(inner_dim, NDSlice::k_dim);
}

static NDSlice create_matmul_nd_slice(NDSlice lhs, NDSlice rhs)
{
    TT_ASSERT(NDSlice::are_multiples(lhs.get_factor(-1), rhs.get_factor(-2)));
    TT_ASSERT(NDSlice::are_multiples(lhs.get_factor(-3), rhs.get_factor(-3)));

    lhs = create_matmul_nd_slice(lhs, 0);
    rhs = create_matmul_nd_slice(rhs, 1);

    TT_ASSERT(NDSlice::are_multiples(lhs.get_factor(NDSlice::k_dim), rhs.get_factor(NDSlice::k_dim)));

    auto higher_precidence = lhs.is_explicit() ? lhs : rhs;
    auto lower_precidence = lhs.is_explicit() ? rhs : lhs;

    std::vector<int> dims;
    std::vector<int> factors;
    for (auto dim : higher_precidence.get_dims())
    {
        int factor = higher_precidence.get_factor(dim);
        dims.push_back(dim);
        factors.push_back(factor);
    }

    for (auto dim : lower_precidence.get_dims())
    {
        bool rhs_already_inserted = std::find(dims.begin(), dims.end(), dim) != dims.end();
        if (rhs_already_inserted)
        {
            TT_ASSERT(NDSlice::are_multiples(higher_precidence.get_factor(dim), lower_precidence.get_factor(dim)));
            continue;
        }
        int factor = lhs.get_factor(dim);
        dims.push_back(dim);
        factors.push_back(factor);
    }

    return NDSlice(dims, factors, NDSlice::Spec::Inferred);
}

static NDSlice create_matmul_operand_nd_slice(NDSlice const& matmul, int input_port_id)
{
    int inner_dim = input_port_id == 0 ? -1 : -2;
    NDSlice ignore_opposite_outer_dim = matmul.remove_dim(inner_dim);
    return ignore_opposite_outer_dim.replace_dim(NDSlice::k_dim, inner_dim);
}

static NDSlice get_node_nd_slice(
    graphlib::Graph* graph,
    graphlib::Node* node,
    std::unordered_map<graphlib::NodeId, FracturedNodes> const& node_to_fractured_nodes,
    FractureGroup const& group)
{
    auto already_fractured = node_to_fractured_nodes.find(node->id());
    if (already_fractured != node_to_fractured_nodes.end())
    {
        return already_fractured->second.first;
    }

    auto group_lookup =
            std::find_if(group.begin(), group.end(), [node](auto const& f) { return node->name() == std::get<0>(f); });
    if (group_lookup == group.end())
    {
        // If not is not part of the group, return empty NDSlice, (e.g. we are gathering to this node)
        return NDSlice();
    }
    else
    {
        auto const& [name, dims, factors] = *group_lookup;
        // API (if specified) takes precedence
        if (not dims.empty())
            return NDSlice(dims, factors, NDSlice::Spec::Explicit, (int)node->shape().size());
    }

    std::vector<std::pair<int, NDSlice>> operand_nd_slice;
    for (auto operand : graph->operand_data_edges(node))
    {
        auto match = node_to_fractured_nodes.find(operand.producer_node_id);
        if (match == node_to_fractured_nodes.end())
            continue;

        operand_nd_slice.emplace_back(operand.consumer_input_port_id, match->second.first);
    }

    graphlib::OpNode* op = node->as<graphlib::OpNode>();
    NDSlice nd_slice;

    if (operand_nd_slice.empty())
    {
        log_fatal(LogFracture, "No fracture factors found for node: {}", node->name());
    }
    else if (op->is_matmul())
    {
        if (operand_nd_slice.size() == 1)
        {
            auto [port, operand] = operand_nd_slice.at(0);
            nd_slice = create_matmul_nd_slice(operand, port);
        }
        else
        {
            TT_ASSERT(operand_nd_slice.size() == 2);
            auto [in0_port, lhs] = operand_nd_slice.at(0);
            auto [in1_port, rhs] = operand_nd_slice.at(1);
            TT_ASSERT(in0_port == 0);
            TT_ASSERT(in1_port == 1);
            nd_slice = create_matmul_nd_slice(lhs, rhs);
        }
    }
    else
    {
        auto match = std::find_if(
            operand_nd_slice.begin(),
            operand_nd_slice.end(),
            [](auto const& operand_nd_slice) { return operand_nd_slice.second.is_explicit(); });

        // Take nd_slice that was explicitly specified, fallback to taking the first inferred
        nd_slice = (match != operand_nd_slice.end()) ? match->second : operand_nd_slice.front().second;
        for (auto const& [p, f] : operand_nd_slice)
        {
            TT_ASSERT(
                f.is_inferred() or f == nd_slice,
                "Explicitly specified Eltwise fracture factors should all be equivalent");
        }

        // This isn't matmul so always remove inner dim
        nd_slice = nd_slice.remove_dim(NDSlice::k_dim);
    }

    nd_slice.set_spec(NDSlice::Spec::Inferred);

    return nd_slice;
}

NDSlice get_operand_nd_slice(graphlib::Graph* graph, graphlib::Edge edge, NDSlice consumer_nd_slice)
{
    graphlib::OpNode* consumer = graph->node_by_id(edge.consumer_node_id)->as<graphlib::OpNode>();
    return consumer->is_matmul() ? create_matmul_operand_nd_slice(consumer_nd_slice, edge.consumer_input_port_id)
                                 : consumer_nd_slice;
}

static FracturedNodes gather(
    graphlib::Graph* graph,
    graphlib::PyOpNode* op,
    std::unordered_map<graphlib::NodeId, FracturedNodes> const& node_to_fractured_nodes,
    std::uint32_t fracture_group_id,
    NDSlice consumer_nd_slice = {})
{
    auto [nd_slice, fractured_ops] = node_to_fractured_nodes.at(op->id());
    auto gather_nd_slice = nd_slice / consumer_nd_slice;

    if (gather_nd_slice.empty())
        return std::make_pair(nd_slice, fractured_ops);

    log_debug(LogFracture, "  {:8} {:16} {}", "gather", op->name(), gather_nd_slice);
    for (auto dim : nd_slice.get_dims_reversed())
    {
        auto pre_gather = nd_slice;
        auto post_gather = nd_slice.replace_factor(dim, consumer_nd_slice.get_factor(dim));
        int factor = pre_gather.get_factor(dim);
        std::vector<graphlib::NodeId> gathered_ops;

        for (auto const& pre_gather_slice : pre_gather.get_slices())
        {
            auto post_gather_slice = pre_gather_slice.view(post_gather);
            auto gather_name = fractured_name(op, post_gather_slice.total_index, dim, "gather");
            graphlib::PyOpNode* gather_op =
                graph->has_node_with_name(gather_name)
                    ? graph->get_node_by_name(gather_name)->as<graphlib::PyOpNode>()
                    : graph->add_node(
                        create_gather(op, pre_gather.get_shape(op->shape()), dim, factor, gather_name, fracture_group_id),
                        graph->get_subgraph_id_for_node(op->id()))->as<graphlib::PyOpNode>();

            graph->add_edge(graphlib::Edge(
                fractured_ops.at(pre_gather_slice.total_index),
                0,
                gather_op->id(),
                pre_gather_slice.total_index,
                graphlib::EdgeType::kData));

            fractured_ops.at(pre_gather_slice.total_index) = gather_op->id();
            if (post_gather_slice.total_index >= (int)gathered_ops.size())
                gathered_ops.resize(post_gather_slice.total_index + 1);
            gathered_ops[post_gather_slice.total_index] = gather_op->id();
        }

        // Fixup nary add
        for (graphlib::NodeId& gather_op_id : gathered_ops)
        {
            graphlib::PyOpNode* gather_op = graph->node_by_id(gather_op_id)->as<graphlib::PyOpNode>();
            if (gather_op->op_name() == "add")
            {
                gather_op = graphlib::cascade_nary_to_binary_op(graph, gather_op)->as<graphlib::PyOpNode>();
                gather_op_id = gather_op->id();
            }

            graphlib::calculate_and_set_node_shape(graph, gather_op);
        }

        nd_slice = post_gather;
        fractured_ops = gathered_ops;
    }

    std::vector<graphlib::NodeId> consumer_fractured_ops;
    for (auto consumer_slice : consumer_nd_slice.get_slices())
    {
        TT_ASSERT(consumer_slice.total_index == (int)consumer_fractured_ops.size());
        consumer_fractured_ops.push_back(fractured_ops.at(consumer_slice.total_index));
    }

    return std::make_pair(consumer_nd_slice, consumer_fractured_ops);
}

static FracturedNodes scatter(
    graphlib::Graph* graph,
    graphlib::Edge edge,
    std::unordered_map<graphlib::NodeId, FracturedNodes> const& node_to_fractured_nodes,
    FractureGroup const& group)
{
    auto match = node_to_fractured_nodes.find(edge.producer_node_id);
    std::vector<graphlib::NodeId> fractured_ops = (match != node_to_fractured_nodes.end())
                                                      ? match->second.second
                                                      : std::vector<graphlib::NodeId>{edge.producer_node_id};
    graphlib::Node* producer = graph->node_by_id(edge.producer_node_id);
    graphlib::OpNode* consumer = graph->node_by_id(edge.consumer_node_id)->as<graphlib::OpNode>();
    NDSlice producer_nd_slice = (match != node_to_fractured_nodes.end()) ? match->second.first : NDSlice();
    NDSlice consumer_nd_slice = get_node_nd_slice(graph, consumer, node_to_fractured_nodes, group);
    NDSlice scatter_nd_slice = get_operand_nd_slice(graph, edge, consumer_nd_slice);

    if (match != node_to_fractured_nodes.end() and scatter_nd_slice != producer_nd_slice and
        not node_in_group(producer, group))
    {
        // This case means we are trying to scatter in a different way from the original input so we have to reset
        fractured_ops = std::vector<graphlib::NodeId>{edge.producer_node_id};
        producer_nd_slice = NDSlice();
    }
    else if (match != node_to_fractured_nodes.end() and (scatter_nd_slice / producer_nd_slice).empty())
    {
        // Already fully scattered
        return match->second;
    }

    log_debug(LogFracture, "  {:8} {:16} {}", "scatter", producer->name(), scatter_nd_slice);
    for (auto dim : scatter_nd_slice.get_dims())
    {
        auto pre_scatter = producer_nd_slice;
        auto post_scatter = scatter_nd_slice.trunc(dim);
        std::vector<graphlib::NodeId> scatter_ops;

        for (auto const& post_scatter_slice : post_scatter.get_slices())
        {
            auto pre_scatter_slice = post_scatter_slice.view(pre_scatter);
            TT_ASSERT(pre_scatter_slice.total_index < (int)fractured_ops.size());
            auto producer_id = fractured_ops.at(pre_scatter_slice.total_index);
            graphlib::Node* slice_op = graph->node_by_id(producer_id);

            for (auto [dim, index, factor] : post_scatter_slice.indices)
            {
                TT_ASSERT(dim != NDSlice::k_dim);
                TT_ASSERT(factor % producer_nd_slice.get_factor(dim) == 0);
                std::string slice_name = fractured_name(producer, index, dim, "scatter_" + consumer->name());
                int multiple = post_scatter.get_factor(dim) / pre_scatter.get_factor(dim);
                index %= multiple;
                factor /= producer_nd_slice.get_factor(dim);
                slice_op = graph->has_node_with_name(slice_name)
                               ? graph->get_node_by_name(slice_name)
                               : create_slice(graph, slice_op, slice_op->shape(), dim, index, factor, slice_name);
                if (slice_op->id() != producer_id)
                    graph->add_edge(graphlib::Edge(producer_id, 0, slice_op->id(), 0, graphlib::EdgeType::kData));
            }

            graphlib::calculate_and_set_node_shape(graph, slice_op);

            TT_ASSERT(post_scatter_slice.total_index == (int)scatter_ops.size());
            scatter_ops.push_back(slice_op->id());
        }

        fractured_ops = scatter_ops;
        producer_nd_slice = post_scatter;
    }

    return std::make_pair(scatter_nd_slice, fractured_ops);
}

FracturedNodes fracture_parameter(
    graphlib::Graph* graph, graphlib::InputNode* param, std::vector<int> dims, std::vector<int> factors)
{
    TT_ASSERT(is_parameter(param));
    TT_ASSERT(dims.size() == factors.size());
    TT_ASSERT(not dims.empty());

    NDSlice nd_slice(dims, factors, NDSlice::Spec::Explicit, (int)param->shape().size());
    std::vector<graphlib::NodeId> fractured_params;

    // Force create consteval graph for original input if didn't exist
    // The clone below will automatically clone the consteval graph too
    param->get_consteval_graph(graph, true, true);

    log_debug(LogFracture, "  {:8} {:16} {}", "param", param->name(), nd_slice);
    for (auto const& slice : nd_slice.get_slices())
    {
        graphlib::Node* slice_op = graph->add_node(
            param->clone(fractured_name(param, slice.total_index)), graph->get_subgraph_id_for_node(param->id()));

        if (param->as<graphlib::InputNode>()->is_parameter())
            slice_op->as<graphlib::InputNode>()->set_fractured_parameter_mapping(param->name());

        graphlib::ConstEvalGraph* consteval_graph = slice_op->as<graphlib::InputNode>()->get_consteval_graph();
        TT_ASSERT(consteval_graph);

        for (auto [dim, index, factor] : slice.indices)
        {
            if (dim == NDSlice::k_dim)
                log_fatal("Illegal dim specified for parameter input, k_dim only legal for matmul op types");
            std::string new_op_name = fractured_name(param, index, dim);
            consteval_graph->promote_node(create_slice(param, slice_op->shape(), dim, index, factor, new_op_name));
        }

        TT_ASSERT(slice.total_index == (int)fractured_params.size());
        fractured_params.push_back(slice_op->id());
    }

    return std::make_pair(nd_slice, fractured_params);
}

static FracturedNodes fracture_op(
    graphlib::Graph* graph,
    graphlib::PyOpNode* op,
    std::unordered_map<graphlib::NodeId, FracturedNodes> const& node_to_fractured_nodes,
    FractureGroup const& group)
{
    std::vector<graphlib::NodeId> fractured_ops;
    NDSlice nd_slice = get_node_nd_slice(graph, op, node_to_fractured_nodes, group);
    bool fractured_k = nd_slice.get_factor(NDSlice::k_dim) > 1;

    if (fractured_k and not op->is_matmul())
        log_fatal("Illegal dim specified for op type {}, k_dim only legal for matmul op types", op->op_name());

    log_debug(LogFracture, "  {:8} {:16} {}", "op", op->name(), nd_slice);
    for (auto const& slice : nd_slice.get_slices())
    {
        graphlib::Node* fractured_op = graph->add_node(
            op->clone(fractured_name(op, slice.total_index)), graph->get_subgraph_id_for_node(op->id()));
        fractured_op->as<graphlib::TaggedNode>()->tag("fracture_origin", op->name());

        for (auto operand : graph->operand_data_edges(op))
        {
            auto match = node_to_fractured_nodes.find(operand.producer_node_id);
            TT_ASSERT(match != node_to_fractured_nodes.end());
            auto const& [operand_nd_slice, operand_node_ids] = match->second;
            auto operand_slice = slice.operand_view(operand_nd_slice, operand.consumer_input_port_id, op->is_matmul());
            graphlib::Edge fractured_operand(
                operand_node_ids.at(operand_slice.total_index),
                0,
                fractured_op->id(),
                operand.consumer_input_port_id,
                graphlib::EdgeType::kData);
            graph->add_edge(fractured_operand);
        }

        graphlib::calculate_and_set_node_shape(graph, fractured_op);

        TT_ASSERT(slice.total_index == (int)fractured_ops.size());
        fractured_ops.push_back(fractured_op->id());
    }

    return std::make_pair(nd_slice, fractured_ops);
}

static FracturedNodes fracture_slice_stack_tm(
    graphlib::Graph* graph,
    graphlib::PyOpNode* op,
    std::unordered_map<graphlib::NodeId, FracturedNodes> const& node_to_fractured_nodes,
    FractureGroup const& group)
{
    graphlib::OpType nop_op_type = graphlib::OpType("nop", {}, {});
    NDSlice nd_slice = get_node_nd_slice(graph, op, node_to_fractured_nodes, group);
    graphlib::OpType tm_op_type = op->op_type();
    bool is_slice = op->op_name() == "hslice" or op->op_name() == "vslice";
    int tm_dim = tm_op_type.op[0] == 'h' ? -1 : -2;
    int& tm_factor = std::get<int>(tm_op_type.attr.at(0));

    int pre_dim = is_slice ? tm_dim : -3;
    int post_dim = is_slice ? -3 : tm_dim;
    int fracture_factor = nd_slice.get_factor(pre_dim);

    // Update the tm factor
    TT_ASSERT(tm_factor % fracture_factor == 0);
    tm_factor /= fracture_factor;

    // For stack we need to fixup the shape before it goes into generic op fracture
    // For slice we need to fixup the shape after it comes out of generic op fracture
    if (not is_slice)
    {
        auto shape = op->shape();
        shape[pre_dim] *= fracture_factor;
        shape[post_dim] /= fracture_factor;
        op->set_shape(shape);
    }

    // Swap out the op_type for a nop and fracture regularly
    op->change_op_type(nop_op_type);
    auto [result_nd_slice, fractured_ops] = fracture_op(graph, op, node_to_fractured_nodes, group);

    result_nd_slice = result_nd_slice.replace_dim(pre_dim, post_dim);

    for (auto id : fractured_ops)
    {
        graphlib::OpNode* fractured_op = graph->node_by_id(id)->as<graphlib::OpNode>();
        fractured_op->change_op_type(tm_op_type);
        graphlib::calculate_and_set_node_shape(graph, fractured_op);
    }

    return std::make_pair(result_nd_slice, fractured_ops);
}

static FracturedNodes fracture_tm(
    graphlib::Graph* graph,
    graphlib::PyOpNode* op,
    std::unordered_map<graphlib::NodeId, FracturedNodes> const& node_to_fractured_nodes,
    FractureGroup const& group)
{
    if (op->op_name() == "unsqueeze" or op->op_name() == "squeeze" or op->op_name() == "transpose")
    {
        auto [nd_slice, fractured_ops] = fracture_op(graph, op, node_to_fractured_nodes, group);
        if (op->op_name() == "transpose")
        {
            int dim0 = op->op_type().get_attr_as<int>("dim0");
            int dim1 = op->op_type().get_attr_as<int>("dim1");
            int factor0 = nd_slice.get_factor(dim0);
            int factor1 = nd_slice.get_factor(dim1);
            nd_slice = nd_slice.replace_factor(dim0, factor1);
            nd_slice = nd_slice.replace_factor(dim1, factor0);
        }
        return std::make_pair(nd_slice, fractured_ops);
    }
    else if (
        op->op_name() == "hslice" or op->op_name() == "vslice" or op->op_name() == "hstack" or
        op->op_name() == "vstack")
    {
        return fracture_slice_stack_tm(graph, op, node_to_fractured_nodes, group);
    }
    else if (op->op_name() == "select")
    {
        NDSlice nd_slice = get_node_nd_slice(graph, op, node_to_fractured_nodes, group);
        int select_dim = std::get<int>(op->op_attrs().at(0));
        if (nd_slice.get_factor(select_dim) != 1)
            log_fatal(
                "Op {}: Cannot fracture along select dimension[{}] factor[{}]",
                op->name(),
                select_dim,
                nd_slice.get_factor(select_dim));
        return fracture_op(graph, op, node_to_fractured_nodes, group);
    }
    else
    {
        log_fatal(LogFracture, "Unsupported op fracture type: {} {}", op->name(), op->op_name());
    }
}

static FracturedNodes fracture_writeback(
    graphlib::Graph* graph,
    graphlib::OutputNode* output,
    std::unordered_map<graphlib::NodeId, FracturedNodes> const& node_to_fractured_nodes,
    FractureGroup const&,
    std::uint32_t fracture_group_id)
{
    auto operands = graph->operand_data_edges(output);
    auto users = graph->user_edges(output);
    TT_ASSERT(operands.size() == 1);
    TT_ASSERT(users.size() == 1);
    TT_ASSERT(users.front().edge_type == graphlib::EdgeType::kPartialDataCopy);
    graphlib::Edge producer_edge = operands.front();
    graphlib::Edge consumer_edge = users.front();
    graphlib::Node* producer = graph->node_by_id(producer_edge.producer_node_id);
    graphlib::Node* consumer = graph->node_by_id(consumer_edge.consumer_node_id);

    if (node_to_fractured_nodes.find(producer->id()) == node_to_fractured_nodes.end())
        log_fatal(
            LogFracture,
            "Writeback nodes must have producer node fractured: {} -> {}",
            producer->name(),
            output->name());

    if (node_to_fractured_nodes.find(consumer->id()) == node_to_fractured_nodes.end())
        log_fatal(
            LogFracture,
            "Writeback nodes must have consumer node fractured: {} -> {}",
            output->name(),
            consumer->name());

    auto [producer_nd_slice, unused] = node_to_fractured_nodes.at(producer->id());
    auto [consumer_nd_slice, fractured_consumers] = node_to_fractured_nodes.at(consumer->id());
    auto [output_nd_slice, fractured_producers] =
        gather(graph, producer->as<graphlib::PyOpNode>(), node_to_fractured_nodes, fracture_group_id, consumer_nd_slice);
    TT_ASSERT(fractured_producers.size() == fractured_consumers.size());
    TT_ASSERT(output_nd_slice.volume() == (int)fractured_consumers.size());

    log_debug(LogFracture, "  {:8} {:16} {}", "writeback", output->name(), output_nd_slice);

    std::vector<graphlib::NodeId> fractured_outputs;
    for (auto slice : output_nd_slice.get_slices())
    {
        graphlib::OutputNode* fractured_output =
            graph->add_node(
                output->clone(fractured_name(output, slice.total_index)),
                graph->get_subgraph_id_for_node(output->id()))->as<graphlib::OutputNode>();
        fractured_output->set_shape(graph->node_by_id(fractured_producers.at(slice.total_index))->shape());
        handle_writeback_golden_transforms(output, fractured_output, slice);

        graph->add_edge(graphlib::Edge(
            fractured_producers.at(slice.total_index), 0, fractured_output->id(), 0, producer_edge.edge_type));
        graph->add_edge(graphlib::Edge(
            fractured_output->id(), 0, fractured_consumers.at(slice.total_index), 0, consumer_edge.edge_type));

        fractured_outputs.push_back(fractured_output->id());
    }

    return std::make_pair(output_nd_slice, fractured_outputs);
}

static void fracture_group(
    graphlib::Graph* graph,
    FractureGroup const& group,
    FractureChipIds const& chip_ids,
    FractureChipIdAssignments& chip_id_assignments,
    std::uint32_t fracture_group_id)
{
    std::unordered_map<graphlib::NodeId, FracturedNodes> node_to_fractured_nodes;
    auto group_nodes = get_group_nodes(graph, group);

    auto get_dims_factors = [&group](graphlib::Node* n)
    {
        auto match =
            std::find_if(group.begin(), group.end(), [n](auto const& f) { return n->name() == std::get<0>(f); });
        TT_LOG_ASSERT(match != group.end(), "Node not found in group {}", n->name());
        return *match;
    };

    //
    // Fracture parameters
    //
    std::vector<graphlib::InputNode*> params = graphlib::sorted<graphlib::InputNode>(graph, group_nodes, is_parameter);
    for (graphlib::InputNode* param : params)
    {
        auto const& [param_name, dims, factors] = get_dims_factors(param);
        TT_ASSERT(node_to_fractured_nodes.find(param->id()) == node_to_fractured_nodes.end());
        node_to_fractured_nodes[param->id()] = fracture_parameter(graph, param, dims, factors);
    }

    //
    // Scatter/Fracture ops
    //
    std::vector<graphlib::PyOpNode*> ops = graphlib::sorted<graphlib::PyOpNode>(graph, group_nodes);
    for (auto* op : ops)
    {
        for (auto operand : graph->operand_data_edges(op))
        {
            // Scatter
            auto scatter_nodes = scatter(graph, operand, node_to_fractured_nodes, group);
            node_to_fractured_nodes[operand.producer_node_id] = scatter_nodes;

            // Gather
            if (graphlib::PyOpNode* producer =
                    dynamic_cast<graphlib::PyOpNode*>(graph->node_by_id(operand.producer_node_id));
                producer)
            {
                NDSlice consumer_nd_slice = get_node_nd_slice(graph, op, node_to_fractured_nodes, group);
                NDSlice operand_nd_slice = get_operand_nd_slice(graph, operand, consumer_nd_slice);
                node_to_fractured_nodes[operand.producer_node_id] =
                    gather(graph, producer, node_to_fractured_nodes, fracture_group_id, operand_nd_slice);
            }
        }

        auto fractured_nodes = op->is_tm() ? fracture_tm(graph, op, node_to_fractured_nodes, group)
                                           : fracture_op(graph, op, node_to_fractured_nodes, group);
        node_to_fractured_nodes[op->id()] = fractured_nodes;
        assign_fracture_group_ids(graph, fracture_group_id, fractured_nodes);
    }

    //
    // Fracture writeback queues
    //
    assign_writeback_output_order(graph);
    std::vector<graphlib::OutputNode*> outputs = graphlib::sorted<graphlib::OutputNode>(graph, group_nodes);
    for (auto* output : outputs)
    {
        node_to_fractured_nodes[output->id()] = fracture_writeback(graph, output, node_to_fractured_nodes, group, fracture_group_id);
    }

    //
    // Gather non-fractured users
    //
    for (auto* op : ops)
    {
        for (auto user : graph->user_data_edges(op))
        {
            bool user_fractured = node_to_fractured_nodes.find(user.consumer_node_id) != node_to_fractured_nodes.end();
            if (user_fractured)
                continue;
            node_to_fractured_nodes[user.producer_node_id] = gather(graph, op, node_to_fractured_nodes, fracture_group_id);
            auto const& [nd_slice, fractured_node_ids] = node_to_fractured_nodes[user.producer_node_id];
            user.producer_node_id = fractured_node_ids.front();
            graph->add_edge(user);
        }
    }

    assign_chip_ids(graph, ops, chip_id_assignments, chip_ids, node_to_fractured_nodes);

    //
    // Cleanup
    //
    for (auto param : params)
        graph->remove_node(param);

    for (auto* op : ops)
        graph->remove_node(op);

    for (auto* output : outputs)
        graph->remove_node(output);
}

static void assert_valid_group(
    graphlib::Graph* graph,
    std::vector<std::string>& assert_non_overlapping_groups,
    FractureGroup const& inferred_group)
{
    assert_non_overlapping_groups.reserve(inferred_group.size());
    for (auto const& [name, dim, idx] : inferred_group)
    {
        auto* node = graph->get_node_by_name(name);
        auto* input = dynamic_cast<graphlib::InputNode*>(node);
        if (input and input->is_activation())
            log_fatal("Activation {} appears in fracture group, activations cannot be fractured", input->name());

        auto orig_name = node->as<graphlib::TaggedNode>()->tag_value_or<std::string>("fracture_origin", "");
        if (orig_name.empty())
            orig_name = name;
        if (std::find(assert_non_overlapping_groups.begin(), assert_non_overlapping_groups.end(), orig_name) !=
            assert_non_overlapping_groups.end())
            log_fatal("Node {} appears in multiple fracture groups, overlapping groups is not allowed", orig_name);
        assert_non_overlapping_groups.push_back(orig_name);
    }
}

FractureChipIdAssignments fracture(graphlib::Graph* graph, FractureGroups const& fracture_groups)
{
    std::vector<std::string> assert_non_overlapping_groups;
    FractureChipIdAssignments chip_id_assignments;
    std::uint32_t fracture_group_id = 0;

    for (auto const& [group, chip_ids] : fracture_groups)
    {
        auto expanded_group = expand_query_fracture_group(graph, group);
        auto inferred_group = infer_op_group(graph, expanded_group);

        log_debug(LogFracture, "Fracture group {}:", fracture_group_id);
        log_debug(LogFracture, "  - api group: {}", group);
        log_debug(LogFracture, "  - inferred group: {}", inferred_group);
        log_debug(LogFracture, "  - chip_ids: {}", chip_ids);

        assert_valid_group(graph, assert_non_overlapping_groups, inferred_group);

        fracture_group(graph, inferred_group, chip_ids, chip_id_assignments, fracture_group_id++);
    }
    return chip_id_assignments;
}
}  // namespace tt::passes
