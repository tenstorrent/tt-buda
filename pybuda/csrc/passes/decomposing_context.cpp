// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "decomposing_context.hpp"
#include "buda_passes.hpp"
#include "utils/assert.hpp"
#include "utils/logger.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "placer/dram.hpp"
#include "placer/utils.hpp"
#include "reportify/reportify.hpp"


namespace tt {

using NodeType = graphlib::NodeType;
using Edge = graphlib::Edge;
using EdgeType = graphlib::EdgeType;


// TODO: move tags to a vector of enums
NodeContext DecomposingContext::op(
    graphlib::OpType const &op_type,
    std::vector<NodeContext> const &operands,
    bool copy_tms,
    bool dont_decompose,
    bool optimize_hoist,
    DataFormat output_df)
{
    std::string suffix = ".dc." + op_type.op + "." + std::to_string(op_index);

    graphlib::PyOpNode *new_node =
        this->graph->add_node(graphlib::create_node<graphlib::PyOpNode>(this->node_->name() + suffix, op_type), subgraph_idx);

    if (dont_decompose) {
        new_node->as<graphlib::TaggedNode>()->tag("dont_decompose");
    }

    if (optimize_hoist) {
        new_node->as<graphlib::TaggedNode>()->tag("optimize_hoist");
    }

    // record the original op {name, type} before decomposition as tags
    new_node->as<graphlib::TaggedNode>()->add_tags(
        this->node_->as<graphlib::TaggedNode>()->get_tags());

    new_node->set_epoch_type(this->node_->get_epoch_type());
    if (output_df == DataFormat::Invalid) {
        new_node->set_output_df(this->node_->output_df());
    } else {
        new_node->set_output_df(output_df);
    }
    new_node->set_golden_transforms(this->node_->get_golden_transforms());

    py::module_ eval_module = py::module_::import("pybuda.op.eval.pybuda");
    py::function pybuda_shape = eval_module.attr("get_f_pybuda_shape")(op_type);
    std::vector<std::vector<std::uint32_t>> operand_tuples;
    for (NodeContext const &op_node : operands) operand_tuples.push_back(op_node.shape.as_vector());
    py::tuple ret = pybuda_shape(operand_tuples);
    graphlib::Shape shape = graphlib::Shape::create(ret[0].cast<std::vector<std::uint32_t>>());
    std::vector<graphlib::DimBroadcast> broadcasts = ret[1].cast<std::vector<graphlib::DimBroadcast>>();

    new_node->set_shape(shape);

    for (int i = 0; i < (int)operands.size(); i++) {

        graphlib::Node *current_node = this->graph->node_by_id(operands[i].id);

        if (new_node->get_epoch_type() != current_node->get_epoch_type()) {

            TT_ASSERT((current_node->get_epoch_type() == NodeEpochType::Forward and new_node->get_epoch_type() == NodeEpochType::Backward) or
                      (new_node->get_epoch_type() == NodeEpochType::Optimizer));

            graphlib::Edge edge(current_node->id(), 0, new_node->id(), 0, graphlib::EdgeType::kAutogradFwdToBwd);
            graph->add_edge(edge);

        }

        graphlib::Edge edge(operands[i].id, (graphlib::PortId)0, new_node->id(), (graphlib::PortId)i, graphlib::EdgeType::kData);
        graph->add_edge(edge);

        if (copy_tms) {
            for(Edge op_edge : graph->operand_data_edges(this->node_)) {
                if (op_edge.producer_node_id == operands[i].id) {
                    graph->get_edge_attributes(edge)->set_tms(graph->get_edge_attributes(op_edge)->get_tms());
                }
            }
        }
        for (graphlib::DimBroadcast broadcast : broadcasts) {
            if (i == std::get<0>(broadcast)) {
                std::shared_ptr<graphlib::EdgeAttributes> attr = graph->get_edge_attributes(edge);
                attr->set_broadcast_dim(std::get<1>(broadcast), std::get<2>(broadcast));
            }
        }
    }

    inserted_nodes.push_back(new_node);

    this->op_index++;
    return NodeContext(new_node);
}

// Fuse provided operand with output of node being decomposed
void DecomposingContext::fuse(NodeContext operand, graphlib::PortId producer_out_port) {
    output_node_id = operand.id;
    
    // Inherit gradient op from original node
    graph->node_by_id(output_node_id)->as<graphlib::PyOpNode>()->set_gradient_op(node_->is_gradient_op());


    // Map operand control edges
    for (Edge in_edge : graph->operand_edges(node_)) {
        if (in_edge.edge_type == EdgeType::kData)
            continue;

        if (in_edge.edge_type == EdgeType::kAutogradFwdToGradient)
            continue;

        TT_ASSERT(
            in_edge.edge_type != EdgeType::kControl or in_edge.edge_type != EdgeType::kDataLoopback or
            in_edge.edge_type != EdgeType::kControlLoop);

        for (graphlib::PyOpNode *inserted_node : inserted_nodes) {
            Edge new_in_edge(
                in_edge.producer_node_id, in_edge.producer_output_port_id, inserted_node->id(), 0, in_edge.edge_type);
            this->graph->add_edge(new_in_edge);
        }
    }

    for (Edge in_edge : graph->operand_edges(node_)) {
        if (in_edge.edge_type != EdgeType::kAutogradFwdToGradient)
            continue;

        Edge new_in_edge(in_edge.producer_node_id, in_edge.producer_output_port_id, operand.id, 0, in_edge.edge_type);
        this->graph->add_edge(new_in_edge);
        break;
    }

    for (Edge out_edge : graph->user_edges(node_)) {
        Edge new_out_edge(
            operand.id,
            producer_out_port,
            out_edge.consumer_node_id,
            out_edge.consumer_input_port_id,
            out_edge.edge_type);
        this->graph->add_edge(new_out_edge);
        this->graph->get_edge_attributes(new_out_edge)->set_tms(graph->get_edge_attributes(out_edge)->get_tms());
    }
}

NodeContext DecomposingContext::tensor(std::shared_ptr<void> tensor, graphlib::Shape tensor_shape, DataFormat df)
{
    auto new_node = graph->add_node(graphlib::create_node<graphlib::ConstantInputNode>(
        "dc.input_tensor." + this->node_->name() + "." + std::to_string(this->op_index), tensor, tensor_shape), subgraph_idx);
    new_node->set_shape(tensor_shape);

    new_node->set_output_df((df != DataFormat::Invalid) ? df : this->node_->output_df());
    new_node->set_epoch_type(this->node_->get_epoch_type());
    this->op_index++;

    return NodeContext(new_node);
}

std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>> decompose_pybuda_graph(
    Graph *graph, const char *dispatcher_name, std::shared_ptr<void> compiler_cfg)
{
    std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>> inserted_node_id_mapping;
    py::module_ eval_module = py::module_::import("pybuda.op.eval.pybuda");
    uint32_t nodes_removed = 1;
    while(nodes_removed)
    {
        nodes_removed = 0;

        for (graphlib::Node *node : graphlib::topological_sort(*graph))
        {

            if (node->node_type() != graphlib::NodeType::kPyOp)
                continue;

            graphlib::PyOpNode* py_node = node->as<graphlib::PyOpNode>();

            graphlib::OpType type = py_node->op_type();
            if (py_node->as<graphlib::TaggedNode>()->has_tag("dont_decompose")) {
                continue;
            }

            py::function pybuda_decompose = eval_module.attr(dispatcher_name)(type);

            std::vector<NodeContext> inputs;
            for(Edge op_edge : graph->operand_data_edges(node)) {
                inputs.push_back(NodeContext(graph->node_by_id(op_edge.producer_node_id), op_edge.producer_output_port_id));
                inputs.back().shape = py_node->shape_of_operand(graph, graph->node_by_id(op_edge.producer_node_id));
                inputs.back().unbroadcast_shape = py_node->shape_of_operand(graph, graph->node_by_id(op_edge.producer_node_id), true);
            }

            DecomposingContext dc(graph, py_node, compiler_cfg);

            log_trace(LogGraphCompiler, "Decomposing {}", node->name());
            pybuda_decompose(&dc, inputs);

            if (dc.get_op_index() == 0) {
                // No ops were added
                continue;
            }

            inserted_node_id_mapping.push_back({dc.get_output_node_id(), node->id()});

            // Remove node that was decomposed from graph
            auto operands = graph->data_operands(node);
            graph->remove_node(node);

            // Remove any dangling operands
            for (auto const *operand : operands)
            {
                if (graph->data_users(operand).empty())
                {
                    graph->remove_node(operand);
                }
            }

            nodes_removed++;
        }
    }

    // Fixup changes in rank after decomp
    for (auto [output_node_id, _] : inserted_node_id_mapping)
    {
        if (not graph->has_node_with_id(output_node_id))
            continue;
        for (graphlib::Edge edge : graph->user_data_edges(graph->node_by_id(output_node_id)))
            handle_change_rank(graph, edge);
    }

    return inserted_node_id_mapping;
}

} // namespace tt
