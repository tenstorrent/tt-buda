// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/lowering_context.hpp"
#include "buda_passes.hpp"
#include "utils/assert.hpp"
#include "utils/logger.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "placer/dram.hpp"
#include "placer/utils.hpp"
#include "reportify/reportify.hpp"
#include "passes/decomposing_context.hpp"


namespace tt {

using NodeType = graphlib::NodeType;
using Edge = graphlib::Edge;
using EdgeType = graphlib::EdgeType;

Node *LoweringContext::get_or_insert_node(NodeContext node) {
    Node *new_node = nullptr;
    Node *old_node = old_graph->node_by_id(node.id);

    auto match = old_to_new.find(old_node);
    if (match == old_to_new.end()) {
        TT_ASSERT(old_node->node_type() != graphlib::NodeType::kPyOp);
        new_node = lower_queue(old_graph, new_graph, old_node, old_to_new);
    } else {
        new_node = match->second;
    }

    return new_node;
}

// Insert new op in lowered graph, from python
template <typename NodeT>
NodeT *LoweringContext::lower_node(graphlib::OpType const &op_type, std::vector<NodeContext> const &operands)
{
    std::string suffix;
    if (op_index > 0) {
        suffix = ".lc" + std::to_string(op_index);
    }
    NodeT *new_node = new_graph->add_node(
        graphlib::create_node<NodeT>(node->name() + suffix, op_type), subgraph_idx);

    new_node->copy_lowered_op_attributes(node);
    old_to_new.insert_or_assign(node, new_node);

    std::vector<Node *> new_operands;
    for (const NodeContext &op : operands) 
    {
        // Already a new-graph op
        if (new_graph->has_node_with_id(op.id)) {
            new_operands.push_back(new_graph->node_by_id(op.id));
            continue;
        }
            
        // Old op, translate
        new_operands.push_back(get_or_insert_node(op));
    }

    for (std::size_t i = 0; i < new_operands.size(); i++)
    {
        Edge new_edge = Edge(new_operands[i]->id(), operands[i].output_index, new_node->id(), i, graphlib::EdgeType::kData);
        new_graph->add_edge(new_edge);

        // check if this is a new node, and skip as there are no edges to copy
        if (new_graph->has_node_with_id(operands[i].id))
            continue;


        // Find the old edge, and copy over attributes
        for (Edge old_edge : old_graph->operand_data_edges(node)) 
        {
            if ( (old_edge.producer_node_id == operands[i].id) && 
                 (old_edge.producer_output_port_id == operands[i].output_index) )
            {
                std::shared_ptr<graphlib::EdgeAttributes> new_attr = new_graph->get_edge_attributes(new_edge);
                lower_edge_tms(old_graph, old_edge, new_attr);
                break;
            }
        }
    }

    copy_operand_edges_to_new_graph(old_graph, new_graph, node, new_node, old_to_new, true /* control only */);

    op_index++;
    return new_node;
}

NodeContext LoweringContext::op(
    graphlib::OpType const &op_type,
    std::vector<NodeContext> const &operands,
    std::string const &tag,
    int tile_height,
    int tile_width)
{
    Node *lowered_node = lower_node<graphlib::BudaOpNode>(op_type, operands);
    TileDim target_tile_dim = graphlib::get_tile_dim_from_height_width(tile_height, tile_width);

    lowered_node->set_tile_dim(target_tile_dim); // propagate TileDim to buda op
    if (tag != "")
        lowered_node->as<graphlib::TaggedNode>()->tag(tag);

    return NodeContext(lowered_node);
}

NodeContext LoweringContext::nary_tm(graphlib::OpType const &op_type, std::vector<NodeContext> const &operands)
{
    graphlib::BudaNaryTMNode *tm = lower_node<graphlib::BudaNaryTMNode>(op_type, operands);

    graphlib::OpType nop_op_type = graphlib::OpType("nop", {}, {});
    graphlib::BudaOpNode *nop = lower_node<graphlib::BudaOpNode>(nop_op_type, {tm});
    return NodeContext(nop);
}

// Insert new tm in lowered graph, from python
NodeContext LoweringContext::tm(graphlib::OpType const &tm_op_type, NodeContext const &operand)
{
    graphlib::OpType nop_op_type = graphlib::OpType("nop", {}, {});
    graphlib::BudaOpNode *nop = lower_node<graphlib::BudaOpNode>(nop_op_type, {operand});

    std::vector<graphlib::Edge> operands_edges = new_graph->operand_data_edges(nop);
    TT_ASSERT(operands_edges.size() == 1);
    graphlib::Edge edge = operands_edges[0];
    new_graph->get_edge_attributes(edge)->append_tm(tm_op_type);

    return NodeContext(nop);
}

void LoweringContext::set_output_df(NodeContext node, DataFormat df) { get_or_insert_node(node)->set_output_df(df); }

void LoweringContext::set_runtime_tensor_transform(NodeContext node, graphlib::RuntimeTensorTransform t)
{
    get_or_insert_node(node)->as<graphlib::InputNode>()->set_runtime_tensor_transform(t);
}

void LoweringContext::set_broadcast_dim(NodeContext src, NodeContext dest, int dim, int factor, bool explicit_bcast)
{
    for (Edge e : new_graph->user_data_edges(new_graph->node_by_id(src.id))) {
        if (e.producer_output_port_id != (std::uint32_t)src.output_index)
            continue;

        if (e.consumer_node_id != dest.id)
            continue;

        TT_ASSERT(dim >= 0, "Dimension in broadcast must be positive");
        new_graph->get_edge_attributes(e)->set_broadcast_dim(dim, factor, explicit_bcast);
    }

}

std::vector<std::uint32_t> LoweringContext::shape(NodeContext node, bool use_new_graph) const
{
    Graph *graph = use_new_graph ? new_graph : old_graph;
    return graph->node_by_id(node.id)->shape().as_vector();
}

std::vector<std::uint32_t> LoweringContext::pybuda_shape() const
{
    return node->shape().as_vector();
}

NodeContext LoweringContext::constant(float value, std::pair<int, int> rc_dims)
{
    auto new_node = new_graph->add_node(
        graphlib::create_node<graphlib::ConstantInputNode>(
            "input_constant" + std::to_string(constant_index++) + "_" + node->name(),
            value,
            rc_dims.first,
            rc_dims.second),
        subgraph_idx);

    new_node->set_shape(graphlib::Shape::single_tile());
    new_node->set_output_df(node->output_df());
    new_node->as<graphlib::TaggedNode>()->add_tags(this->node->as<graphlib::TaggedNode>()->get_tags());

    return NodeContext(new_node);
}

NodeContext LoweringContext::constant_tile(std::vector<float> value)
{
    auto new_node = new_graph->add_node(graphlib::create_node<graphlib::ConstantInputNode>(
        "input_constant" + std::to_string(constant_index++) + "_" + node->name(), value), subgraph_idx);

    new_node->set_shape(graphlib::Shape::single_tile());
    new_node->set_output_df(node->output_df());
    new_node->as<graphlib::TaggedNode>()->add_tags(this->node->as<graphlib::TaggedNode>()->get_tags());

    return NodeContext(new_node);
}

NodeContext LoweringContext::tensor(std::shared_ptr<void> tensor, graphlib::Shape shape, DataFormat df)
{
    auto new_node = new_graph->add_node(graphlib::create_node<graphlib::ConstantInputNode>(
        "lc.input_tensor." + node->name() + "." + std::to_string(this->op_index++), tensor, shape), subgraph_idx);
    new_node->set_shape(graphlib::Shape::create_buda(shape.as_vector()));
    new_node->set_output_df((df != DataFormat::Invalid) ? df : node->output_df());
    new_node->as<graphlib::TaggedNode>()->add_tags(this->node->as<graphlib::TaggedNode>()->get_tags());
    return NodeContext(new_node);
}

NodeContext LoweringContext::tensor_with_blob(
    std::shared_ptr<void> tensor, graphlib::Shape shape, sparse::SparseBUDA sparse_buda, DataFormat df)
{
    NodeContext nc = LoweringContext::tensor(tensor, shape, df);
    new_graph->node_by_id(nc.id)->as<tt::graphlib::ConstantInputNode>()->set_sparse_buda(sparse_buda);
    return nc;
}

bool requires_lowering_to_ram(Node* node) {
    using graphlib::InputNodeType;

    if (node->node_type() == NodeType::kInput) {
        InputNodeType input_type = node->as<graphlib::InputNode>()->input_type();
        return (input_type == InputNodeType::Parameter) or (input_type == InputNodeType::OptimizerParameter);
    } else if (node->node_type() == NodeType::kQueue) {
        return node->as<graphlib::QueueNode>()->is_grad_accumulator();
    }
    return false;
}

int calculate_tile_size(int val)
{
    // We might not even care about large dim size 
    // that are not divisible by 32
    if (val > 32)
        return 32;

    int smallest_pad = 31;
    int current_tile_size = 32;

    std::vector<int> tile_sizes = {32, 16, 8, 4, 2, 1};

    for (auto tile_size_ : tile_sizes)
    {
        int rem = val % tile_size_;
        int pad = tile_size_ - rem;
        if (rem == 0 and smallest_pad != 0) {
            // Pick the largest tile size that divides evenly
            smallest_pad = 0;
            current_tile_size = tile_size_;
        } else if (pad <= smallest_pad) {
            // pick the tile size with smallest pad
            smallest_pad = pad;
            current_tile_size = tile_size_;
        }
    }
    return current_tile_size;
}

Node *lower_queue(Graph *old_graph, Graph *new_graph, Node *old_node, NodeToNodeMap &old_to_new) {
    Node *new_node = new_graph->add_node(old_node->clone(), old_graph->get_subgraph_id_for_node(old_node->id()));

    if (env_as<bool>("PYBUDA_ENABLE_TINY_TILE")) {
        graphlib::Shape shape = old_node->shape();
        shape = shape.canonical();
        int tile_size_r = calculate_tile_size(shape[-2]); 
        int tile_size_c = graphlib::Shape::BUDA_TILE_DIM; // Force Column to 32 for now
        auto calculated_tile_dim = graphlib::get_tile_dim_from_height_width(tile_size_r, tile_size_c);
        old_node->set_tile_dim(calculated_tile_dim);
    }
    new_node->set_shape(graphlib::Shape::to_buda(old_node->shape()));

    if (requires_lowering_to_ram(old_node)) {
        new_node->as<graphlib::QueueNode>()->set_memory_access_type(graphlib::MemoryAccessType::RAM);
    }

    copy_operand_edges_to_new_graph(old_graph, new_graph, old_node, new_node, old_to_new);
    old_to_new.insert(std::make_pair(old_node, new_node));

    // Output queues should inherit their df from producer
    if (new_node->node_type() == NodeType::kOutput) {
        auto operands = new_graph->data_operands(new_node);
        TT_ASSERT(operands.size() == 1);
        Node *producer = operands[0];
        new_node->set_output_df(producer->output_df());
    }

    if (graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(new_node))
    {
        graphlib::RuntimeTensorTransform runtime_tensor_transform = input->get_runtime_tensor_transform();

        bool buda_shape_already_reinterpret =
            runtime_tensor_transform.type == graphlib::RuntimeTensorTransformType::ReinterpretShape &&
            runtime_tensor_transform.original_shape.is_valid() and
            runtime_tensor_transform.original_shape == new_node->shape();
        if (buda_shape_already_reinterpret)
        {
            // clear the reinterpret shape
            input->set_runtime_tensor_transform(graphlib::RuntimeTensorTransform());
        }
    }

    if (graphlib::OutputNode *output = dynamic_cast<graphlib::OutputNode *>(new_node))
    {
        graphlib::RuntimeTensorTransform runtime_tensor_transform = output->get_runtime_tensor_transform();

        bool buda_shape_already_reinterpret =
            runtime_tensor_transform.type == graphlib::RuntimeTensorTransformType::ReinterpretShape &&
            runtime_tensor_transform.original_shape.is_valid() and
            runtime_tensor_transform.reinterpreted_shape == new_node->shape();
        if (buda_shape_already_reinterpret)
        {
            // clear the reinterpret shape
            output->set_runtime_tensor_transform(graphlib::RuntimeTensorTransform());
        }
    }

    //
    // WA for backend/golden issue which doesn't handle ops that format convert.  This is especially exposed
    // since we will demote F32 ops to F16b, so this workaround also demotes inputs of F32 to F16b which
    // enables our current test suite to pass.
    //
    // tenstorrent/budabackend#274
    //
    /*if (new_node->node_type() == NodeType::kInput and new_node->output_df() == DataFormat::Float32) {
        new_node->set_output_df(DataFormat::Float16_b);
        log_warning(
            LogGraphCompiler,
            "Demoting f32 input to f16b tenstorrent/budabackend#274");
    }*/
    return new_node;
}

// Use python + lowering context to convert PyBuda node to Buda node
void lower_node(const LoweringContext &lc)
{
    graphlib::PyOpNode *node = lc.get_node();
    graphlib::OpType type = node->op_type();
    auto eval_module = py::module_::import("pybuda.op.eval.pybuda");
    py::function pybuda_lower = eval_module.attr("get_f_pybuda_lower")(type);

    std::vector<NodeContext> inputs;
    for(Edge opedge : lc.get_old_graph()->operand_data_edges(node))
    {
        inputs.push_back(
            NodeContext(lc.get_old_graph()->node_by_id(opedge.producer_node_id), opedge.producer_output_port_id));
        inputs.back().shape =
            node->shape_of_operand(lc.get_old_graph(), lc.get_old_graph()->node_by_id(opedge.producer_node_id));
        inputs.back().unbroadcast_shape =
            node->shape_of_operand(lc.get_old_graph(), lc.get_old_graph()->node_by_id(opedge.producer_node_id), true);
    }

    std::vector<NodeContext> outputs;
    for(Edge user_edge : lc.get_old_graph()->user_data_edges(node))
    {
        outputs.push_back(
            NodeContext(lc.get_old_graph()->node_by_id(user_edge.consumer_node_id), user_edge.consumer_input_port_id));
        outputs.back().shape = lc.get_old_graph()->node_by_id(user_edge.consumer_node_id)->shape();
    }

    pybuda_lower(lc, inputs, outputs);
    
}

// Copy incoming edges from old graph to new node on new graph after lowering
void copy_operand_edges_to_new_graph(
        Graph *old_graph,
        Graph *new_graph,
        Node *old_node,
        Node *new_node,
        const NodeToNodeMap &old_to_new,
        bool control_only,
        bool loopback_only)
{
    for (Edge edge : old_graph->operand_edges(old_node))
    {
        if (control_only && (edge.edge_type == graphlib::EdgeType::kData))
            continue;

        if (loopback_only != (edge.edge_type == graphlib::EdgeType::kDataLoopback or edge.edge_type == graphlib::EdgeType::kPartialDataCopy))
            continue;

        try {
            Node *operand = old_to_new.at(old_graph->node_by_id(edge.producer_node_id));
            Edge new_edge = Edge(operand->id(), edge.producer_output_port_id, new_node->id(), edge.consumer_input_port_id, edge.edge_type);
            new_graph->add_edge(new_edge);

            std::shared_ptr<graphlib::EdgeAttributes> new_attr = new_graph->get_edge_attributes(new_edge);
            lower_edge_tms(old_graph, edge, new_attr);
            //new_graph->copy_edge_attributes(edge, new_edge, old_graph);
            
        } catch (std::out_of_range &e) {
            log_fatal("Input operand not mapped to new graph during lowering: {}", 
                    old_graph->node_by_id(edge.producer_node_id)->name());
        }
    }
}

void lower_edge_tms(Graph *old_graph, Edge &old_edge, std::shared_ptr<graphlib::EdgeAttributes> new_attr)
{
    // Broadcasts were in the original dimensions, so we need to convert to 4d buda
    std::vector<graphlib::OpType> old_tms = old_graph->get_edge_attributes(old_edge)->get_tms();

    for (const graphlib::OpType &tm : old_tms)
    {
        // Handle delta calculation for producers that are greater then 4D. For 4D shapes
        // and below, we need to account for 4 dimensions to match the Buda expectations.
        int delta = 0;
        int producer_rank = old_graph->node_by_id(old_edge.producer_node_id)->shape().as_vector().size();
        if (producer_rank <= 4) {
            delta = 4 - producer_rank;
            producer_rank = 4;
        }

        auto new_tm = graphlib::OpType(tm);

        // If TM attr is referenced backwards (negative indexing), directly convert to positive axis.
        if (std::get<int>(new_tm.attr[0]) < 0) {
            std::get<int>(new_tm.attr[0]) += producer_rank;
        } else {
            std::get<int>(new_tm.attr[0]) += delta;
        }

        if ( (std::get<int>(new_tm.attr[0]) >= 2) && (std::get<int>(new_tm.attr[1]) % graphlib::Shape::BUDA_TILE_DIM != 0) )
        {
            // snap up to tile dim
            std::get<int>(new_tm.attr[1]) += graphlib::Shape::BUDA_TILE_DIM - (std::get<int>(new_tm.attr[1]) % graphlib::Shape::BUDA_TILE_DIM);
        }
        //std::cout << " to " << new_tm.attr[0] << ", " << new_tm.attr[1] << std::endl;

        if (tm.op == "broadcast" and std::get<int>(new_tm.attr[0]) >= 2)
        {
            std::get<int>(new_tm.attr[1]) /= graphlib::Shape::BUDA_TILE_DIM;
        }

        TT_ASSERT(std::get<int>(new_tm.attr[0]) >= 0 && std::get<int>(new_tm.attr[0]) <= 3, "Invalid broadcast dim after lowering");
        new_attr->append_tm(new_tm);
    }
}


} // namespace tt
