// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "graph_lib/node_types.hpp"

#include <memory>
#include <vector>
#include <string>
#include <tuple>

#include "utils/assert.hpp"
#include "graph_lib/utils.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/python_bindings.hpp"

namespace tt {

namespace graphlib {

template<> const TaggedNode* Node::as<TaggedNode>() const
{
    const TaggedNode* tagged_node = dynamic_cast<TaggedNode const *>(this);
    TT_ASSERT(tagged_node != nullptr);
    return tagged_node;
}
template<> const OpNode* Node::as<OpNode>() const
{
    TT_ASSERT(this->node_type() == NodeType::kPyOp || this->node_type() == NodeType::kBudaOp);
    return dynamic_cast<OpNode const *>(this);
}

template <>
const PyOpNode *Node::as<PyOpNode>() const
{
    TT_ASSERT(this->node_type() == NodeType::kPyOp);
    return dynamic_cast<PyOpNode const *>(this);
}

template <>
const BudaOpNode *Node::as<BudaOpNode>() const
{
    TT_ASSERT(this->node_type() == NodeType::kBudaOp);
    return dynamic_cast<BudaOpNode const *>(this);
}

template <>
const BudaNaryTMNode *Node::as<BudaNaryTMNode>() const
{
    TT_ASSERT(this->node_type() == NodeType::kBudaNaryTM);
    return dynamic_cast<BudaNaryTMNode const *>(this);
}

template<> TaggedNode* Node::as<TaggedNode>()
{
    TaggedNode* tagged_node = dynamic_cast<TaggedNode *>(this);
    TT_ASSERT(tagged_node != nullptr);
    return dynamic_cast<TaggedNode *>(this);
}

template <>
OpNode *Node::as<OpNode>()
{
    TT_ASSERT(this->node_type() == NodeType::kPyOp || this->node_type() == NodeType::kBudaOp);
    return dynamic_cast<OpNode *>(this);
}

template <>
PyOpNode *Node::as<PyOpNode>()
{
    TT_ASSERT(this->node_type() == NodeType::kPyOp);
    return dynamic_cast<PyOpNode *>(this);
}

template <>
BudaOpNode *Node::as<BudaOpNode>()
{
    TT_ASSERT(this->node_type() == NodeType::kBudaOp);
    return dynamic_cast<BudaOpNode *>(this);
}

template <>
BudaNaryTMNode *Node::as<BudaNaryTMNode>()
{
    TT_ASSERT(this->node_type() == NodeType::kBudaNaryTM);
    return dynamic_cast<BudaNaryTMNode *>(this);
}

template<> InputNode* Node::as<InputNode>()
{
    TT_ASSERT(this->node_type() == NodeType::kInput);
    return dynamic_cast<InputNode *>(this);
}

template<> const InputNode* Node::as<InputNode>() const
{
    TT_ASSERT(this->node_type() == NodeType::kInput);
    return dynamic_cast<InputNode const *>(this);
}

template<> ConstantInputNode* Node::as<ConstantInputNode>()
{
    TT_ASSERT(this->as<InputNode>()->is_constant());
    return dynamic_cast<ConstantInputNode *>(this);
}

template<> const ConstantInputNode* Node::as<ConstantInputNode>() const
{
    TT_ASSERT(this->as<InputNode>()->is_constant());
    return dynamic_cast<ConstantInputNode const *>(this);
}

template<> OutputNode* Node::as<OutputNode>()
{
    TT_ASSERT(this->node_type() == NodeType::kOutput);
    return dynamic_cast<OutputNode *>(this);
}

template<> const OutputNode* Node::as<OutputNode>() const
{
    TT_ASSERT(this->node_type() == NodeType::kOutput);
    return dynamic_cast<OutputNode const *>(this);
}

template<> QueueNode* Node::as<QueueNode>()
{
    TT_ASSERT(
        (this->node_type() == NodeType::kQueue) || 
        (this->node_type() == NodeType::kInput) || 
        (this->node_type() == NodeType::kOutput));
    return dynamic_cast<QueueNode *>(this);
}

template<> const QueueNode* Node::as<QueueNode>() const
{
    TT_ASSERT(
        (this->node_type() == NodeType::kQueue) || 
        (this->node_type() == NodeType::kInput) || 
        (this->node_type() == NodeType::kOutput));
    return dynamic_cast<QueueNode const *>(this);
}

template<> EpochToEpochQueueNode* Node::as<EpochToEpochQueueNode>()
{
    TT_ASSERT(this->node_type() == NodeType::kQueue);
    TT_ASSERT(this->as<QueueNode>()->is_epoch_to_epoch());
    return dynamic_cast<EpochToEpochQueueNode *>(this);
}

template<> const EpochToEpochQueueNode* Node::as<EpochToEpochQueueNode>() const
{
    TT_ASSERT(this->node_type() == NodeType::kQueue);
    TT_ASSERT(this->as<QueueNode>()->is_epoch_to_epoch());
    return dynamic_cast<EpochToEpochQueueNode const *>(this);
}


template<> BufferingQueueNode* Node::as<BufferingQueueNode>()
{
    TT_ASSERT(this->node_type() == NodeType::kQueue);
    TT_ASSERT(this->as<QueueNode>()->is_buffering());
    return dynamic_cast<BufferingQueueNode *>(this);
}

template<> const BufferingQueueNode* Node::as<BufferingQueueNode>() const
{
    TT_ASSERT(this->node_type() == NodeType::kQueue);
    TT_ASSERT(this->as<QueueNode>()->is_buffering());
    return dynamic_cast<BufferingQueueNode const *>(this);
}

bool is_permute_xy_order(const std::vector<int>& order) {
    const int rank = (int)order.size();
    for (int i = 0; i < rank - 2; ++i) {
        if (order[i] != i) {
            return false;
        }
    }
    return order[rank - 2] == rank - 1 && order[rank - 1] == rank - 2;
}

std::vector<int> create_permute_xy_order(const int rank) {
    std::vector<int> order = {};
    order.reserve(rank);
    for (auto i = 0; i < rank - 2; ++i) {
        order[i] = i;
    }
    order[rank - 2] = rank - 1;
    order[rank - 1] = rank - 2;
    return order;
}

std::unique_ptr<Node> BudaOpNode::clone(std::string const& name) {
    std::unique_ptr<BudaOpNode> node = create_node<BudaOpNode>(this->name(), this->op_type());
    node->Node::clone(this, name);
    node->set_gradient_op(this->is_gradient_op());
    node->set_golden_transforms(this->get_golden_transforms());
    node->set_intermediate_df(this->intermediate_df());
    node->set_accumulate_df(this->accumulate_df());
    node->set_math_fidelity(this->math_fidelity());
    node->add_tags(this->as<TaggedNode>()->get_tags());
    return node;
}

void BudaOpNode::copy_lowered_op_attributes(PyOpNode *node)
{
    epoch_type_ = node->get_epoch_type();
    set_gradient_op(node->is_gradient_op());
    set_output_df(node->output_df());
    set_intermediate_df(node->output_df()); // by default, same as output
    // accumulate df will not be set here, we'll have an overall default

    // If there are golden transforms, they operate on pybuda shapes,
    // so we need to insert narrowing in order make BUDA compatible
    set_golden_transforms(node->get_golden_transforms());
    if (not get_golden_transforms().empty())
    {
        int r = node->shape().size() > 1 ? node->shape()[-2] : 1;
        int c = node->shape()[-1];
        if ((r % Shape::BUDA_TILE_DIM) != 0)
            add_golden_transform(graphlib::OpType("narrow", {-2, 0, r, r}, {}));
        if ((c % Shape::BUDA_TILE_DIM) != 0)
            add_golden_transform(graphlib::OpType("narrow", {-1, 0, c, c}, {}));
    }

    if (node->has_golden_id()) set_golden_id(node->golden_id());
    this->as<graphlib::TaggedNode>()->add_tags(node->as<graphlib::TaggedNode>()->get_tags());
}

void BudaOpNode::copy_parent_op_attributes(BudaOpNode *node)
{
    epoch_type_ = node->get_epoch_type();
    set_output_df(node->output_df());
    set_intermediate_df(node->intermediate_df());
    set_accumulate_df(node->accumulate_df());
}

void BudaNaryTMNode::copy_lowered_op_attributes(PyOpNode *node)
{
    epoch_type_ = node->get_epoch_type();
    set_output_df(node->output_df());
    this->as<graphlib::TaggedNode>()->add_tags(node->as<graphlib::TaggedNode>()->get_tags());
}

std::unique_ptr<Node> PyOpNode::clone(std::string const& name) {
    std::unique_ptr<PyOpNode> node = create_node<PyOpNode>(this->name(), this->op_type());
    node->Node::clone(this, name);
    node->set_gradient_op(this->is_gradient_op());
    node->set_golden_transforms(this->get_golden_transforms());
    node->add_tags(this->as<TaggedNode>()->get_tags());
    return node;
}

void PyOpNode::copy_parent_op_attributes(PyOpNode *node)
{
    epoch_type_ = node->get_epoch_type();
    set_output_df(node->output_df());
}

bool OpNode::is_tm() const
{
    std::string path = node_type() == NodeType::kPyOp ? "pybuda.op.eval.pybuda" : "pybuda.op.eval.buda";
    py::object eval_module = py::module_::import(path.c_str());
    py::function is_tm = eval_module.attr("is_tm");
    return is_tm(op_type()).cast<bool>();
}

// Figure out output dafa format based on the input formats.
// TODO: add control on how to choose.
void OpNode::set_output_df_from_operands(const Graph *graph)
{
    auto operands = graph->data_operands(this);
    if (operands.size() == 1) {
        set_output_df(operands[0]->output_df());
        return;
    }

    // Somewhat arbitrary
    set_output_df(operands[0]->output_df());
}

bool OpNode::should_pair_with_sparse(const OpNode *sparse_op_node, const Graph *graph) const
{
    TT_ASSERT(sparse_op_node->is_sparse_matmul());
    if (is_matmul_not_sparse() or op_name().compare("reduce") == 0)
    {
        if (this->has_tag("original_op_name") and sparse_op_node->has_tag("original_op_name"))
        {
            std::string original_op_name = std::get<std::string>(this->tag_value("original_op_name"));
            std::string sparse_original_op_name = std::get<std::string>(sparse_op_node->tag_value("original_op_name"));
            if (original_op_name.compare(sparse_original_op_name) == 0)
            {
                if (graph->data_users(sparse_op_node).size() == 1)
                {
                    bool can_be_paired = false;
                    for (const Node *operand_node : graph->data_operands(this))
                    {
                        if (operand_node == sparse_op_node)
                        {
                            can_be_paired = true;
                            continue;
                        }

                        if (operand_node->node_type() != graphlib::NodeType::kInput)
                        {
                            can_be_paired = false;
                            break;
                        }
                    }

                    return can_be_paired;
                }
            }
        }
    }

    return false;
}

py::function OpNode::py_attr(char const *attr_name) const { return op_type().py_attr(attr_name, get_ir_level()); }
py::function OpNode::py_attr(char const *attr_name) { return op_type().py_attr(attr_name, get_ir_level()); }

InputNode::InputNode(const std::string &name, InputNodeType input_type, bool requires_grad) :
    QueueNode(name, QueueNodeType::Input, NodeType::kInput), input_type_(input_type), requires_grad_(requires_grad) {}

InputNode::~InputNode() = default;

std::unique_ptr<Node> InputNode::clone(std::string const& name) {
    std::unique_ptr<InputNode> node = create_node<InputNode>(this->name(), this->input_type(), this->requires_grad());
    node->Node::clone(this, name);
    if (consteval_graph_)
        node->consteval_graph_ = consteval_graph_->clone(node.get());
    node->tile_broadcast_dims_ = tile_broadcast_dims_;
    node->runtime_tensor_transform = runtime_tensor_transform;
    node->add_tags(this->as<TaggedNode>()->get_tags());
    return node;
}

void InputNode::clone_consteval_graph_from(Node* original)
{
    graphlib::InputNode* original_input = original->as<graphlib::InputNode>();
    if (original_input->get_consteval_graph())
    {
        this->consteval_graph_ = original_input->get_consteval_graph()->clone(this, this->name());
    }
}

std::unique_ptr<Node> QueueNode::clone(std::string const& name) {
    std::unique_ptr<QueueNode> node = create_node<QueueNode>(this->name(), queue_type_);
    node->Node::clone(this, name);
    node->entries_ = entries_;
    node->add_tags(this->as<TaggedNode>()->get_tags());
    return node;
}

std::unique_ptr<Node> EpochToEpochQueueNode::clone(std::string const& name) {
    std::unique_ptr<EpochToEpochQueueNode> node = create_node<EpochToEpochQueueNode>(this->name(), cross_epoch_type_, cross_chip_type_);
    node->Node::clone(this, name);
    node->entries_ = entries_;
    node->add_tags(this->as<TaggedNode>()->get_tags());
    return node;
}


std::unique_ptr<Node> BufferingQueueNode::clone(std::string const& name) {
    std::unique_ptr<BufferingQueueNode> node = create_node<BufferingQueueNode>(this->name(), this->get_num_entries());
    node->Node::clone(this, name);
    node->add_tags(this->as<TaggedNode>()->get_tags());
    return node;
}

std::unique_ptr<Node> OutputNode::clone(std::string const& name) {
    std::unique_ptr<OutputNode> node = create_node<OutputNode>(this->name());
    node->Node::clone(this, name);
    node->requires_grad_ = requires_grad_;
    node->is_loss_output_ = is_loss_output_;
    node->runtime_tensor_transform = runtime_tensor_transform;
    node->add_tags(this->as<TaggedNode>()->get_tags());
    return node;
}

static py::function get_f_instance(IRLevel ir_level)
{
    auto eval_module =
        py::module_::import((ir_level == IRLevel::IR_BUDA) ? "pybuda.op.eval.buda" : "pybuda.op.eval.pybuda");
    return eval_module.attr("get_f_instance");
}

py::function OpType::py_attr(char const *name, IRLevel ir_level) const
{
    auto instance = get_f_instance(ir_level)(this);
    return instance.attr(name);
}

py::function OpType::py_attr(char const *name, IRLevel ir_level)
{
    auto instance = get_f_instance(ir_level)(this);
    return instance.attr(name);
}

std::unique_ptr<Node> ConstantInputNode::clone(std::string const& name) {
    std::unique_ptr<ConstantInputNode> node;
    switch (this->node_type_) {
        case ConstantInputNodeType::SingleValue:
            node = create_node<ConstantInputNode>(this->name(), this->constant_value_);
            break;
        case ConstantInputNodeType::SingleTile:
            node = create_node<ConstantInputNode>(this->name(), this->tile_value_);
            break;
        case ConstantInputNodeType::Tensor:
            node = create_node<ConstantInputNode>(this->name(), this->tensor_handle_, this->tensor_shape_);
            break;
    }

    node->Node::clone(this, name);
    if (consteval_graph_)
        node->consteval_graph_ = consteval_graph_->clone(node.get());
    node->add_tags(this->as<TaggedNode>()->get_tags());
    node->sparse_buda = sparse_buda;
    return node;
}

bool ConstantInputNode::equivalent(const ConstantInputNode *other) const
{
    if (node_type_ != other->node_type_) return false;

    if (is_single_value())
        return constant_value() == other->constant_value();

    if (is_single_tile()) {
        return tile_value() == other->tile_value();
    }

    TT_ASSERT(is_tensor());
    return compare_tensors(tensor(), other->tensor());
}

bool EdgeAttributes::has_broadcast_dims() const
{
    return std::find_if(tms.begin(), tms.end(), [](const OpType &o) { return o.op == "broadcast"; }) != tms.end();
}

void EdgeAttributes::clear_broadcast_dims() 
{ 
    tms.erase(std::remove_if(tms.begin(), tms.end(), [](const OpType &o) { return o.op == "broadcast"; }), tms.end());
}

void EdgeAttributes::remove_broadcast_dim(int dim)
{
    auto filter = [=](const OpType &o) { 
        return o.op == "broadcast" && std::get<int>(o.attr[0]) == dim; 
    };
    tms.erase(std::remove_if(tms.begin(), tms.end(), filter), tms.end());
}


/*static*/ std::shared_ptr<EdgeAttributes> EdgeAttributes::create(EdgeType edge_type) {
    if (edge_type == EdgeType::kControlLoop) {
        return std::make_shared<LoopEdgeAttributes>(edge_type);
    }
    return std::make_shared<EdgeAttributes>(edge_type);
}


/*static*/ template<> const std::shared_ptr<LoopEdgeAttributes> EdgeAttributes::as<LoopEdgeAttributes>(const std::shared_ptr<EdgeAttributes>& base)
{
    TT_ASSERT(base->edge_type() == EdgeType::kControlLoop);
    return std::static_pointer_cast<LoopEdgeAttributes>(base);;
}

/*static*/ template<> std::shared_ptr<LoopEdgeAttributes>  EdgeAttributes::as<LoopEdgeAttributes>(std::shared_ptr<EdgeAttributes>& base)
{
    TT_ASSERT(base->edge_type() == EdgeType::kControlLoop);
    return std::static_pointer_cast<LoopEdgeAttributes>(base);
}

std::string QueueNode::queue_type_string() const
{
    switch (queue_type_) 
    {
        case QueueNodeType::EpochToEpoch: return "epoch_to_epoch";
        case QueueNodeType::GradAccumulator: return "grad_accumulator";
        case QueueNodeType::Input: return "input";
        case QueueNodeType::Output: return "output";
        case QueueNodeType::Buffering: return "buffering";
    }
    return "unknown";

}

std::string QueueNode::memory_access_type_string() const
{
    switch (this->memory_access_type_)
    {
        case MemoryAccessType::FIFO: return "FIFO"; // using 'queue' for backend integration
        case MemoryAccessType::RAM: return "RAM";
    }
    return "unknown";

}

std::string InputNode::input_type_string() const
{
    switch (input_type_)
    {
        case InputNodeType::Accumulator: return "accumulator";
        case InputNodeType::Activation: return "input";
        case InputNodeType::Loss: return "loss";
        case InputNodeType::Parameter: return "parameter";
        case InputNodeType::Constant: return "constant";
        case InputNodeType::OptimizerParameter: return "optimizer_parameter";
        case InputNodeType::Target: return "target";
    }
    return "unknown";
}

ConstEvalGraph *InputNode::get_consteval_graph(Graph* graph, bool create, bool promote_input)
{
    if (create and !consteval_graph_) {
        TT_ASSERT(graph, "Runtime Graph must be provided to create consteval graph");
        consteval_graph_ = std::make_unique<ConstEvalGraph>(
            this->name() + ".consteval_graph", this, promote_input, graph->get_subgraph_id_for_node(this->id()));
    }
    return consteval_graph_.get();
}
void InputNode::clear_consteval_graph() {
    if (consteval_graph_) {
        consteval_graph_.reset();
    }
}

std::ostream &operator<<(std::ostream &out, const OpType &op_type) {
    out << op_type.as_string();
    return out;
}

std::ostream &operator<<(std::ostream &out, InputNodeType t)
{
    switch (t)
    {
        case InputNodeType::Parameter: out << "InputNodeType::Parameter"; break;
        case InputNodeType::Constant: out << "InputNodeType::Constant"; break;
        case InputNodeType::Accumulator: out << "InputNodeType::Accumulator"; break;
        case InputNodeType::Activation: out << "InputNodeType::Activation"; break;
        case InputNodeType::Loss: out << "InputNodeType::Loss"; break;
        case InputNodeType::OptimizerParameter: out << "InputNodeType::OptimizerParameter"; break;
        case InputNodeType::Target: out << "InputNodeType::Target"; break;
        default: out << "InputNodeType::Unknown"; break;
    }
    return out;
}

std::ostream &operator<<(std::ostream &out, const UBlockOrder &ublock_order) {
    switch (ublock_order) {
        case UBlockOrder::R:
        out << "UBlockOrder::R";
        break;
        case UBlockOrder::C: out << "UBlockOrder::C"; break;
    }
    return out;
}
    
bool TaggedNode::has_tag(const std::string &tag) const
{
    return this->hints.find(tag) != this->hints.end(); 
}

void TaggedNode::tag(const std::string &tag, const TagValue& tag_value)
{
    this->hints[tag] = tag_value;
}

TagValue TaggedNode::tag_value(const std::string &tag) const
{
    return this->hints.at(tag);
}

void TaggedNode::add_tags(const TagHints& other_tags)
{
    this->hints.insert(other_tags.begin(), other_tags.end());
}

const TagHints& TaggedNode::get_tags() const {
    return this->hints;
}

}  // namespace graphlib
}  // namespace tt
