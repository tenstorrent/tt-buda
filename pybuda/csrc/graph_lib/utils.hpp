// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>
#include <string_view>
#include <typeindex>
#include <typeinfo>

#include "graph_lib/defines.hpp"
namespace py = pybind11;

#include "graph_lib/edge.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "python_bindings_common.hpp"
#include "third_party/json/json.hpp"

template <typename T>
constexpr auto type_name(const T &) noexcept
{
    std::string_view name = __PRETTY_FUNCTION__;
    std::string_view prefix = "auto type_name(const T &) [T = ";
    std::string_view suffix = "]";
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    return name;
}

namespace tt
{

namespace graphlib
{
struct OpType;
class QueueNode;
class InputNode;
class BudaOpNode;

// pass through
bool default_node_filter(Node *);

// Checks if given opnode is element-wise
class OpNode;
bool is_eltwise(const OpNode *op);
bool is_eltwise_nary(const OpNode *op);
bool is_eltwise_unary(const OpNode *op);
bool is_eltwise_binary(const OpNode *op);
bool is_reduce_z(OpNode const *op);

// Find Row/Col size of TileDim
int get_row_size_from_tile_size(TileDim tile_dim);
int get_col_size_from_tile_size(TileDim tile_dim);
TileDim get_tile_dim_from_height_width(int tile_height, int tile_width);

// Returns a topological sort of all nodes in the graph
std::vector<Node *> topological_sort(
    Graph const &graph, std::function<bool(Node *)> node_filter = default_node_filter, bool unroll_loops = false);

std::vector<std::vector<Node *>> topological_generations(const Graph &graph);

// Returns vector of all visible nodes in the graph.
//
std::vector<Node *> visible_nodes(Graph const &graph, std::function<bool(Node *)> node_filter = default_node_filter);

// Find the longest path from the graph. Optionally look for paths that don't start from ordered inputs.
std::vector<Node *> get_longest_path(const Graph *graph, bool from_inputs_only = true);

std::vector<Node *> get_nodes_with_indegree_zero(Graph *graph);
std::vector<Node *> get_nodes_with_outdegree_zero(Graph *graph);
std::vector<Node *> get_nodes_with_data_outdegree_zero(Graph *graph);

// Insert new node on the given edge. Node attributes will be picked up from consumer node.
// Returns new edges to and from the new node.
std::pair<Edge, Edge> insert_node_on_edge(
    Graph *graph,
    Edge &edge,
    Node *node,
    bool inherit_consumer_attrs = true,
    bool remove_edge = true,
    std::uint32_t consumer_index = 0,
    bool place_tms_on_outgoing = false);

QueueNode *create_buffering_queue(
    Graph *graph, const graphlib::Node *producer_node, const std::string name, int num_entries);

// Creates and inserts a nop node on the given edge.
// Returns newly created node and edges.
std::tuple<BudaOpNode*, Edge, Edge> insert_nop_on_edge(Graph *graph, Edge &edge, const std::string &nop_name, bool is_buffering = false, bool hoist_tms = false, bool remove_edge = true);

// Bypass queue, connecting its source to its destination. There has to be only one source for queue, and user is
// defined by user_edge. Diference from bypassing node (bypass_node) is that here we can bypass some users of queue and
// leave some attached to queue. Also, we don't have control edges, but only kData edges.
std::unique_ptr<Node> connect_queue_src_to_queue_user(Graph *graph, Node *queue, Edge &user_edge, bool remove_queue);

// Bypass node, connecting its source to its destination(s). The node must only have one input operand.
// Optionally, user can provide callback on each of the newly created edges, and original edge.
std::unique_ptr<Node> bypass_node(
    Graph *graph, Node *node, bool remove_node, std::function<void(Edge, Edge)> callback = [](Edge, Edge) {});

// Replace node with a new one, removing the old one and reconnecting all edges as before.
// The new node must have the same number of operands, or skip_operands must be set.
void replace_node(Graph *graph, Node *original_node, Node *new_node, bool skip_operands);

// Swap the order of producer / consumer pair along this edge.
//   operand_callback will be called for each of the new producer operands
//   user_callback will be called for each of the new consumer users
//   returns the swapped edge
Edge swap(
    Graph *graph,
    Edge edge,
    std::function<void(Edge)> operand_callback = [](Edge) {},
    std::function<void(Edge)> user_callback = [](Edge) {});

// Return the subgraph bounded by producer and consumer, where producer and consumer
// are not necessarily directly connected.  Producer must appear before consumer.
// The returned subgraph will not contain producer or consumer.
// It follows that if the producer and consumer are directly connected or there isn't
// any path from producer to consumer, then an empty vector will be returned.
std::vector<Node *> subgraph(const Graph *graph, Node *producer, Node *consumer);

// Return nodes reachable from a given start node, using only data edges
std::vector<Node *> reachable_nodes(
    const Graph *graph,
    Node *start,
    std::function<bool(Node *)> node_filter = default_node_filter,
    bool ancenstors_only = false);

std::vector<Node *> top_row(Graph const *graph, std::vector<Node *> const &nodes);

std::vector<Node *> bot_row(Graph const *graph, std::vector<Node *> const &nodes);

// Check if there is a data dependency between two nodes(producer, consumer), return true if it exists
bool check_producer_consumer(
    Graph *graph, Node *producer, Node *consumer, std::function<bool(Node *)> node_filter = default_node_filter);

// Return a subset of graph nodes in their respective topological order
// There are two ways to filter:
// - Template type T will automatically filter nodes of this type
//     std::vector<OpNode*> op_nodes = sorted<OpNode>(graph, nodes /* std::vector<Node*> */);
// - Additionally a filter fn can be supplied
//     std::vector<OpNode*> tms = sorted<OpNode>(graph, nodes, [](auto* n) { return n->is_tm(); });
// - Template argument deduction
//     std::vector<InputNode*> inputs = sorted(graph, nodes /* std::vector<InputNode*> */);
template <typename T, typename U, typename Fn = bool(T *)>
inline std::vector<T *> sorted_impl(
    graphlib::Graph const *graph, std::vector<U *> const &nodes, Fn fn = [](T *) { return true; })
{
    std::vector<T *> sorted_nodes;
    sorted_nodes.reserve(nodes.size());

    for (U *n : nodes)
    {
        if (T *t = dynamic_cast<T *>(n); t and fn(t))
            sorted_nodes.push_back(t);
    }

    auto topo = graphlib::topological_sort(*graph);
    std::sort(
        sorted_nodes.begin(),
        sorted_nodes.end(),
        [&topo](graphlib::Node *a, graphlib::Node *b)
        { return std::find(topo.begin(), topo.end(), a) < std::find(topo.begin(), topo.end(), b); });

    return sorted_nodes;
}

template <typename T, typename U, typename Fn = bool(T *), typename = std::enable_if_t<!std::is_same_v<T, U>>>
inline std::vector<T *> sorted(
    graphlib::Graph const *graph, std::vector<U *> const &nodes, Fn fn = [](T *) { return true; })
{
    return sorted_impl<T, U>(graph, nodes, fn);
}

template <typename U, typename Fn = bool(U *), typename = std::enable_if_t<std::is_same_v<U, U>>>
inline std::vector<U *> sorted(
    graphlib::Graph const *graph, std::vector<U *> const &nodes, Fn fn = [](U *) { return true; })
{
    return sorted_impl<U, U>(graph, nodes, fn);
}

// Convert an op with N inputs to a cascade of binary ops.  Op type must be associative and commutative.
// Returns the final sink
graphlib::Node *cascade_nary_to_binary_op(graphlib::Graph *graph, graphlib::Node *nary_op);

// Replace implicit bcasts with explicit ones
void convert_implicit_to_explicit_bcasts(Graph *graph, Edge edge);

// Swap dimensions of any broadcast tms, return true if change made
bool swap_broadcast_dims(graphlib::Graph *graph, graphlib::Edge edge, int old_dim, int new_dim);

// Insert squeezes / unsqueezes to satisfy change in rank
void handle_change_rank(graphlib::Graph *graph, graphlib::Edge edge);
void handle_change_rank(graphlib::Graph *graph, graphlib::Node *node);

// This function clones the input producer node and creates a new edge connection replacing
// the old edge. user_edge must come from an input node, returns new edge.
graphlib::Edge clone_input_forking_edge(
    graphlib::Graph *graph, graphlib::Edge user_edge, bool allow_single_user = false);

graphlib::Shape default_tm_evaluator(graphlib::OpType const &tm, graphlib::Shape shape, graphlib::IRLevel ir_level);
graphlib::Shape ignore_broadcast_tm_evaluator(
    graphlib::OpType const &tm, graphlib::Shape shape, graphlib::IRLevel ir_level);

graphlib::Shape post_tms_shape(
    graphlib::Shape input_shape,
    std::vector<OpType> const &tms,
    std::function<graphlib::Shape(graphlib::OpType const &, graphlib::Shape, graphlib::IRLevel)> tm_evaluator =
        default_tm_evaluator,
    graphlib::IRLevel ir_level = IRLevel::IR_BUDA);

graphlib::Shape post_tms_shape(
    Graph const *graph,
    graphlib::Edge edge,
    std::function<graphlib::Shape(graphlib::OpType const &, graphlib::Shape, graphlib::IRLevel)> tm_evaluator =
        default_tm_evaluator);

std::pair<int, int> get_padding(graphlib::Graph const *graph, graphlib::Node const *node);

bool tms_support_kernel_broadcast(
    Shape producer_shape, std::vector<OpType> const &tms, UBlockOrder ublock_order, int ublock_ct, bool is_buda = true);

// Calculate node shape from operand shapes, using python callback
void calculate_and_set_node_shape(Graph *graph, Node *node);

tt::graphlib::Node *get_input_queue_producer(Graph const *graph, tt::graphlib::InputNode const *node);
std::vector<tt::graphlib::UBlockOrder> get_input_ublock_order(Graph const *graph, Node const *node);
tt::graphlib::UBlockOrder get_input_queue_ublock_order(Graph const *graph, Node const *node);
tt::graphlib::UBlockOrder get_output_ublock_order(Graph const *graph, Node const *node);

// Insert NOP on an edge with transpose TM, then flip ublock order for better streaming
// returns true if nop inserted
bool try_insert_nop_on_transpose_edge(Graph *graph, Edge &edge);

// Return a vector of pairs of optimizer parameter input nodes and optimizer key names for a given model parameter node
class InputNode;
std::vector<std::pair<InputNode *, std::string>> get_optimizer_param_info(
    const Graph *graph, const Node *model_parameter);

bool is_constant_input(const Node *node);
bool is_recompute(const Graph *graph, const Node *node);
Node *get_fwd_from_recompute(const Graph *graph, const Node *node);

bool can_swap_operands(Graph *graph, Node *node);
void swap_operands(Graph *graph, Node *node);

Edge retrieve_between_edge(Graph *graph, Node *producer, Node *consumer);
bool are_bcasts_between_ops(Graph *graph, Node *producer, Node *consumer);
//
// Consteval
//
bool is_consteval_capable_input_type(Node *node);
bool is_consteval_capable_input(Graph *graph, InputNode *input);
// Note: if allow_forks is true, caller must explicitly deal with splitting the fork, consteval
//       inputs have no way of naturally dealing with a fork. Only used by consteval pass.
bool is_consteval_capable_op(Graph *graph, Node *node, bool allow_forks = false);
// Returns removed runtime node if successful consteval else nullptr
std::unique_ptr<Node> try_consteval_op(Graph *graph, Node *node, bool dump_graph = false);

bool try_consteval_input_no_operand_forks(Graph *graph, InputNode *input, bool dump_graph = false);

class ConstEvalGraph
{
   public:
    explicit ConstEvalGraph(
        std::string const &name, Node *runtime_input, bool promote_input, unsigned int subgraph_id, int unique_id = -1);
    Graph *get_graph()
    {
        TT_ASSERT(not ran_autograd or not graph_updated_since_autograd);
        return &consteval_graph;
    }
    Node *get_output() { return consteval_output; }
    bool has_node_with_name(std::string const &n) const { return consteval_graph.has_node_with_name(n); }
    std::unique_ptr<Node> promote_node(std::unique_ptr<Node> &&consteval_node);
    std::unique_ptr<Node> promote_node(Graph *runtime_graph, Node *runtime_node);
    std::unique_ptr<ConstEvalGraph> clone(Node *runtime_input, std::string const &new_input_node_name = "");
    void pad_output_to_buda_dims(std::string const &name_prefix);
    void set_needs_autograd(bool new_needs_autograd) { needs_autograd = new_needs_autograd; }
    void autograd();

   private:
    std::unique_ptr<Node> promote_node(
        Graph *runtime_graph, Node *runtime_node, std::unique_ptr<Node> &&consteval_node);
    Node *graft(Graph *other);

   private:
    Graph consteval_graph;
    Node *runtime_input = nullptr;
    Node *consteval_output = nullptr;
    std::unordered_map<NodeId, NodeId> runtime_to_consteval_map;
    bool needs_autograd = false;
    bool ran_autograd = false;
    bool graph_updated_since_autograd = false;
    unsigned int subgraph_id_;
};

enum class RuntimeTensorTransformType
{
    NoTransform = 0,
    ReinterpretShape,
    Prestride,
    EmbeddingIndex,
    ConstantInput,
    Unpad,
    Concatenate,
};

NLOHMANN_JSON_SERIALIZE_ENUM(
    tt::graphlib::RuntimeTensorTransformType,
    {
        {tt::graphlib::RuntimeTensorTransformType::NoTransform, "NoTransform"},
        {tt::graphlib::RuntimeTensorTransformType::ReinterpretShape, "ReinterpretShape"},
        {tt::graphlib::RuntimeTensorTransformType::Prestride, "Prestride"},
        {tt::graphlib::RuntimeTensorTransformType::EmbeddingIndex, "EmbeddingIndex"},
        {tt::graphlib::RuntimeTensorTransformType::ConstantInput, "ConstantInput"},
        {tt::graphlib::RuntimeTensorTransformType::Unpad, "Unpad"},
        {tt::graphlib::RuntimeTensorTransformType::Concatenate, "Concatenate"},
    });

class RuntimeTensorTransform
{
    // TODO: Refactor this properly
   public:
    RuntimeTensorTransformType type = RuntimeTensorTransformType::NoTransform;

    // ReinterpretShape
    graphlib::Shape original_shape;
    graphlib::Shape reinterpreted_shape;

    // Unpad
    graphlib::Shape unpadded_shape;

    RuntimeTensorTransform() = default;

    // Temporary fields for Prestride until refactor
    int stride_height;
    int stride_width;
    int kernel_height;
    int kernel_width;

    int concat_group;
    int concat_index;
    int concat_dim;

    RuntimeTensorTransform(
        RuntimeTensorTransformType type,
        graphlib::Shape original_shape,
        graphlib::Shape reinterpreted_shape,
        graphlib::Shape unpadded_shape,
        int stride_height,
        int stride_width,
        int kernel_height,
        int kernel_width,
        int concat_group,
        int concat_index,
        int concat_dim) :
        type(type),
        original_shape(original_shape),
        reinterpreted_shape(reinterpreted_shape),
        unpadded_shape(unpadded_shape),
        stride_height(stride_height),
        stride_width(stride_width),
        kernel_height(kernel_height),
        kernel_width(kernel_width),
        concat_group(concat_group),
        concat_index(concat_index),
        concat_dim(concat_dim)
    {
    }

    RuntimeTensorTransform(Shape original_shape, Shape reinterpreted_shape)
    {
        this->type = RuntimeTensorTransformType::ReinterpretShape;

        this->original_shape = original_shape;
        this->reinterpreted_shape = reinterpreted_shape;
    }
    RuntimeTensorTransform(Shape unpadded_shape)
    {
        this->type = RuntimeTensorTransformType::Unpad;

        this->unpadded_shape = unpadded_shape;
    }

    static RuntimeTensorTransform ConcatenateOnHost(int group, int index, int dim)
    {
        RuntimeTensorTransform transform;
        transform.type = RuntimeTensorTransformType::Concatenate;
        transform.concat_group = group;
        transform.concat_index = index;
        transform.concat_dim = dim;
        return transform;
    }

    static RuntimeTensorTransform EmbeddingIndex(Shape original_shape)
    {
        RuntimeTensorTransform transform;
        transform.type = RuntimeTensorTransformType::EmbeddingIndex;
        transform.original_shape = original_shape;
        return transform;
    }

    void swap_original_and_reinterpreted_shapes()
    {
        if (this->type == RuntimeTensorTransformType::ReinterpretShape)
        {
            std::swap(original_shape, reinterpreted_shape);
        }
    }

    void set_constant_input_tensor(py::object tensor)
    {
        this->type = RuntimeTensorTransformType::ConstantInput;
        this->constant_tensor = make_shared_py_object(tensor);
    }

    py::object get_constant_input_tensor() { return borrow_shared_py_object(this->constant_tensor); }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
        RuntimeTensorTransform,
        type,
        original_shape,
        reinterpreted_shape,
        unpadded_shape,
        stride_height,
        stride_width,
        kernel_height,
        kernel_width,
        concat_group,
        concat_index,
        concat_dim);

   private:
    // Constant Input
    std::shared_ptr<void> constant_tensor;
};
bool are_different_ranked_shapes_equivalent(Shape a, Shape b);

bool is_linked_queue(const Graph *graph, const Node *node);

bool is_input_host_queue(bool input_queues_on_host, const Graph *graph, const Node *node);

bool is_output_host_queue(bool output_queues_on_host, const Graph *graph, const Node *node);

// Wrapper graph management utility class for Node.
// If remove_from_graph is set to true on destruction of NodeGraphContainer
// graph->remove_node(node) will be invoked.
//
class NodeGraphContainer
{
   public:
    Node *node;
    Graph *graph;
    bool remove_from_graph;

    NodeGraphContainer(Node *node, Graph *graph) : node(node), graph(graph), remove_from_graph(true){};
    ~NodeGraphContainer();
};
}  // namespace graphlib
}  // namespace tt
