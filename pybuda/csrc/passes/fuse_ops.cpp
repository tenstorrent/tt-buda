// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/fuse_ops.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "balancer/balancer_utils.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/query.hpp"
#include "graph_lib/shape.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"

namespace tt
{

using Graph = graphlib::Graph;
using Node = graphlib::Node;
using Edge = graphlib::Edge;
using EdgeType = graphlib::EdgeType;
using EdgeAttributes = graphlib::EdgeAttributes;
using PortId = graphlib::PortId;

// Sub-topo order of ops in schedule whole result will be broadcast to another schedule
struct SubTopo
{
    bool broadcast_r, broadcast_c;
    std::vector<BudaOpNode *> ops;
    std::unordered_set<BudaOpNode *> outputs;
};

// Represents a group of ops to be fused together. Provides algorithms to generate, modify, and legalize the groups.
class FusionGroup
{
   private:
    std::uint32_t id;
    std::unordered_map<std::string, BudaOpNode *> nodes;
    graphlib::NodeEpochType epoch_type;
    bool has_matmul;
    bool has_reduce;
    bool has_broadcast_c;
    std::uint32_t reduce_dim = 0;

    std::vector<BudaOpNode *> topo_order;
    BudaOpNode* output_op = nullptr;

    // Remove the cases where one part of the fork is inside the fused op, and the other is not, "wrapping" around other
    // ops
    void remove_fork_wraps(Graph *graph, std::unordered_map<graphlib::NodeId, FusionGroupP> &fused_nodes);

    // Look through the output cone of the node, and see if it converges on something in the fused op.
    BudaOpNode *converge_back_on_fused_op(Graph *graph, Node *node, std::unordered_set<Node *> &visisted) const;

    // Pick first output in topo sort, and remove all others
    void pick_first_output(
        Graph *graph, std::unordered_map<graphlib::NodeId, FusionGroupP> &fused_nodes, const std::vector<Node *> &topo);

    // Remove op, and everything below it
    void remove_op_and_below(
        Graph *graph, BudaOpNode *op, std::unordered_map<graphlib::NodeId, FusionGroupP> &fused_nodes);

    // Create schedules needed to execute the op
    void create_schedules(
        Graph *graph,
        BudaOpNode *output_op,
        std::vector<FusedSchedule> &schedules,
        std::vector<Edge> &input_edges,
        InputMapping &input_mapping,
        bool reuse_dest_on_srcA_only);

    // Get input cone of ops for the give node, and allowed ops.
    // If stop_on_base_op is set, we won't include ops which are basis for the schedules (matmul/reduce ops).
    std::unordered_set<BudaOpNode *> get_input_cone(
        const Graph *graph,
        BudaOpNode *node,
        const std::unordered_set<BudaOpNode *> &allowed_ops,
        bool stop_on_base_op = false) const;

    // Reuse dest if possible. Returns true if dest is reused.
    bool reuse_dest_if_possible(
        BudaOpNode *op,
        std::vector<FusedSubOpInput> &inputs,
        std::uint32_t prev_output_allocated_buffer,
        bool reuse_dest_on_srcA_only,
        FusedSchedule &sch);

   public:
    FusionGroup() : id(next_fuse_id++), has_matmul(false), has_reduce(false) {}

    FusionGroupP clone();

    void add_op(BudaOpNode *op)
    {
        if (empty())
            epoch_type = op->get_epoch_type();
        else
            TT_ASSERT(op->get_epoch_type() == epoch_type);
        nodes.insert(std::make_pair(op->name(), op));
    }
    bool has_op(Node *op) const { return nodes.count(op->name()) > 0; }
    bool has_op(const std::string name) const { return nodes.count(name) > 0; }
    bool empty() const { return nodes.empty(); }
    bool single_op() const { return nodes.size() == 1; }
    std::size_t count() const { return nodes.size(); }
    std::uint32_t get_id() const { return id; }
    bool has_matmul_op() const { return has_matmul; }
    bool has_broadcast_c_tm() const { return has_broadcast_c; }
    bool has_reduce_op() const { return has_reduce; }
    std::uint32_t get_reduce_dim() const { return reduce_dim; }
    Node* get_output_op() const { return output_op; }

    void set_reduce_op(uint32_t reduce_dim)
    {
        TT_ASSERT(has_reduce == false);
        this->has_reduce = true;
        this->reduce_dim = reduce_dim;
    }

    const std::vector<BudaOpNode *> &get_topo_order() const
    {
        TT_ASSERT(topo_order.size() > 0, "Call legalize() to legalize op and generator topo order");
        return topo_order;
    }

    void remove_op(BudaOpNode *op, std::unordered_map<graphlib::NodeId, FusionGroupP> &fused_nodes)
    {
        TT_ASSERT(has_op(op));
        nodes.erase(op->name());
        fused_nodes.erase(op->id());
    }

    // Generate topological order of fused ops, using graph topological order as reference
    void generate_topo_order(const std::vector<Node *> &graph_topo);

    void clear(std::unordered_map<graphlib::NodeId, FusionGroupP> &fused_nodes)
    {
        for (auto node : nodes)
        {
            fused_nodes[node.second->id()] = nullptr;
        }
        nodes.clear();
    }

    graphlib::NodeEpochType get_epoch_type() const
    {
        TT_ASSERT(!empty());
        return epoch_type;
    }

    std::vector<graphlib::Node const*> get_nodes_as_vector() const
    {
        std::vector<graphlib::Node const*> vec;
        vec.reserve(nodes.size());
        for (auto const& [name, node] : nodes) vec.push_back(node);
        return vec;
    }

    int get_dram_input_count(graphlib::Graph const *graph, std::vector<graphlib::Node const *> additional_nodes = {}) const
    {
        return get_input_count(
            graph,
            additional_nodes,
            [](auto const *node) { return dynamic_cast<graphlib::QueueNode const *>(node) != nullptr; });
    }

    int get_input_count(
        graphlib::Graph const *graph,
        std::vector<graphlib::Node const *> additional_nodes = {},
        std::function<bool(graphlib::Node const *)> filter = [](auto const *) { return true; }) const
    {
        int input_count = 0;
        std::vector<graphlib::Node const *> set = get_nodes_as_vector();
        set.insert(set.end(), additional_nodes.begin(), additional_nodes.end());

        for (auto const* node : set)
        {
            for (auto const* operand : graph->data_operands(node))
            {
                if (not filter(operand))
                    continue;
                bool in_fused_set = std::find(set.begin(), set.end(), operand) != set.end();
                input_count += int(not in_fused_set);
            }
        }
        return input_count;
    }

    int get_output_count(graphlib::Graph const *graph, std::vector<graphlib::Node const *> additional_nodes = {}) const
    {
        int output_count = 0;
        std::vector<graphlib::Node const *> set = get_nodes_as_vector();
        set.insert(set.end(), additional_nodes.begin(), additional_nodes.end());

        for (auto const* node : set)
        {
            for (auto const* user : graph->data_users(node))
            {
                bool in_fused_set = std::find(set.begin(), set.end(), user) != set.end();
                output_count += int(not in_fused_set);
            }
        }
        return output_count;
    }

    int get_connection_count(
        graphlib::Graph const *graph, std::vector<graphlib::Node const *> additional_nodes = {}) const
    {
        return get_input_count(graph, additional_nodes) + get_output_count(graph, additional_nodes);
    }

    bool legalize(
        Graph *graph, std::unordered_map<graphlib::NodeId, FusionGroupP> &fused_nodes, const std::vector<Node *> &topo)
    {
        remove_fork_wraps(graph, fused_nodes);
        pick_first_output(graph, fused_nodes, topo);
        if (count() >= 2)
        {
            return true;
        }
        clear(fused_nodes);
        return false;  // nothing left
    }

    // Remove fused ops from the graph, replace with a new fused op. Return pointer to new op.
    void fuse(Graph *graph, FusionGroupP self, bool reuse_dest_on_srcA_only);

    void print() const
    {
        log_trace(LogFuser, "Fused op id={}", id);
        for (auto &[name, op] : nodes)
        {
            log_trace(LogFuser, "  {}: {}", name, op->op_type().op);
        }
    }

    static uint32_t next_fuse_id;
};

uint32_t FusionGroup::next_fuse_id = 0;

// TODO: There is already is_matmul and is_depthwise in BudaOpNode class... Remove one implementation...
// TODO: Add const strings instead of comparing with specifically typed string every time...
bool is_matmul(BudaOpNode *op) { return op->is_matmul(); }
bool is_reduce_max(BudaOpNode *op) { return (op->op_type().op == "reduce"); }
bool is_splice(BudaOpNode *op) { return (op->op_type().op == "splice"); }
bool is_buffer(BudaOpNode *op) { return (op->op_type().op == "buffer"); }

bool is_tile_broadcast(BudaOpNode *op)
{
    return op->as<graphlib::TaggedNode>()->has_tag("tile_broadcast_r") ||
           op->as<graphlib::TaggedNode>()->has_tag("tile_broadcast_c");
}

// Return false if not a reduce, and reduce_dim set if it is
bool find_reduce_dim(BudaOpNode *op, std::uint32_t &reduce_dim)
{
    if (is_reduce_max(op))
    {
        reduce_dim = std::get<int>(op->op_type().attr[0]);
        return true;
    }

    if (!is_matmul(op))
        return false;

    if (op->as<graphlib::TaggedNode>()->has_tag("reduce_r"))
    {
        reduce_dim = 2;
        return true;
    }

    if (op->as<graphlib::TaggedNode>()->has_tag("reduce_c"))
    {
        reduce_dim = 3;
        return true;
    }

    return false;
}

bool is_reduce(BudaOpNode *op)
{
    // Checking already allowed ops to determine if it's a reduce, i.e. we need to break the schedule here
    if (!is_matmul(op) && !is_reduce_max(op))
        return false;

    std::uint32_t reduce_dim;
    return find_reduce_dim(op, reduce_dim);
}

// Return dim/amount pair indicating how src shape should be broadcast to dst. Only one dim can be broadcast.
std::pair<std::uint32_t, std::uint32_t> get_broadcast(const Graph *graph, BudaOpNode *src, BudaOpNode *dst)
{
    const auto &tms = graph->get_edge_attributes(graph->get_edges(src, dst)[0])->get_tms();

    std::pair<std::uint32_t, std::uint32_t> brcst = {0, 0};

    for (auto tm : tms)
    {
        if (tm.op == "tile_broadcast")
            continue;

        if (tm.op != "broadcast")
            TT_ASSERT("Unsupported TM inside fused op! {} between {} and {}", tm.op, src->name(), dst->name());

        TT_ASSERT(brcst.second == 0, "More than one broadcast between {} and {} in fused op", src->name(), dst->name());

        int dim = std::get<int>(tm.attr[0]);
        int factor = std::get<int>(tm.attr[1]);

        if (factor == 1)
            factor = 0;  // broadcast to 1 is not really a broadcast

        brcst = {dim, factor};

        TT_ASSERT(
            dim == 2 || dim == 3,
            "Invalid broadcast dim inside fused op: {}, between {} and {}",
            dim,
            src->name(),
            dst->name());
    }

    return brcst;
}

// Return dim/amount pair indicating how src shape should be broadcast to dst. Only one dim can be broadcast.
std::pair<bool, bool> get_tile_broadcast(const Graph *graph, Node *src, BudaOpNode *dst)
{
    const auto &tms = graph->get_edge_attributes(graph->get_edges(src, dst)[0])->get_tms();

    for (auto tm : tms)
    {
        if (tm.op != "tile_broadcast")
            continue;

        int dim = std::get<int>(tm.attr[0]);

        TT_ASSERT(
            dim == 2 || dim == 3,
            "Invalid tile broadcast dim inside fused op: {}, between {} and {}",
            dim,
            src->name(),
            dst->name());

        if (dim == 2)
            return {true, false};

        return {false, true};
    }

    return {false, false};
}

bool is_allowed_matmul(Graph *graph, BudaOpNode *op)
{
    // Allowed matmuls are matmuls where one of the operands is a single tile.. these are reduce and tile broadcast ops
    if (!is_matmul(op))
        return false;

    // Fusing depthwise matmuls hasn't been tested
    if (op->is_depthwise_matmul())
        return false;

    bool allow_matmul = not env_as<bool>("PYBUDA_NO_FUSE_MATMUL");

    if (!allow_matmul)
        return false;

    auto operands = graph->data_operands(op);
    if (operands.size() > 2)
        return false;  // fused matmul with bias

    bool allow_reduce = env_as<bool>("PYBUDA_FUSE_REDUCE");
    bool allow_broadcast = not env_as<bool>("PYBUDA_FUSE_NO_TILE_BROADCAST");

    if (op->as<graphlib::TaggedNode>()->has_tag("reduce_r") || op->as<graphlib::TaggedNode>()->has_tag("reduce_c"))
        return allow_reduce;

    if (is_tile_broadcast(op))
        return allow_broadcast;

    return false;
}

std::unordered_set<BudaOpNode *> FusionGroup::get_input_cone(
    const Graph *graph,
    BudaOpNode *node,
    const std::unordered_set<BudaOpNode *> &allowed_ops,
    bool stop_on_base_op) const
{
    std::unordered_set<BudaOpNode *> input_cone;
    input_cone.insert(node);
    for (Node *operand : graph->data_operands(node))
    {
        if (operand->node_type() == graphlib::kBudaOp)
        {
            if (allowed_ops.count(operand->as<BudaOpNode>()) == 0)
                continue;

            if (stop_on_base_op)
            {
                std::uint32_t new_reduce_dim;
                bool is_reduce_op = find_reduce_dim(operand->as<BudaOpNode>(), new_reduce_dim);

                // Don't go further in case of encountering the base op (matmul/reduce).
                if (is_reduce_op || is_matmul(operand->as<BudaOpNode>()))
                    continue;
            }

            auto sub_cone = get_input_cone(graph, operand->as<BudaOpNode>(), allowed_ops, stop_on_base_op);
            input_cone.insert(sub_cone.begin(), sub_cone.end());
        }
    }
    return input_cone;
}

struct Buffer
{
    std::uint32_t id;
    std::uint32_t allocated_schedule_index;
    std::vector<std::uint32_t> outstanding_users;
};

class BufferAllocator
{
    std::uint32_t count;
    std::unordered_map<std::uint32_t, std::shared_ptr<Buffer>> buffers;

    // Once allocated for a particular data type, that buffer must always have the same one
    std::unordered_map<std::uint32_t, DataFormat> data_formats;

   public:
    BufferAllocator(std::uint32_t count) : count(count) {}
    std::shared_ptr<Buffer> allocate(
        std::vector<std::uint32_t> users,
        std::uint32_t schedule,
        bool local,
        const std::unordered_set<std::uint32_t> &blacklisted_buffers,
        DataFormat df)
    {
        for (std::uint32_t i = 0; i < count; i++)
        {
            if (buffers.count(i) > 0)
                continue;  // already allocated

            if (blacklisted_buffers.count(i) > 0)
                continue;  // not allowed

            auto it = data_formats.find(i);
            if ((it == data_formats.end()) || (it->second == df))
            {
                std::vector<std::uint32_t> local_users(users.size(), 0);
                local_users[schedule] = users[schedule];

                buffers[i] = std::make_shared<Buffer>(Buffer{i, schedule, local ? local_users : users});
                data_formats[i] = df;
                return buffers[i];
            }
        }
        TT_THROW("Ran out of intermediate buffers.");
        return 0;  // avoid warning
    }

    // return true if this buffer is now done with, and another flag to indicate if this was a cross-schedule deallocate
    std::pair<bool, bool> deallocate(std::uint32_t id, std::uint32_t schedule)
    {
        auto it = buffers.find(id);
        TT_LOG_ASSERT(it != buffers.end(), "Deallocating {}, which has already been deallocated.", id);

        auto &outstanding_users = it->second->outstanding_users;
        TT_ASSERT(outstanding_users[schedule] > 0);
        outstanding_users[schedule]--;

        if (outstanding_users[schedule] == 0)
        {
            // Check if there are later uses
            for (std::size_t i = schedule + 1; i < outstanding_users.size(); i++)
                if (outstanding_users[i] > 0)
                    return std::make_pair(false, false);

            auto allocated_schedule_index = it->second->allocated_schedule_index;
            buffers.erase(it);

            return std::make_pair(true, allocated_schedule_index != schedule);
        }

        return std::make_pair(false, false);
    }
};

// Reuse dest if possible. Returns true if dest is reused.
bool FusionGroup::reuse_dest_if_possible(
    BudaOpNode *op,
    std::vector<FusedSubOpInput> &inputs,
    std::uint32_t prev_output_allocated_buffer,
    bool reuse_dest_on_srcA_only,
    FusedSchedule &sch)
{
    // Reusing dest not allowed for matmul.
    if (is_matmul(op))
    {
        return false;
    }

    std::optional<std::uint32_t> reused_input_index = std::nullopt;
    for (std::size_t index = 0; index < inputs.size(); index++)
    {
        auto i = inputs[index];
        if ((i.type == FusedSubOpInput::INTERMED) && (i.index == prev_output_allocated_buffer))
        {
            // Dest can be reused only on 1 input and can't be reused if any input has broadcast.
            if ((reused_input_index.has_value()) || (i.broadcast.second != 0))
                return false;

            if (i.has_tile_broadcast() && index > 0)
                return false;

            reused_input_index = index;
        }
    }

    // Dest value not used on any of inputs.
    if (!reused_input_index.has_value())
        return false;

    // Check if dest resuse is only allowed on input 0.
    if (reuse_dest_on_srcA_only && (reused_input_index.value() != 0))
        return false;

    log_debug(LogFuser, "Reusing dest from previous fused op for {}", op->name());

    // Modify the previous op.
    TT_ASSERT(sch.ops.back().output_type == FusedSubOp::OutputType::INTERMED);
    sch.ops.back().output_type = FusedSubOp::OutputType::DEST;

    // Modify reused input.
    inputs[reused_input_index.value()].type = FusedSubOpInput::InputType::DEST;

    return true;
}

// Create schedules needed to execute the op
void FusionGroup::create_schedules(
    Graph *graph,
    BudaOpNode *output_op,
    std::vector<FusedSchedule> &schedules,
    std::vector<Edge> &input_edges,
    InputMapping &input_mapping,
    bool reuse_dest_on_srcA_only)
{
    // Each matmul/reduce operation needs to have its own schedule and be the last operation of that schedule.
    // Schedule algorithm:
    //  - Find all matmul/reduce ops which will form the basis for the schedules.
    //      - Track all dependencies between the base ops (matmul/reduce).
    //  - Order the schedules so that the dependencies between the base ops are satisfied.
    //  - Fill in the schedules with rest of the ops (each time starting from base op).

    // Set of nodes in the op, for input cone searching
    std::unordered_set<BudaOpNode *> node_set;
    for (auto &[name, node] : nodes) node_set.insert(node);

    // We need to recalculate has_reduce, since we could've pruned the reduce ops in legalize().
    this->has_reduce = false;
    this->reduce_dim = 0;

    std::unordered_map<BudaOpNode*, std::unordered_set<BudaOpNode*>> schedule_dependencies;

    for (auto &[name, node] : nodes)
    {
        std::uint32_t reduce_dim;
        bool is_reduce_op = find_reduce_dim(node, reduce_dim);
        bool is_matmul_op = is_matmul(node);

        // We need to make schedules only for reduce/matmul.
        if (!is_reduce_op && !is_matmul_op)
            continue;

        this->has_matmul |= is_matmul_op;
        if (is_reduce_op)
        {
            this->has_reduce = true;
            this->reduce_dim = reduce_dim;
        }

        auto input_cone = get_input_cone(graph, node, node_set);
        schedule_dependencies.insert(std::make_pair(node, std::unordered_set<BudaOpNode*>{}));
        for (BudaOpNode* c: input_cone)
        {
            if (c == node)
                continue;

            if (is_reduce(c) || is_matmul(c))
                schedule_dependencies[node].insert(c);
        }
    }

    std::vector<BudaOpNode *> schedule_output_nodes;  // list of outputs for each schedule
    auto scheduled = [&schedule_output_nodes](BudaOpNode *op) -> bool
    {
        return std::find(schedule_output_nodes.begin(), schedule_output_nodes.end(), op) !=
               schedule_output_nodes.end();
    };

    // Now that we have schedule dependencies, figure out the required order for the schedules.
    while (schedule_dependencies.size() > schedule_output_nodes.size())
    {
        bool progress = false;

        for (auto& [node, dependecies] : schedule_dependencies)
        {
            if (scheduled(node))
                // We have already scheduled this one - continue...
                continue;

            bool ok = true;
            for (auto d : dependecies)
            {
                if (!scheduled(d))
                {
                    // Dependency not satisfied so we cannot schedule this one.
                    ok = false;
                    break;
                }
            }

            if (!ok)
                continue;

            progress = true;
            schedule_output_nodes.push_back(node);
            break;
        }

        TT_LOG_ASSERT(progress, "Deadlock trying to find reduce without dependencies");
    }

    if (schedule_output_nodes.size() == 0 || schedule_output_nodes.back() != output_op)
    {
        // Output of the fused op still doesn't have its schedule, so we need to schedule it.
        schedule_output_nodes.push_back(output_op);
    }

    // Make sure we have enough schedules to cover all matmul/reduce ops.
    TT_ASSERT(schedule_output_nodes.size() >= schedule_dependencies.size());

    // Generate topo sort for each schedule
    std::vector<std::vector<BudaOpNode *>> all_topos;
    for (BudaOpNode *schedule_output_node : schedule_output_nodes)
    {
        // Get ops to schedule, while stopping on base ops (matmul/reduce), which must've already been scheduled
        auto input_cone = get_input_cone(graph, schedule_output_node, node_set, true /* stop_on_base_op */);

        // Topo sort
        std::vector<BudaOpNode *> input_cone_topo;
        for (BudaOpNode *node : get_topo_order())
            if (input_cone.count(node) > 0)
                input_cone_topo.push_back(node);

        all_topos.push_back(input_cone_topo);
    }

    // To efficiently allocate intermed buffers, we need to find the number of readers for each op
    std::unordered_map<BudaOpNode *, std::vector<std::uint32_t>> readers;
    for (auto node : node_set)
    {
        readers[node] = std::vector<std::uint32_t>(all_topos.size(), 0);
    }

    for (std::size_t schedule_index = 0; schedule_index < all_topos.size(); schedule_index++)
    {
        auto input_cone_topo = all_topos[schedule_index];
        for (BudaOpNode *op : input_cone_topo)
        {
            auto operands = graph->data_operands(op);
            for (Node *operand : operands)
            {
                if ((operand->node_type() == graphlib::kBudaOp) && node_set.count(operand->as<BudaOpNode>()) > 0)
                {
                    TT_LOG_ASSERT(readers.count(operand->as<BudaOpNode>()) > 0, operand->name());
                    readers[operand->as<BudaOpNode>()][schedule_index] += 1;
                }
            }
        }
    }

    BufferAllocator buf_allocator(8);
    std::unordered_map<BudaOpNode *, std::shared_ptr<Buffer>> buffers;  // evaluated ops and their output buffer number
    std::uint32_t input_id = 0;
    has_broadcast_c = false;

    for (std::size_t schedule_index = 0; schedule_index < all_topos.size(); schedule_index++)
    {
        auto input_cone_topo = all_topos[schedule_index];

        // Create ops in schedule
        FusedSchedule sch;
        std::unordered_set<std::uint32_t> blacklisted_buffers;  // buffers that can't be used any more in this schedule
        std::optional<std::uint32_t> prev_output_allocated_buffer = std::nullopt;
        for (BudaOpNode *op : input_cone_topo)
        {
            std::vector<FusedSubOpInput> inputs;
            std::vector<std::uint32_t> popped_buffers;
            std::vector<std::uint32_t> popped_last_buffers;
            auto operands = graph->data_operands(op);
            auto operand_edges = graph->operand_data_edges(op);

            // Op output DF is not set yet at this point... And, as of now, back-end doesn't really support
            // changing intermediate data formats, so we're going to hard-code this to a constant.
            // DataFormat df = op->output_df();
            DataFormat df = DataFormat::Float16_b;
            std::shared_ptr<Buffer> output_buffer;

            auto allocate_buffer = [&buffers,
                                    &output_buffer,
                                    &readers,
                                    &buf_allocator,
                                    schedule_index,
                                    df,
                                    &blacklisted_buffers](auto op)
            {
                TT_LOG_ASSERT(readers.count(op) > 0, op->name());
                bool local = !is_reduce(op) && !is_matmul(op);
                output_buffer = buf_allocator.allocate(readers[op], schedule_index, local, blacklisted_buffers, df);
                buffers[op] = output_buffer;
            };

            if (is_reduce(op) && (op != output_op))
            {
                // Allocate buffers before deallocating inputs if this is reduce, since we need to accumulate into a new
                // buffer
                allocate_buffer(op);
            }

            for (std::uint32_t i = 0; i < operands.size(); i++)
            {
                Node *operand = operands.at(i);
                if ((operand->node_type() != graphlib::kBudaOp) || !has_op(operand->as<BudaOpNode>()))
                {
                    std::uint32_t input_index = input_id;
                    input_mapping[op].insert(std::make_pair(i, input_id++));
                    input_edges.push_back(operand_edges.at(i));
                    inputs.push_back(FusedSubOpInput{
                        FusedSubOpInput::InputType::INPUT,
                        input_index,
                        {0, 0},
                        get_tile_broadcast(graph, operand, op)});
                }
                else
                {
                    auto it = buffers.find(operand->as<BudaOpNode>());
                    TT_LOG_ASSERT(it != buffers.end(), "Can't find source buffer for {}", operand->name());
                    std::pair<std::uint32_t, std::uint32_t> broadcast =
                        get_broadcast(graph, operand->as<BudaOpNode>(), op);

                    // Check if broadcast C exists and mark it in fused group.
                    if (broadcast.first == 3)
                    {
                        has_broadcast_c = true;
                    }

                    inputs.push_back(FusedSubOpInput{
                        FusedSubOpInput::InputType::INTERMED,
                        it->second->id,
                        broadcast,
                        get_tile_broadcast(graph, operand, op)});
                    auto [pop, pop_last] = buf_allocator.deallocate(it->second->id, schedule_index);
                    if (pop_last)
                    {
                        popped_last_buffers.push_back(it->second->id);
                        blacklisted_buffers.insert(it->second->id);  // don't use this buffer in this schedule
                    }
                    else if (pop)
                        popped_buffers.push_back(it->second->id);
                }
            }

            if (op == output_op)
            {
                output_buffer = nullptr;
            }
            else if (!is_reduce(op))
            {
                allocate_buffer(op);
            }

            std::unordered_map<std::string, std::uint32_t> attrs;
            // TODO: this needs to be set by legalizer, and not here...
            if (op->op_type().op == "matmul")
            {
                attrs["u_kt"] = operands[0]->shape().ct();  // not this simple.. TODO
                auto op0_tms = graph->get_edge_attributes(graph->get_edges(operands[0], op)[0])->get_tms();
                for (auto tm : op0_tms)
                    if ((tm.op == "broadcast") && (std::get<int>(tm.attr[0]) == 3))
                        attrs["u_kt"] *= std::get<int>(tm.attr[1]);

                if (is_reduce(op))
                {
                    attrs["m_k"] = 1;  // this is really a placeholder that we can override after ublocks are asigned
                }
                else
                {
                    // tile broadcast
                    attrs["m_k"] = 1;
                }
            }

            // See if it we can reuse previous dest
            if (prev_output_allocated_buffer.has_value())
            {
                auto it = std::find(popped_buffers.begin(), popped_buffers.end(), prev_output_allocated_buffer.value());
                if (it != popped_buffers.end())
                {
                    if (reuse_dest_if_possible(
                            op, inputs, prev_output_allocated_buffer.value(), reuse_dest_on_srcA_only, sch))
                    {
                        // If dest is reused cleanup buffer tracking.
                        popped_buffers.erase(it);
                    }
                }
            }

            std::uint32_t output_buffer_id = (output_buffer != nullptr) ? output_buffer->id : 0;

            // If the operator has relu activation, don't reuse it (for better performance).
            bool dont_reuse = op->buda_attrs().find("relu_en") != op->buda_attrs().end();

            if (output_buffer == nullptr || dont_reuse)
                prev_output_allocated_buffer.reset();
            else
                prev_output_allocated_buffer = output_buffer->id;  // save for dest reuse

            sch.ops.push_back(FusedSubOp{
                op->name(),
                op->op_type(),
                tt::balancer::get_op_shape(graph, op),
                inputs,
                (output_buffer == nullptr) ? FusedSubOp::OutputType::OUTPUT : FusedSubOp::OutputType::INTERMED,
                output_buffer_id,
                df,
                attrs,
                popped_buffers,
                popped_last_buffers});
        }
        schedules.push_back(sch);
    }
}

// Only R/C broadcasts allowed, and only one of them
bool are_allowed_tms(
    const std::vector<graphlib::OpType> &tms,
    bool disable_broadcast = false,
    bool is_matmul = false,
    PortId input_id = 0)
{
    bool broadcast_seen = false;

    for (auto tm : tms)
    {
        if ((tm.op != "broadcast") && (tm.op != "tile_broadcast"))
            return false;

        if (tm.op == "broadcast" and disable_broadcast)
            return false;

        if (tm.op == "broadcast")
        {
            if (broadcast_seen)
                return false;  // can't have more than one
            broadcast_seen = true;

            // Only broadcast C on input 0 is allowed for matmul.
            int dim = std::get<int>(tm.attr[0]);
            if (is_matmul && dim != 3 && input_id != 0)
                return false;
        }

        int dim = std::get<int>(tm.attr[0]);
        if ((dim != 2) && (dim != 3))
            return false;
    }
    return true;
}

// Checks if tile broadcast can be removed by going through all current BE limitations.
// In some cases updates graph structure by swapping data operands to overcome BE limitations.
bool is_tile_broadcast_replaceable(
    graphlib::Graph *graph,
    std::unordered_map<graphlib::NodeId, FusionGroupP> &fused_nodes,
    graphlib::Edge &edge,
    Node *src_node,
    std::vector<Edge> src_edges)
{
    Node *node = graph->node_by_id(edge.consumer_node_id);
    if (node->node_type() != graphlib::kBudaOp)
    {
        return false;
    }

    // Tile broadcast merge is enabled only for ops that are fused.
    BudaOpNode *user_node = node->as<BudaOpNode>();
    auto fused_op = fused_nodes.find(user_node->id());
    if (fused_op == fused_nodes.end() || fused_op->second == nullptr)
    {
        return false;
    }

    // Tile broadcast merge is enabled only for unary and binary ops.
    if (graph->data_operands(user_node).size() > 2)
    {
        return false;
    }

    // Tile broadcast is not supported for any op lowered to matmul.
    if (is_matmul(user_node))
    {
        return false;
    }

    // If tile broadcast is enabled only for port 1 and it is on port 0, swap opearands if possible.
    if (edge.consumer_input_port_id == 0)
    {
        if (!graphlib::can_swap_operands(graph, user_node))
        {
            return false;
        }

        // Before swapping the operands, check if the edge which is currently on port 1
        // contains tile_broadcast tm. If it does, we can't swap the operands.
        auto operand_edges = graph->operand_data_edges(user_node);
        for (auto operand_edge : operand_edges)
        {
            if (operand_edge.consumer_input_port_id == 1
                && graph->get_edge_attributes(operand_edge)->has_tm("tile_broadcast"))
            {
                return false;
            }
        }

        graphlib::swap_operands(graph, user_node);
        edge.consumer_input_port_id = 1;
    }

    // Edge can handle only one tile broadcast operation.
    // If this is second don't allow merge.
    EdgeAttributes attr(graph->get_edge_attributes(edge)->edge_type());
    attr.copy_from(*graph->get_edge_attributes(edge));
    for (auto tm : attr.get_tms())
    {
        if ("tile_broadcast" == tm.op)
        {
            return false;
        }
    }

    // If src node is also fused we need to check if edge can handle all TMs from source and output egde.
    fused_op = fused_nodes.find(src_node->id());
    if (fused_op != fused_nodes.end() && fused_op->second != nullptr)
    {
        // Add source edge TMs to already initialized user edge attributes.
        for (auto edge : src_edges) attr.append_tms(graph->get_edge_attributes(edge)->get_tms());

        if (!are_allowed_tms(attr.get_tms()))
            return false;
    }

    return true;
}

// Replace tile broadcasts with edge attributes, and remove constant inputs
void replace_tile_broadcasts(
    Graph *graph,
    Node *node,
    std::unordered_map<graphlib::NodeId, FusionGroupP> &fused_nodes,
    std::unordered_set<Node *> &to_delete_nodes)
{
    if (!(node->node_type() == graphlib::kBudaOp))
    {
        return;
    }

    std::uint32_t broadcast_dim;
    std::uint32_t src_operand, brcst_operand;
    if (node->as<graphlib::TaggedNode>()->has_tag("tile_broadcast_r"))
    {
        broadcast_dim = 2;
        src_operand = 1;
        brcst_operand = 0;
    }
    else if (node->as<graphlib::TaggedNode>()->has_tag("tile_broadcast_c"))
    {
        broadcast_dim = 3;
        src_operand = 0;
        brcst_operand = 1;
    }
    else
        return;  // not a tile broadcast

    BudaOpNode *op = node->as<BudaOpNode>();
    Node *src_node = graph->data_operands(op)[src_operand];
    Node *brcst_node = graph->data_operands(op)[brcst_operand];
    auto user_edges = graph->user_edges(op);
    bool all_ok = true;
    if (fused_nodes.count(op->id()) > 0)
    {
        FusionGroupP fused_group = fused_nodes[op->id()];
        if (fused_group != nullptr && fused_group->get_output_op() == op)
        {
            // Don't replace tile broadcast when it's the output of the fused op.
            return;
        }
    }

    for (Edge user_edge : user_edges)
    {
        std::vector<Edge> src_edges = graph->get_edges(src_node, op);

        if (!is_tile_broadcast_replaceable(graph, fused_nodes, user_edge, src_node, src_edges))
        {
            all_ok = false;
            break;
        }

        // Tile broadcast is possible only on operand b and it could happen that we swapped inputs and that user_edge
        // value is not valid anymore instead of refreshing it using hardcoded port 1.
        // Merge tile broadcast to input TM.
        Edge new_edge = Edge(src_node->id(), 0, user_edge.consumer_node_id, 1, user_edge.edge_type);

        // Copy TMs from tile_brodacst output edge to new edge.
        graph->add_edge(new_edge);
        graph->copy_edge_attributes(user_edge, new_edge);
        graph->remove_edge(user_edge);

        // Add tile_broadcast TM to edge TMs.
        graph->get_edge_attributes(new_edge)->prepend_tm(graphlib::OpType("tile_broadcast", {(int)broadcast_dim}));

        // Copy TMs from tile_brodacst operand edge to new edge.
        for (Edge src_edge : src_edges)
        {
            for (auto tm : graph->get_edge_attributes(src_edge)->get_tms())
                graph->get_edge_attributes(new_edge)->prepend_tm(tm);
        }
    }

    if (all_ok)
    {
        // Clean up leftover operand data edges.
        // TODO: shouldn't this be removed on graph level when node is removed?
        for (auto op_edge : graph->operand_data_edges(op))
        {
            graph->remove_edge(op_edge);
        }

        // If op is marked for fusing remove it from fusing structures first.
        if ((fused_nodes.count(op->id()) > 0) && (fused_nodes[op->id()] != nullptr))
        {
            fused_nodes[op->id()]->remove_op(op, fused_nodes);
        }

        // Mark node for deletion
        to_delete_nodes.insert(op);

        bool remove_brcst = true;
        for (auto user : graph->users(brcst_node))
            if (to_delete_nodes.count(user) == 0)
                remove_brcst = false;

        // If there are no more users remove broadcast input node too.
        if (remove_brcst)
        {
            to_delete_nodes.insert(brcst_node);
        }
    }
}

FusionGroupP FusionGroup::clone()
{
    FusionGroupP fusion_group_clone = std::make_shared<FusionGroup>(*this);

    // Assign new unique id to the clone.
    fusion_group_clone->id = FusionGroup::next_fuse_id++;
    return fusion_group_clone;
}

// Remove fused ops from the graph, replace with a new fused op.
void FusionGroup::fuse(Graph *graph, FusionGroupP self, bool reuse_dest_on_srcA_only)
{
    std::vector<Edge> input_edges;
    InputMapping input_mapping;
    BudaOpNode *output_op = nullptr;
    std::vector<Edge> output_edges;
    std::unordered_map<graphlib::Node *, std::uint32_t> input_ids;
    std::unordered_map<std::uint32_t, std::uint32_t> input_reuse;
    bool is_out = false;
    for (BudaOpNode *op : get_topo_order())
    {
        std::vector<Edge> user_edges = graph->user_data_edges(op);
        for (Edge edge : user_edges)
        {
            if ((edge.edge_type == graphlib::EdgeType::kDataLoopback) ||
                ((edge.edge_type == graphlib::EdgeType::kData) && !has_op(graph->node_by_id(edge.consumer_node_id))))
            {
                is_out = true;
                output_edges = user_edges;
            }

            if (is_out)
            {
                TT_ASSERT(output_op == nullptr, "Can't have more than one output");
                output_op = op;
                break;
            }
        }
    }

    // TT_ASSERT(input_edges.size() <= 8, "Too many inputs into fused op - 8 is max");

    TT_ASSERT(output_op != nullptr);
    std::vector<FusedSchedule> schedules;
    create_schedules(graph, output_op, schedules, input_edges, input_mapping, reuse_dest_on_srcA_only);

    std::unordered_set<std::uint32_t> ops_to_remove;
    for (BudaOpNode *op : get_topo_order()) ops_to_remove.insert(op->id());

    // Record non-data edges going into and out of ops that are going to be removed
    std::unordered_set<Edge> incoming_non_data_edges, outgoing_non_data_edges;
    for (BudaOpNode *op : get_topo_order())
    {
        auto edges = graph->operand_edges(op);
        for (Edge edge : edges)
        {
            if ((edge.edge_type != graphlib::EdgeType::kData) && (ops_to_remove.count(edge.producer_node_id) == 0))
                incoming_non_data_edges.insert(edge);
        }

        edges = graph->user_edges(op);
        for (Edge edge : edges)
        {
            if (((edge.edge_type != graphlib::EdgeType::kData) &&
                 (edge.edge_type != graphlib::EdgeType::kDataLoopback)) &&
                (ops_to_remove.count(edge.consumer_node_id) == 0))
            {
                outgoing_non_data_edges.insert(edge);
            }
        }
    }

    std::vector<graphlib::OpType::Attr> attrs;
    if (has_reduce or has_broadcast_c)
    {
        attrs.push_back((int)reduce_dim);
        attrs.push_back(has_broadcast_c);
    }
    BudaOpNode *new_op = graph->add_node(
        graphlib::create_node<graphlib::BudaOpNode>(
            "_fused_op_" + std::to_string(id), graphlib::OpType("fused_op", attrs)),
        graph->get_subgraph_id_for_node(output_op->id()));
    new_op->set_shape(output_op->shape());

    if (output_op->as<graphlib::TaggedNode>()->has_tag("original_op_type"))
    {
        new_op->tag("original_op_type", output_op->as<graphlib::TaggedNode>()->tag_value("original_op_type"));
    }

    graph->copy_node_attributes(output_op, new_op);

    for (BudaOpNode *op : get_topo_order())
    {
        graph->remove_node(op);
    }

    for (std::uint32_t i = 0; i < input_edges.size(); i++)
    {
        Edge old_edge = input_edges.at(i);
        Edge new_input_edge =
            Edge(old_edge.producer_node_id, old_edge.producer_output_port_id, new_op->id(), i, EdgeType::kData);
        graph->add_edge(new_input_edge);
        graph->copy_edge_attributes(old_edge, new_input_edge);
    }

    for (Edge output_edge : output_edges)
    {
        Edge new_output_edge = Edge(
            new_op->id(), 0, output_edge.consumer_node_id, output_edge.consumer_input_port_id, output_edge.edge_type);
        graph->add_edge(new_output_edge);
        graph->copy_edge_attributes(output_edge, new_output_edge);
    }

    for (Edge incoming_edge : incoming_non_data_edges)
    {
        Edge new_incoming_edge = Edge(incoming_edge.producer_node_id, 0, new_op->id(), 0, incoming_edge.edge_type);
        graph->add_edge(new_incoming_edge);
    }

    for (Edge outgoing_edge : outgoing_non_data_edges)
    {
        Edge new_outgoing_edge = Edge(new_op->id(), 0, outgoing_edge.consumer_node_id, 0, outgoing_edge.edge_type);
        graph->add_edge(new_outgoing_edge);
    }

    new_op->set_fused_op(std::make_shared<FusedOp>(self, new_op, input_mapping, output_op, schedules));
}

// Look through the output cone of the node, and see if it converges on something in the fused op.
BudaOpNode *FusionGroup::converge_back_on_fused_op(Graph *graph, Node *node, std::unordered_set<Node *> &visited) const
{
    // Depth-first search for something that's in this fused op
    if (visited.count(node) > 0)
        return nullptr;

    visited.insert(node);
    // std::cout << " -- checking " << node->name() << std::endl;
    for (Node *user : graph->data_users(node))
    {
        if (user->node_type() != graphlib::kBudaOp)
            continue;

        if (has_op(user->name()))
        {
            return user->as<BudaOpNode>();
        }

        if (user->get_epoch_type() != node->get_epoch_type())
            continue;

        BudaOpNode *converged = converge_back_on_fused_op(graph, user, visited);
        if (converged != nullptr)
            return converged;
    }

    return nullptr;
}

// Remove op, and everything below it
void FusionGroup::remove_op_and_below(
    Graph *graph, BudaOpNode *op, std::unordered_map<graphlib::NodeId, FusionGroupP> &fused_nodes)
{
    remove_op(op, fused_nodes);

    for (Node *user : graph->data_users(op))
    {
        if (user->node_type() != graphlib::kBudaOp)
            continue;

        if (has_op(user->name()))
            remove_op_and_below(graph, user->as<BudaOpNode>(), fused_nodes);
    }
}

// Remove the cases where one part of the fork is inside the fused op, and the other is not, "wrapping" around
// other ops
void FusionGroup::remove_fork_wraps(Graph *graph, std::unordered_map<graphlib::NodeId, FusionGroupP> &fused_nodes)
{
    std::unordered_set<BudaOpNode *>
        cleared_nodes;  // nodes we have determined are definitely ok and can't be causing a problem
    while (nodes.size() > 1)
    {
        bool changed = false;
        for (auto &[name, op] : nodes)
        {
            if (cleared_nodes.count(op) > 0)
                continue;

            // Look for outputs that are not in the fused op
            for (Node *user : graph->data_users(op))
            {
                if (has_op(user))
                    continue;

                std::unordered_set<Node *> visited;
                BudaOpNode *converge = converge_back_on_fused_op(graph, user, visited);
                if (converge == nullptr)
                    continue;  // ok, doesn't converge

                // Need to remove converging op, and anything that uses it
                remove_op_and_below(graph, converge, fused_nodes);
                changed = true;
            }

            if (!changed)
                cleared_nodes.insert(op);
            else
                break;  // need to start the search again because the fused op has changed
        }

        // check if we're done changing
        if (!changed)
            break;
    }
}

// Sort nodes in topo order, based on the global topo order
std::vector<BudaOpNode *> extract_topo_order(
    const std::unordered_set<std::string> &nodes, const std::vector<Node *> &graph_topo)
{
    std::vector<BudaOpNode *> topo_order;
    for (Node *node : graph_topo)
        if (nodes.count(node->name()) > 0)
            topo_order.push_back(node->as<BudaOpNode>());
    TT_ASSERT(topo_order.size() == nodes.size());
    return topo_order;
}

// Generate topological order of fused ops, using graph topological order as reference
void FusionGroup::generate_topo_order(const std::vector<Node *> &graph_topo)
{
    TT_ASSERT(topo_order.size() == 0, "Topological order has already been created.");
    std::unordered_set<std::string> node_set;
    for (auto &[name, node] : nodes) node_set.insert(name);
    topo_order = extract_topo_order(node_set, graph_topo);
}

// Pick first output in topo sort, and remove all others
void FusionGroup::pick_first_output(
    Graph *graph, std::unordered_map<graphlib::NodeId, FusionGroupP> &fused_nodes, const std::vector<Node *> &topo)
{
    if (empty() || single_op())
        return;

    Node *first_output = nullptr;
    for (Node *node : topo)
    {
        if (has_op(node))
        {
            // data loopback edges must be outputs of fused ops, we can't fuse back to fwd
            if (graph->user_edges(node, [](Edge edge) { return edge.edge_type == graphlib::EdgeType::kDataLoopback; })
                    .size() > 0)
            {
                first_output = node;
            }

            if (first_output == nullptr)
            {
                for (Node *user : graph->data_users(node))
                {
                    if (!has_op(user->name()))
                    {
                        // Found first output
                        first_output = node;
                        break;
                    }
                }
            }
        }
        if (first_output != nullptr)
            break;
    }

    TT_ASSERT(first_output != nullptr, "There must be an output somewhere");
    output_op = first_output->as<BudaOpNode>();

    // Only nodes in this output's input cone are allowed.
    std::unordered_set<BudaOpNode *> node_set;
    for (auto &[name, node] : nodes) node_set.insert(node);

    auto input_cone = get_input_cone(graph, first_output->as<BudaOpNode>(), node_set);

    std::unordered_set<BudaOpNode *> to_remove;
    for (auto &[name, node] : nodes)
        if (input_cone.count(node) == 0)
            to_remove.insert(node);

    for (BudaOpNode *rm : to_remove) remove_op(rm, fused_nodes);

    // We might've created new outputs by removing nodes, so let's do it again
    if (to_remove.size() > 0)
        pick_first_output(graph, fused_nodes, topo);
}

bool op_tagged_with_fuse_disable(const BudaOpNode *node)
{
    std::vector<std::string> ops_tagged_with_fuse_disable = env_as_vector<std::string>("PYBUDA_DISABLE_FUSE_TAGS");
    if (ops_tagged_with_fuse_disable.empty())
    {
        return false;
    }

    if (node->as<graphlib::TaggedNode>()->has_tag("original_op_type"))
    {
        std::string original_op_type =
            std::get<std::string>(node->as<graphlib::TaggedNode>()->tag_value("original_op_type"));

        // If the original op type is in the list of ops to disable fusing, then disable fusing
        for (const std::string &op_type : ops_tagged_with_fuse_disable)
        {
            if (original_op_type == op_type or node->op_name() == op_type)
            {
                log_debug("Fusion disabled on node: {} because it matches with: {}", node->name(), op_type);
                return true;
            }
        }
    }

    return false;
}

// Handle all prechecks if fusing should be attempted for given node.
// Returns false if node failed prechecks, true othervise.
bool should_fuse_node(
    FusionGroupP fused_op,
    BudaOpNode *node,
    Graph *graph,
    std::unordered_map<graphlib::NodeId, FusionGroupP> &fused_nodes,
    const std::vector<std::vector<std::string>> &op_names_to_chip_break,
    const std::vector<std::vector<std::string>> &op_names_to_epoch_break)
{
    if (node->tag_value_or<bool>("dont_fuse", false))
        return false;

    // If node is already fused or fusion was already attempted for this node.
    if (fused_nodes.count(node->id()) > 0)
        return false;

    // Note: get_dram_input_count isn't perfect because op fusion happens before
    // balancing so we miss out on potential e2e queue inputs.  This only protects
    // graph inputs from being fused past the limit
    if (fused_op->get_input_count(graph, {node}) > FusedOp::kMaxNumInputs or
        fused_op->get_dram_input_count(graph, {node}) > FusedOp::kMaxNumDRAMInputs or
        fused_op->get_connection_count(graph, {node}) >= FusedOp::kMaxNumConnections)
        return false;

    // Filter out operations that by defintion shouldn't be fused.
    // Matmul (except for special cases of reduce and tile_broadcast) and buffer op
    if ((is_matmul(node) && !is_allowed_matmul(graph, node)) || is_buffer(node))
        return false;

    // These operations are not supported for fusing on backend.
    if (is_reduce_max(node) || is_splice(node) || node->is_embedding())
        return false;

    // If it is accumulation op don't fuse it.
    if (node->is_gradient_op())
        return false;

    // If user has tagged the op_type/original_op_type with a tag that should disable fusing, then disable fusing
    if (op_tagged_with_fuse_disable(node))
        return false;

    //  Don't fuse operations that are explicitly marked for chip or epoch break.
    /*
        TODO: More optimal approach would be to fuse this op and don't fuse users of this op in this fusing op run.
        That approach would increase complexity of this change due to explicit dependencies
        on chip/epoch break op name down the stack which would be changed if op is fused.
        Based on this reasoning leaving this as a follow up.
    */
    if (is_str_in_strings(node->name(), op_names_to_chip_break) ||
        is_str_in_strings(node->name(), op_names_to_epoch_break))
        return false;

    // Can't fuse ops that are in differenct epoch types.
    if (!fused_op->empty() && (fused_op->get_epoch_type() != node->get_epoch_type()))
        return false;

    uint32_t reduce_dim = 0;
    if (find_reduce_dim(node, reduce_dim))
    {
        // We cannot allow reduce ops along different dimensions in one fused op.
        if (fused_op->has_reduce_op() && reduce_dim != fused_op->get_reduce_dim())
            return false;
    }

    bool disable_broadcast = env_as<bool>("PYBUDA_FUSE_DISABLE_BROADCAST");

    // Block fusing of unsupported TMs.
    for (Edge operand_edge : graph->operand_data_edges(node))
    {
        Node *operand = graph->node_by_id(operand_edge.producer_node_id);

        // If this is matmul it is not alowed to fuse if input 1 is result of the same fused op.
        if (is_matmul(node) && !is_tile_broadcast(node) && (operand_edge.consumer_input_port_id == 1) && fused_op->has_op(operand))
        {
            return false;
        }

        // If producer op is not fused it means that current egde is not fused, hence no need to check op fusing tm
        // limits.
        if (fused_op->has_op(operand))
        {
            if (!are_allowed_tms(
                    graph->get_edge_attributes(operand_edge)->get_tms(),
                    disable_broadcast,
                    is_matmul(node),
                    operand_edge.producer_node_id))
                return false;
        }
    }

    return true;
}

void expand_search(
    FusionGroupP fused_op,
    Graph *graph,
    BudaOpNode *current_node,
    std::unordered_map<graphlib::NodeId, FusionGroupP> &fused_nodes,
    const std::vector<std::vector<std::string>> &op_names_to_chip_break,
    const std::vector<std::vector<std::string>> &op_names_to_epoch_break)
{
    //
    // search below and above for more ops to fuse
    //

    if (!should_fuse_node(fused_op, current_node, graph, fused_nodes, op_names_to_chip_break, op_names_to_epoch_break))
        return;

    fused_op->add_op(current_node);
    fused_nodes.insert(std::make_pair(current_node->id(), fused_op));

    // If this is the first reduce in this fused op remember its dimension.
    uint32_t reduce_dim = 0;
    if (find_reduce_dim(current_node, reduce_dim) && !fused_op->has_reduce_op())
        fused_op->set_reduce_op(reduce_dim);

    bool disable_broadcast = env_as<bool>("PYBUDA_FUSE_DISABLE_BROADCAST");

    for (Edge operand_edge : graph->operand_data_edges(current_node))
    {
        // Not supported to use fused result as input 1 of matmul.
        if (is_matmul(current_node) && !is_tile_broadcast(current_node) && (operand_edge.consumer_input_port_id == 1))
            continue;

        auto tms = graph->get_edge_attributes(operand_edge)->get_tms();

        // Producer op can be fused only if bellow op can consume all tms as part of fused op.
        if (!are_allowed_tms(tms, disable_broadcast, is_matmul(current_node), operand_edge.producer_node_id))
            continue;

        Node *operand = graph->node_by_id(operand_edge.producer_node_id);
        if (operand->node_type() == graphlib::kBudaOp)
            expand_search(
                fused_op,
                graph,
                operand->as<BudaOpNode>(),
                fused_nodes,
                op_names_to_chip_break,
                op_names_to_epoch_break);
    }

    // Don't go beyond exp for now, because it's expensive and we don't want to do it multiple times
    // which will happen in softmax... TODO make this more generic
    if (current_node->op_type().op == "exp")
        return;

    if (env_as<bool>("PYBUDA_FUSE_STOP_ON_RECIPROCAL") and current_node->op_type().op == "reciprocal")
        return;

    for (Edge user_edge : graph->user_data_edges(current_node))
    {
        // No need to check user tm limitation since expand_search will do all the checks.
        Node *user = graph->node_by_id(user_edge.consumer_node_id);
        if (user->node_type() == graphlib::kBudaOp)
            expand_search(
                fused_op, graph, user->as<BudaOpNode>(), fused_nodes, op_names_to_chip_break, op_names_to_epoch_break);
    }
}

static void tag_ops_dont_fuse(
    graphlib::Graph *graph,
    const std::vector<std::string> &op_names_dont_fuse,
    const std::vector<std::string> &op_names_manual_fuse)
{
    for (auto const& op_name : op_names_dont_fuse)
    {
        if (not graph->has_node_with_name(op_name))
        {
            log_warning(LogFuser, "Node name specified in op_names_dont_fuse doesn't exist in graph {}", op_name);
            continue;
        }

        graph->get_node_by_name(op_name)->as<graphlib::TaggedNode>()->tag("dont_fuse");
    }

    if (not op_names_manual_fuse.empty())
    {
        auto regex_predicate = graphlib::query::Predicate<graphlib::Node *>::anyOf(
            op_names_manual_fuse.begin(), op_names_manual_fuse.end(), graphlib::query::view_node_name);
        auto predicate = graphlib::query::predicate_op_node_type() & regex_predicate.negate();
        for (Node *node : filter_nodes(graph, predicate))
        {
            node->as<graphlib::TaggedNode>()->tag("dont_fuse");
        }
    }
}

// We skip fusing op_types that have override for output_df in amp_properties
void skip_fusing_based_on_amp_properties(graphlib::Graph *graph, const std::vector<tt::passes::AMPNodeProperties> &amp_properties)
{
    std::unordered_set<std::string> op_types_skip_fusing;
    for (const auto& amp_property : amp_properties)
    {
        if (amp_property.op_type.has_value() && amp_property.output_df.has_value())
        {
            op_types_skip_fusing.insert(amp_property.op_type.value());
        }
    }

    for (Node *node : graphlib::topological_sort(*graph))
    {
        if(node->node_type() == graphlib::NodeType::kBudaOp)
        {
            std::string op_type = node->as<graphlib::BudaOpNode>()->op_type().op;
            if (op_types_skip_fusing.find(op_type) != op_types_skip_fusing.end())
            {
                // current node has op_type which has output_df override in amp_properties
                // therefore, we mark that node with dont_fuse to skip fusing.
                node->as<graphlib::TaggedNode>()->tag("dont_fuse");
            }
        }
    }
}

void fuse_ops(
    graphlib::Graph *graph,
    const DeviceConfig &device_config,
    const std::vector<std::vector<std::string>> &op_names_to_chip_break,
    const std::vector<std::vector<std::string>> &op_names_to_epoch_break,
    const std::vector<std::string> &op_names_dont_fuse,
    const std::vector<std::string> &op_names_manual_fuse,
    const std::vector<tt::passes::AMPNodeProperties> &amp_properties)
{
    // Map of node IDs and fused groups that contain that node.
    // If fuse group is nullptr, that means that fusing was already attempted and failed for this node.
    std::unordered_map<graphlib::NodeId, FusionGroupP> fused_nodes;

    // Reinit next_fuse_id, since we could be retrying graph compilation.
    FusionGroup::next_fuse_id = 0;

    log_debug(LogFuser, "Fusing ops...");

    skip_fusing_based_on_amp_properties(graph, amp_properties);
    tag_ops_dont_fuse(graph, op_names_dont_fuse, op_names_manual_fuse);

    std::vector<FusionGroupP> fused_ops;
    FusionGroupP fused_op = std::make_shared<FusionGroup>();
    auto topo = graphlib::topological_sort(*graph);
    bool bisect_fusing = env_as<bool>("PYBUDA_BISECT_FUSING", false);
    // To start, fuse first half of the model and then edit boundaries.
    int first_ind_to_fuse = env_as<int>("PYBUDA_FUSE_OP_FIRST_IND", 0);
    int last_ind_to_fuse = env_as<int>("PYBUDA_FUSE_OP_LAST_IND", floor((topo.size() - 1)/2)); 
    int node_ind = 0;
    if (bisect_fusing)
    {
        for (Node *node : topo)
        {
            if (node_ind < first_ind_to_fuse || node_ind > last_ind_to_fuse)
            {
                log_debug(LogFuser, "skip fusing for node: {}, node index: {}", node->name(), node_ind);
                node->as<graphlib::TaggedNode>()->tag("dont_fuse");
            }
            node_ind ++;
        }
    }

    for (Node *node : topo)
    {
        if (node->node_type() == graphlib::kBudaOp)
        {
            BudaOpNode *op = node->as<BudaOpNode>();
            if (!should_fuse_node(fused_op, op, graph, fused_nodes, op_names_to_chip_break, op_names_to_epoch_break))
                continue;

            log_trace(LogFuser, "Expand search from {}", node->name());
            expand_search(
                fused_op, graph, node->as<BudaOpNode>(), fused_nodes, op_names_to_chip_break, op_names_to_epoch_break);
            log_trace(LogFuser, "Legalize fused op from {}, with {} fused ops", node->name(), fused_op->count());
            if (fused_op->legalize(graph, fused_nodes, topo))
            {
                fused_ops.push_back(fused_op);
                fused_op->print();
                fused_op = std::make_shared<FusionGroup>();
            }
        }
    }

    // Remove all tile broadcasts that can be replaced by input TM.
    std::unordered_set<Node *> to_delete_nodes;
    for (Node *node : topo)
    {
        replace_tile_broadcasts(graph, node, fused_nodes, to_delete_nodes);
    }

    // for (FusionGroupP f : fused_ops) f->print();
    std::uint32_t initial_count = graph->nodes().size();
    std::uint32_t fused_away = 0;
    for (FusionGroupP f : fused_ops) fused_away += f->count() - 1;

    log_debug(LogFuser, "Initial op count: {}, fused away: {}", initial_count, fused_away);

    // Generate fused ops topo order
    for (FusionGroupP f : fused_ops)
    {
        if (!(f->empty()))
        {
            f->generate_topo_order(topo);
        }
    }

    // Remove nodes marked for deletion by tile replace algorithm.
    for (auto op : to_delete_nodes) graph->remove_node(op);

    // Make fusing graph changes
    bool reuse_dest_on_srcA_only = device_config.is_grayskull();
    for (FusionGroupP f : fused_ops)
    {
        if (!(f->empty()))
        {
            f->fuse(graph, f, reuse_dest_on_srcA_only);
        }
    }

    // Clean up - remove any inputs that are no longer used
    for (Node *node : graph->nodes())
    {
        if ((node->node_type() == graphlib::kInput) && graph->user_edges(node).size() == 0)
            graph->remove_node(node);
    }
}

std::shared_ptr<FusedOp> FusedOp::clone(BudaOpNode *parent_buda_node)
{
    return std::make_shared<FusedOp>(this->group->clone(), parent_buda_node, this->inputs, this->output_op, this->schedules);
}

std::uint32_t FusedOp::id() const { return group->get_id(); }

std::uint32_t FusedOp::get_input_count() const
{
    std::uint32_t input_count = 0;
    for (auto sch : schedules)
    {
        for (FusedSubOp op : sch.ops)
        {
            for (auto i : op.inputs)
            {
                if ((i.type == FusedSubOpInput::InputType::INPUT) && (i.index + 1 > input_count))
                    input_count = i.index + 1;
            }
        }
    }
    return input_count;
}

// Return attributes that will be defined on fused op level.
BudaOpAttrs FusedOp::get_operation_attr()
{
    BudaOpAttrs attrs;
    attrs["fused_op_id"] = (int)id();

    if (node->buda_attrs().count("kernel_broadcast") and not env_as<bool>("PYBUDA_DISABLE_FUSED_KERNEL_BROADCAST"))
    {
        attrs["kernel_broadcast"] = node->buda_attrs().at("kernel_broadcast");
    }

    // Currently BE limitation is that approximate_mode can be specified only as fused op attribute.
    // Logic for merging is that if any sub op requires precise result, do not allow approximate mode on entire fuse op.
    bool exists = false;
    graphlib::OpType::Attr value = true;
    for (auto sch : schedules)
    {
        for (FusedSubOp op : sch.ops)
        {
            auto attr = op.op_type.buda_attrs.find("approximate_mode");
            if (attr != op.op_type.buda_attrs.end())
            {
                exists = true;
                value = attr->second;
                continue;
            }

            // Assert if op has attribute that is not supported on sub op level.
            for (auto attr : op.op_type.buda_attrs)
            {
                if ((SubOpAttr[op.op_type.op].count(attr.first) > 0) || (SubOpAttr["*"].count(attr.first) > 0))
                {
                    TT_ASSERT(
                        "Operation: {}, contains attribute: {}, that is lost in fusing.", op.op_type.op, attr.first);
                }
            }
        }
    }

    if (exists)
    {
        attrs["approximate_mode"] = value;
    }

    return attrs;
}

std::pair<std::uint32_t, std::uint32_t> FusedSubOp::get_mblock_for_ublock(
    const std::pair<std::uint32_t, std::uint32_t> ublock, const std::pair<std::uint32_t, std::uint32_t> grid) const
{
    const std::uint32_t rt = ublock.first;
    const std::uint32_t ct = ublock.second;
    const std::uint32_t grid_r = grid.first;
    const std::uint32_t grid_c = grid.second;
    const balancer::TensorShape &ts = op_shape.outputs.at(0);
    TT_ASSERT(ts.rt % grid_r == 0, "For sub-op {}, rt {} is not divisible by grid_r {}", name, ts.rt, grid_r);
    TT_ASSERT(ts.ct % grid_c == 0, "For sub-op {}, ct {} is not divisible by grid_c {}", name, ts.ct, grid_c);
    TT_ASSERT(ts.rt % (grid_r * rt) == 0, "For sub-op {}, rt {} is not divisible by ublock dim {}", name, ts.rt, rt);
    TT_ASSERT(ts.ct % (grid_c * ct) == 0, "For sub-op {}, ct {} is not divisible by ublock dim {}", name, ts.ct, ct);
    std::uint32_t m, n;
    m = ts.rt / (rt * grid_r);
    n = ts.ct / (ct * grid_c);
    return std::make_pair(m, n);
}

// Out of all buda attr return those that should be added to netlist on sub op level.
BudaOpAttrs FusedSubOp::get_sub_op_buda_attr() const
{
    BudaOpAttrs new_attr;
    for (auto attr : op_type.buda_attrs)
    {
        if ((SubOpAttr[op_type.op].count(attr.first) > 0) || (SubOpAttr["*"].count(attr.first) > 0))
        {
            new_attr.insert(attr);
        }
    }

    return new_attr;
}

FusedOp::FusedOp(
    FusionGroupP group,
    BudaOpNode *node,
    InputMapping inputs,
    BudaOpNode *output_op,
    std::vector<FusedSchedule> schedules) :
    group(group), node(node), inputs(inputs), output_op(output_op), schedules(schedules)
{
    has_matmul_ = group->has_matmul_op();
    has_reduce_ = group->has_reduce_op();
    has_broadcast_c_ = group->has_broadcast_c_tm();
    reduce_dim_ = group->get_reduce_dim();
}

}  // namespace tt
