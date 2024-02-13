// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "scheduler/scheduler.hpp"

#include <map>
#include <queue>
#include <unordered_map>
#include <stack>

#include "graph_lib/defines.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "placer/lower_to_placer.hpp"
#include "scheduler/longest_path.hpp"
#include "utils/logger.hpp"

using tt::LogScheduler;
using tt::graphlib::Node;
using tt::graphlib::NodeId;
using tt::graphlib::Edge;
using tt::graphlib::EdgeType;

namespace tt::scheduler {

Schedule run_topological_scheduler(const graphlib::Graph* graph)
{
    Schedule scheduled_nodes;
    for (graphlib::Node* node : graphlib::topological_sort(*graph))
    {
        if (can_schedule_node(node))
        {
            scheduled_nodes.push_back(node->name());
        }
    }
    return scheduled_nodes;
}

static bool requires_visit(const std::unordered_set<NodeId>& visited, NodeId node_id) {
    return visited.find(node_id) == visited.end();
}

void assert_schedule_dependencies_met(const graphlib::Graph* graph, const Schedule& schedule)
{
    std::unordered_map<std::string, std::uint32_t> node_to_schedule_index;
    node_to_schedule_index.reserve(schedule.size());
    for (std::uint32_t i = 0; i < schedule.size(); ++i)
    {
        node_to_schedule_index[schedule[i]] = i;
    }

    for (const std::string& op : schedule)
    {
        Node* node = graph->get_node_by_name(op);
        for (const Node* predecessor_node : get_schedule_predecessors(graph, node))
        {
            if (node_to_schedule_index.find(predecessor_node->name()) != node_to_schedule_index.end())
            {
                TT_LOG_ASSERT(
                    node_to_schedule_index[predecessor_node->name()] < node_to_schedule_index[op],
                    "Scheduler: dependency not met for node: {}",
                    op);
            }
        }
    }
}

void assert_valid_schedule(const graphlib::Graph* graph, const Schedule& schedule) {
    // basic check: verify all nodes have been placed using topo as baseline
    Schedule topo_schedule = run_topological_scheduler(graph);

    std::unordered_set<std::string> scheduled_set;
    for (auto op : schedule) {
        scheduled_set.insert(op);
    }
    for (auto op : topo_schedule) {
        if (scheduled_set.find(op) == scheduled_set.end()) {
            log_warning("{} not found in the scheduled_set.", op);
        }
    }

    if (schedule.size() != topo_schedule.size())
    {
        std::vector<std::string> diff;
        auto copy = schedule;
        std::sort(copy.begin(), copy.end());
        std::sort(topo_schedule.begin(), topo_schedule.end());
        std::set_symmetric_difference(
            copy.begin(), copy.end(), topo_schedule.begin(), topo_schedule.end(), std::back_inserter(diff));
        log_error("Scheduler: not all nodes have been placed using current policy.");
        for (auto name : diff) log_error("  {} {}", name, graph->get_node_by_name(name)->get_type());
        log_fatal("Scheduler: not all nodes have been placed using current policy.");
    }
    assert_schedule_dependencies_met(graph, schedule);
}

static void push_to_node_queue(
    const graphlib::Graph* graph,
    std::queue<Node*>& node_queue,
    const std::vector<std::string>& nodes)
{
    for (const std::string& name : nodes) {
        log_debug(LogScheduler, "Running BFS from module input: {}", name);
        node_queue.push(graph->get_node_by_name(name));
    }
}

using NodeGroup = std::unordered_set<std::string>;
using NodeGroupVector = std::vector<NodeGroup>;
NodeGroup add_group(const graphlib::Graph* graph, const std::string& op)
{
    NodeGroup group;
    Node* node = graph->get_node_by_name(op);
    for (const Edge& operand_edge : graph->operand_data_edges(node))
    {
        if (operand_edge.edge_type == EdgeType::kDataLoopback or
            operand_edge.edge_type == EdgeType::kControlLoop)
        {
            continue;
        }
        Node* predecessor_node = graph->node_by_id(operand_edge.producer_node_id);
        group.insert(predecessor_node->name());
    }
    log_debug(LogScheduler, "Creating group:");
    for (const auto& n : group)
    {
        log_debug(LogScheduler, "\t op: {}", n);
    }

    return group;
}

NodeGroupVector create_groups(const graphlib::Graph* graph, const std::vector<std::string>& ops)
{
    NodeGroupVector groups;
    for (const auto& op : ops)
    {
        groups.emplace_back(add_group(graph, op));
    }
    return groups;
}

std::vector<std::string> discover_ops_for_grouped_inputs(const graphlib::Graph* graph)
{
    constexpr int HEURISTIC_FOR_NUM_INPUTS = 6;
    const std::unordered_set<std::string> target_op_types = {"concatenate", "splice"};

    std::vector<std::string> ops_for_grouped_inputs;
    for (graphlib::Node* node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() == graphlib::NodeType::kPyOp or node->node_type() == graphlib::NodeType::kBudaOp)
        {
            const std::string& op_type = node->as<graphlib::OpNode>()->op_name();
            if (target_op_types.find(op_type) != target_op_types.end())
            {
                int data_inputs = 0;
                for (const Edge& operand_edge : graph->operand_data_edges(node))
                {
                    if (operand_edge.edge_type == EdgeType::kDataLoopback or
                        operand_edge.edge_type == EdgeType::kControlLoop)
                    {
                        continue;
                    }
                    data_inputs += 1;
                }

                if (data_inputs >= HEURISTIC_FOR_NUM_INPUTS)
                {
                    ops_for_grouped_inputs.push_back(node->name());
                }
            }
        }
    }
    for (const auto& op : ops_for_grouped_inputs)
    {
        log_debug(LogScheduler, "Discovered: {}", op);
    }
    return ops_for_grouped_inputs;
}

static void add_schedule_dependencies(
    std::unordered_map<std::string, std::string>& schedule_dependencies,
    const std::vector<std::string>& partial_ordering)
{
    for (size_t i = 1; i < partial_ordering.size(); i++)
    {
        schedule_dependencies[partial_ordering[i]] = partial_ordering[i - 1];
    }
}

// schedule partial data-copies as early as possible. soft-constraint
static std::vector<std::vector<std::string>> get_schedule_constraints_for_partial_data_copies(
    const graphlib::Graph* graph)
{
    std::vector<std::vector<std::string>> partial_orderings;
    for (Node* partial_datacopy_output : graph->ordered_partial_datacopy_outputs())
    {
        // find the op that performs the read from the write-back input/parameter
        auto loopback_edge = graph->user_edges(partial_datacopy_output).at(0);
        auto loopback_input = graph->node_by_id(loopback_edge.consumer_node_id);

        std::vector<std::string> write_back_path;
        for (auto node : graphlib::subgraph(graph, loopback_input, partial_datacopy_output))
        {
            write_back_path.push_back(node->name());
        }

        if (not write_back_path.empty())
        {
            std::reverse(write_back_path.begin(), write_back_path.end());  // graphlib::subgraph in rev order
            partial_orderings.push_back(write_back_path);

            auto writer = graph->get_node_by_name(write_back_path.front());
            for (Node* user : graph->data_users(writer))
            {
                if (std::find(write_back_path.begin(), write_back_path.end(), user->name()) == write_back_path.end())
                {
                    partial_orderings.push_back({write_back_path.back(), user->name()});
                }
            }
        }
    }
    log_trace(LogScheduler, "\t Partial ordering inserted from data-copy-writeback", partial_orderings);
    return partial_orderings;
}

static std::vector<std::vector<std::string>> get_constraints_from_control_edges(const graphlib::Graph* graph)
{
    std::vector<std::vector<std::string>> partial_orderings;
    for (auto edge : graph->edges(graphlib::EdgeType::kControl))
    {
        Node* producer = graph->node_by_id(edge.producer_node_id);
        Node* consumer = graph->node_by_id(edge.consumer_node_id);
        if (can_schedule_node(producer) and can_schedule_node(consumer))
        {
            partial_orderings.push_back({producer->name(), consumer->name()});
        }
    }
    log_trace(LogScheduler, "\t Partial ordering inserted from control edges: {}", partial_orderings);
    return partial_orderings;
}

static std::unordered_map<std::string, std::string> get_schedule_dependencies(
    const SchedulerConfig& config, const graphlib::Graph* graph)
{
    std::unordered_map<std::string, std::string> schedule_dependencies;
    for (const auto& partial_ordering : config.scheduler_constraints)
    {
        add_schedule_dependencies(schedule_dependencies, partial_ordering);
    }
    for (const auto& partial_ordering : get_schedule_constraints_for_partial_data_copies(graph))
    {
        add_schedule_dependencies(schedule_dependencies, partial_ordering);
    }
    for (const auto& partial_ordering : get_constraints_from_control_edges(graph))
    {
        add_schedule_dependencies(schedule_dependencies, partial_ordering);
    }
    return schedule_dependencies;
}

static std::vector<graphlib::NodeId> get_operand_node_ids(
    const std::unordered_map<std::string, std::string>& schedule_dependencies,
    const graphlib::Graph* graph,
    const graphlib::Node* node)
{
    std::vector<graphlib::NodeId> operand_node_ids;
    for (const Edge& operand_edge : graph->operand_data_edges(node)) {
        if (operand_edge.edge_type == EdgeType::kDataLoopback or
            operand_edge.edge_type == EdgeType::kControlLoop)
        {
            continue;
        }
        operand_node_ids.push_back(operand_edge.producer_node_id);
    }
    if (schedule_dependencies.find(node->name()) != schedule_dependencies.end())
    {
        const std::string& dependency = schedule_dependencies.at(node->name());
        Node* dependency_node = graph->get_node_by_name(dependency);
        operand_node_ids.push_back(dependency_node->id());
    }
    return operand_node_ids;
}

// For a given parrent_node, fetch paired node if it exists so that they can be scheduled together.
// Note that valid paired node should have only parrent_node and inputs as operands.
//
const graphlib::Node* get_paired_op_if_exists(const graphlib::Graph* graph, const graphlib::Node* parrent_node)
{
    const graphlib::Node* paired_node = nullptr;
    if (parrent_node->node_type() == graphlib::NodeType::kBudaOp)
    {
        const graphlib::BudaOpNode* op_node = static_cast<const graphlib::BudaOpNode*>(parrent_node);

        // Sparse-dense pair case.
        //
        if (op_node->is_sparse_matmul())
        {
            std::vector<tt::graphlib::Node*> users = graph->data_users(op_node);
            if (users.size() == 1 && users[0]->node_type() == graphlib::kBudaOp)
            {
                const graphlib::BudaOpNode* user_op_node = static_cast<const graphlib::BudaOpNode*>(users[0]);
                if (user_op_node->should_pair_with_sparse(op_node, graph))
                {
                    paired_node = user_op_node;
                }
            }
        }
    }

#ifdef DEBUG
    // Check that paired node has only parrent_node and inputs as operands.
    //
    if (paired_node != nullptr)
    {
        for (const Node* operand_node : graph->data_operands(paired_node))
        {
            TT_ASSERT(operand_node == parrent_node || operand_node->node_type() == graphlib::NodeType::kInput);
        }
    }
#endif

    TT_ASSERT(!paired_node or can_schedule_node(paired_node));

    return paired_node;
}

Schedule run_scheduler(
    const SchedulerConfig& config,
    const graphlib::Graph* graph,
    std::queue<Node*>& node_queue,
    std::unordered_set<NodeId>& visited,
    const NodeGroupVector& groups)
{
    (void)groups;
    std::unordered_map<std::string, std::string> schedule_dependencies = get_schedule_dependencies(config, graph);
    Schedule scheduled_nodes;

    static const bool disable_fj_nop_schedule_fix = env_as<bool>("PYBUDA_TEMP_DISABLE_FJ_NOP_SCHEDULE_FIX");

    // declare a function to handle fracture groups
    std::function<void(Node*)> VisitNode;
    std::function<bool(Node*)> FracVisit = [&](Node* node) -> bool{

        // this fracture group has to be scheduled contiguously in DFS fashion
        // get the fracture group id of the node
        auto fracture_group_id = node->as<graphlib::TaggedNode>()->tag_value("fracture_group_id");

        // collect the nodes in this fracture group
        // also collect the tops and bottoms
        std::unordered_set<NodeId> fracture_group_nodes;
        std::vector<NodeId> fracture_group_tops;
        std::vector<NodeId> fracture_group_bottoms;
        for (auto& node : graph->nodes()) {
            if (node->as<graphlib::TaggedNode>()->has_tag("fracture_group_id")) {
                if (node->as<graphlib::TaggedNode>()->tag_value("fracture_group_id") != fracture_group_id) {
                    continue;
                }
                fracture_group_nodes.insert(node->id());
            }
            if (node->as<graphlib::TaggedNode>()->has_tag("fracture_top")) {
                fracture_group_tops.push_back(node->id());
            }
            if (node->as<graphlib::TaggedNode>()->has_tag("fracture_bottom")) {
                fracture_group_bottoms.push_back(node->id());
            }
        }

        // ensure that the fracture group is allowed to be scheduled by the compiler
        // fracture nodes can participate in schedule dependencies only if all the fracture group nodes are tops or bottoms
        bool fracture_group_allowed = true;
        // iterate over the schedule dependencies
        for (auto& [op, dep] : schedule_dependencies) {
            // if neither op nor dep are in the fracture group, then skip
            if (fracture_group_nodes.find(graph->get_node_by_name(op)->id()) == fracture_group_nodes.end() and
                fracture_group_nodes.find(graph->get_node_by_name(dep)->id()) == fracture_group_nodes.end()) {
                continue;
            }

            // make sure that both op and dep are not in the fracture group
            if (fracture_group_nodes.find(graph->get_node_by_name(op)->id()) != fracture_group_nodes.end() or
                fracture_group_nodes.find(graph->get_node_by_name(dep)->id()) != fracture_group_nodes.end()) {
                fracture_group_allowed = false;
                break;
            }

            // if op is a fracture group top, or dep is a fracture group bottom, then compiler cannot schedule this
            if (graph->get_node_by_name(op)->as<graphlib::TaggedNode>()->has_tag("fracture_top") or
                graph->get_node_by_name(dep)->as<graphlib::TaggedNode>()->has_tag("fracture_bottom")) {
                fracture_group_allowed = false;
                break;
            }
        }

        // if the fracture group is allowed to be scheduled by the compiler
        if (not fracture_group_allowed)  return false;

        // create a DFS stack with the fracture group top
        std::stack<Node*> frac_node_stack;

        // define a function to run a DFS scheduler on fracture group nodes
        std::function<void(void)> DFSFrac = [&]() {
            // return if stack is empty
            if (frac_node_stack.empty()) {
                return;
            }

            // pop a node from the stack
            Node* stk = frac_node_stack.top();
            frac_node_stack.pop();

            // add this node to the schedule
            if (can_schedule_node(stk)){
                scheduled_nodes.push_back(stk->name());
            } else return; // not a schedulable node

            if (requires_visit(visited, stk->id())) {
                // mark this node as visited
                visited.insert(stk->id());
            }

            // if the node is a fracture group bottom, then return
            if (stk->as<graphlib::TaggedNode>()->has_tag("fracture_bottom")) {
                return;
            }

            // iterate over the users of this node and stack them if
            // all the operands of the users have been visited
            bool all_operands_visited = true;
            for (const Edge& user_edge : graph->user_data_edges(stk)) {
                NodeId successor_id = user_edge.consumer_node_id;
                Node* successor_node = graph->node_by_id(successor_id);

                // if the successor node is not in the fracture group, then skip it
                if (fracture_group_nodes.find(successor_node->id()) == fracture_group_nodes.end()) {
                    continue;
                }

                // check if all of the operands of the successor have been visited
                auto predecessors = graph->operands(successor_node);
                // iterate over predecessors
                for (auto& pred : predecessors) {
                    // if the predecessor cannot be scheduled, then skip it
                    if (not can_schedule_node(pred)) {
                        continue;
                    }

                    // if the predecessor is already scheduled, then skip it
                    if (std::find(scheduled_nodes.begin(), scheduled_nodes.end(), pred->name()) != scheduled_nodes.end()) {
                        continue;
                    }
                    all_operands_visited = false;
                    break;
                }
                if (not all_operands_visited) {
                    continue;
                }
                // push the successor node to the stack and visit
                frac_node_stack.push(successor_node);
                DFSFrac();
            }
        };

        // now loop over the fracture group tops and visit them
        for (auto& top : fracture_group_tops) {
            // make sure all operands of the top have been visited
            // visiting them inside this loop, rather than outside
            // to keep these operands close to their users
            // fork nodes delivering input parameters fall in this category
            auto predecessors = graph->operands(graph->node_by_id(top));
            // iterate over predecessors
            for (auto pred : predecessors) {
                // if the predecessor cannot be scheduled, then skip it
                if (not can_schedule_node(pred)) {
                    continue;
                }

                // if the predecessor is already scheduled, then skip it
                if (std::find(scheduled_nodes.begin(), scheduled_nodes.end(), pred->name()) != scheduled_nodes.end()) {
                    continue;
                }

                // otherwise, visit the predecessor
                VisitNode(pred);
            }
            frac_node_stack.push(graph->node_by_id(top));
            DFSFrac();
        }

        // enqueue the successors of fracture group bottoms
        for (auto& bottom : fracture_group_bottoms) {
            for (const Edge& user_edge : graph->user_data_edges(graph->node_by_id(bottom))) {
                NodeId successor_id = user_edge.consumer_node_id;
                Node* successor_node = graph->node_by_id(successor_id);
                node_queue.push(successor_node);
            }
        }
        return true;
    };

    std::function<void(Node*)> AddChildNodes = [&](Node* node)
    {
        // Skip adding buffering nops, we want them last in schedule, otherwise they might cause early epoch breaks.
        // They will get traversed when join node is visited, as the algorithm will traverse up the graph since not all
        // producers of the join node have been visited.
        //
        const std::uint32_t num_users = graph->user_data_edges(node).size();
        std::uint32_t buffering_nop_users = 0;

        for (const Edge& user_edge : graph->user_data_edges(node))
        {
            Node* user = graph->node_by_id(user_edge.consumer_node_id);
            if (!disable_fj_nop_schedule_fix and num_users > 1 and user->node_type() == graphlib::NodeType::kBudaOp and
                user->as<graphlib::BudaOpNode>()->is_buffering_op())
            {
                buffering_nop_users++;
                continue;
            }
            node_queue.push(user);
        }

        // If buffering_nop_users == graph->user_data_edges(node).size(), we fallback to adding all users
        //
        if (!disable_fj_nop_schedule_fix and num_users > 0 and buffering_nop_users == num_users)
        {
            for (const Edge& user_edge : graph->user_data_edges(node))
            {
                node_queue.push(graph->node_by_id(user_edge.consumer_node_id));
            }
        }
    };

    VisitNode = [&](Node* node)
    {
        if (not requires_visit(visited, node->id()))
        {
            return;
        }
        visited.insert(node->id());
        for (NodeId predecessor_id : get_operand_node_ids(schedule_dependencies, graph, node))
        {
            Node* predecessor_node = graph->node_by_id(predecessor_id);
            VisitNode(predecessor_node);
        }

        // if the node is a fracture group top, then call the fracture group scheduler
        if (node->as<graphlib::TaggedNode>()->has_tag("fracture_top"))
        {
            // check if this has already been scheduled, because there are multiple tops in a fracture region, and only
            // the first top needs to be called
            if (std::find(scheduled_nodes.begin(), scheduled_nodes.end(), node->name()) != scheduled_nodes.end())
            {
                return;
            }
            auto scheduled = FracVisit(node);
            if (scheduled)
                return;
        }

        // Get paired op if it exists so that we can schedule it right after the current op.
        //
        const Node* paired_node = get_paired_op_if_exists(graph, node);

        if (can_schedule_node(node) and
            (std::find(scheduled_nodes.begin(), scheduled_nodes.end(), node->name()) == scheduled_nodes.end()))
        {
            scheduled_nodes.push_back(node->name());

            // Schedule paired op right after the current op.
            //
            if (paired_node != nullptr and
                (std::find(scheduled_nodes.begin(), scheduled_nodes.end(), paired_node->name()) ==
                 scheduled_nodes.end()))
            {
                scheduled_nodes.push_back(paired_node->name());
            }
        }

        // Add users to queue
        //
        AddChildNodes(node);
    };

    while (not node_queue.empty())
    {
        Node* node = node_queue.front();
        VisitNode(node);
        node_queue.pop();
    }

    return scheduled_nodes;
}

std::unordered_set<NodeId> get_visited_with_recompute_nodes_marked(
    const graphlib::Graph* graph,
    const std::unordered_set<NodeId>& visited
)
{
    std::unordered_set<NodeId> visited_with_recompute_marked = visited;

    for (graphlib::Node* node : graphlib::topological_sort(*graph))
    {
        if (graphlib::is_recompute(graph, node))
        {
            visited_with_recompute_marked.insert(node->id());
        }
    }

    return visited_with_recompute_marked;
}

std::vector<std::string> get_valid_schedule(const graphlib::Graph* graph, const vector<string>& schedule)
{
    std::unordered_map<std::string, int> node_to_schedule_index = get_op_to_schedule_index(schedule);
    tt::ordered_map<std::string, std::uint32_t> node_to_indegree;
    for (const std::string& op : schedule)
    {
        std::uint32_t indegree = 0;
        Node* node = graph->get_node_by_name(op);
        for (const Node* predecessor_node : get_schedule_predecessors(graph, node))
        {
            if (node_to_schedule_index.find(predecessor_node->name()) != node_to_schedule_index.end())
            {
                indegree += 1;
            }
        }
        node_to_indegree[op] = indegree;
    }

    std::deque<std::string> ops_to_process;
    for (const std::string& op : schedule)
    {
        if (node_to_indegree[op] == 0)
        {
            ops_to_process.push_back(op);
        }
    }

    std::vector<std::string> valid_schedule;
    valid_schedule.reserve(schedule.size());
    while (not ops_to_process.empty())
    {
        std::string op = ops_to_process.front();
        ops_to_process.pop_front();
        valid_schedule.push_back(op);
        Node* node = graph->get_node_by_name(op);

        for (const Node* successor_node : get_schedule_successors(graph, node))
        {
            if (node_to_schedule_index.find(successor_node->name()) != node_to_schedule_index.end())
            {
                node_to_indegree[successor_node->name()] -= 1;
                if (node_to_indegree[successor_node->name()] == 0)
                {
                    ops_to_process.push_back(successor_node->name());
                }
            }
        }
    }
    TT_LOG_ASSERT(
        valid_schedule.size() == schedule.size(), "Valid schedule size does not match original schedule size");

    return valid_schedule;
}

struct BackwardOpInfo
{
    std::string name;
    int schedule_index;

    BackwardOpInfo(const std::string& name, int schedule_index) : name(name), schedule_index(schedule_index) {}
    bool operator<(const BackwardOpInfo& rhs) { return this->schedule_index < rhs.schedule_index; }
};

unordered_map<string, vector<string>> get_ordered_fwd_to_bwd_ops(
    const graphlib::Graph* graph, const vector<string>& original_schedule)
{
    auto fwd_to_bwd_nodes = ::tt::placer::lowering::get_fwd_to_bwd_nodes(graph);
    std::unordered_map<std::string, std::vector<std::string>> fwd_to_bwd_ops_to_place;
    std::unordered_set<std::string> visited_ops;
    std::unordered_map<std::string, int> op_to_schedule_index = get_op_to_schedule_index(original_schedule);

    for (int i = original_schedule.size() - 1; i >= 0; --i)
    {
        const auto& fwd_node_name = original_schedule[i];
        if (fwd_to_bwd_nodes.find(fwd_node_name) != fwd_to_bwd_nodes.end())
        {
            fwd_to_bwd_ops_to_place[fwd_node_name] = {};
            vector<BackwardOpInfo> bwd_node_placement_order;
            for (const string& bwd_node_name : fwd_to_bwd_nodes.at(fwd_node_name))
            {
                if (visited_ops.find(bwd_node_name) == visited_ops.end())
                {
                    bwd_node_placement_order.emplace_back(bwd_node_name, op_to_schedule_index.at(bwd_node_name));
                    visited_ops.insert(bwd_node_name);
                }
            }
            std::sort(bwd_node_placement_order.begin(), bwd_node_placement_order.end());
            for (const auto& bwd_op_info : bwd_node_placement_order)
            {
                fwd_to_bwd_ops_to_place[fwd_node_name].push_back(bwd_op_info.name);
            }
        }
    }
    return fwd_to_bwd_ops_to_place;
}

// Instead of naively placing the backward ops via module-first/topological ordering, we know a better schedule would
// place the backward ops in the reverse order of the forward ops to back-propagate the gradients and weight updates.
// - While we do have a "fwd->bwd" mapping that provides information on which backward ops are associated with a
//   forward, it's not reliable to directly use this mapping to place the backward ops because it's non-trivial to
//   maintain this mapping as the graph is mutated by transformations. Instead, we'll treat this mapping as a hint to
//   guide the scheduling of the backward graph and defer the responsibility of guarding against data-dependency
//   violations to the `get_valid_schedule` function.
// - The reason it's a better schedule and produces less e2e queues is because the mapping serves as a higher-level
//   organization/grouping of the nodes we know should cluster and be associated together when scheduling.
std::vector<std::string> optimize_bwd_schedule(
    const graphlib::Graph* graph,
    const std::vector<std::string>& original_schedule,
    const std::vector<std::string>& fwd_schedule)
{
    std::vector<string> optimized_bwd_schedule;

    auto fwd_to_bwd_ops = get_ordered_fwd_to_bwd_ops(graph, original_schedule);
    for (auto it = fwd_schedule.rbegin(); it != fwd_schedule.rend(); ++it)
    {
        const string& fwd_op = *it;
        for (const string& bwd_op_name : fwd_to_bwd_ops.at(fwd_op))
        {
            optimized_bwd_schedule.push_back(bwd_op_name);
        }
    }
    return get_valid_schedule(graph, optimized_bwd_schedule);
}

std::vector<std::string> optimize_schedule(const graphlib::Graph* graph, const std::vector<std::string>& scheduled_ops)
{
    auto fwd_schedule = get_filtered_schedule(graph, scheduled_ops, NodeEpochType::Forward);
    auto bwd_schedule = optimize_bwd_schedule(graph, scheduled_ops, fwd_schedule);
    auto opt_schedule = get_filtered_schedule(graph, scheduled_ops, NodeEpochType::Optimizer);

    std::vector<std::string> optimized_schedule;
    optimized_schedule.reserve(scheduled_ops.size());
    optimized_schedule.insert(std::end(optimized_schedule), std::begin(fwd_schedule), std::end(fwd_schedule));
    optimized_schedule.insert(std::end(optimized_schedule), std::begin(bwd_schedule), std::end(bwd_schedule));
    optimized_schedule.insert(std::end(optimized_schedule), std::begin(opt_schedule), std::end(opt_schedule));

    return optimized_schedule;
}

std::vector<std::string> move_output_ops_to_end(const graphlib::Graph* graph, const std::vector<std::string>& scheduled_ops)
{
    std::vector<std::string> new_schedule;
    std::vector<std::string> output_ops;
    for(auto& op_name: scheduled_ops)
    {
        Node* op_node = graph->get_node_by_name(op_name);
        auto consumers = graph->users(op_node);
        bool feeds_graph_output_queue = std::all_of(consumers.begin(), consumers.end(), [](Node* n) { return n->node_type() == graphlib::NodeType::kOutput; });
        if(feeds_graph_output_queue)
        {
            output_ops.push_back(op_name);
        }
        else
        {
            new_schedule.push_back(op_name);
        }
    }
    for(auto& output_op_name: output_ops)
    {
        new_schedule.push_back(output_op_name);
    }
    return new_schedule;
}

Schedule run_module_by_module_scheduler(const SchedulerConfig& config, const graphlib::Graph* graph)
{
    Schedule scheduled_nodes;

    std::unordered_set<NodeId> visited;
    std::queue<Node*> node_queue;

    NodeGroupVector groups = create_groups(graph, discover_ops_for_grouped_inputs(graph));

    push_to_node_queue(graph, node_queue, graph->get_ordered_input_names());
    Schedule fwd_schedule = run_scheduler(config, graph, node_queue, visited, groups);
    scheduled_nodes.insert(std::end(scheduled_nodes), std::begin(fwd_schedule), std::end(fwd_schedule));

    push_to_node_queue(graph, node_queue, graph->get_ordered_output_gradient_names());
    auto visited_with_recompute_marked = get_visited_with_recompute_nodes_marked(graph, visited);
    Schedule temp = run_scheduler(config, graph, node_queue, visited_with_recompute_marked, groups);

    push_to_node_queue(graph, node_queue, graph->get_ordered_output_gradient_names());
    push_to_node_queue(graph, node_queue, temp);
    Schedule temp2 = run_scheduler(config, graph, node_queue, visited, groups);
    scheduled_nodes.insert(std::end(scheduled_nodes), std::begin(temp2), std::end(temp2));


    // sort the schedule based on fwd/bwd/opt
    std::stable_sort(std::begin(scheduled_nodes), std::end(scheduled_nodes),
            [&graph](const std::string& a, const std::string& b) {
            Node* node_a = graph->get_node_by_name(a);
            Node* node_b = graph->get_node_by_name(b);

            return (int)node_a->get_epoch_type() < (int)node_b->get_epoch_type();
        });

    Schedule optimized_schedule = optimize_schedule(graph, scheduled_nodes);
    assert_valid_schedule(graph, optimized_schedule);
    return optimized_schedule;
}

SchedulerPolicy policy_from_string(const std::string& policy_str)
{
    if (policy_str == "Topological") {
        return SchedulerPolicy::Topological;
    } else if (policy_str == "ModuleInputsBFS") {
        return SchedulerPolicy::ModuleInputsBFS;
    } else if (policy_str == "LongestPath") {
        return SchedulerPolicy::LongestPath;
    }

    log_error(LogScheduler, "Failed to parse scheduler policy from string: {}", policy_str);
    log_error(LogBalancer, "Falling back to SchedulerPolicy::ModuleInputsBFS");
    return SchedulerPolicy::ModuleInputsBFS;
}

std::ostream& operator<<(std::ostream& stream, SchedulerPolicy scheduler_policy) {
    switch (scheduler_policy) {
        case SchedulerPolicy::Topological: stream << "SchedulerPolicy::Topological"; break;
        case SchedulerPolicy::ModuleInputsBFS: stream << "SchedulerPolicy::ModuleInputsBFS"; break;
        case SchedulerPolicy::LongestPath: stream << "SchedulerPolicy::LongestPath"; break;
        default: stream << "SchedulerPolicy::Unknown"; break;
    }
    return stream;
}

// In future, we can use the balancer to help guide decisions about how to schedule ops.
// For now, just implementing naive baseline.
//
// If we need the schedule in multiple parts of the compile, we can either
// 1. cache the schedule and fetch it off of an obj. like the graph (less intrusive)
// 2. explicitly embed a new ScheduleEdgeType to impose scheduling dependecies
//    directly on the graph (more intrusive)
Schedule run_scheduler(const SchedulerConfig& config, const graphlib::Graph* graph)
{
    log_debug(LogScheduler, "Running Scheduler with Policy: {}", config.policy);
    Schedule schedule;

    if (not config.scheduler_constraints.empty())
    {
        log_debug(LogScheduler, "Running Scheduler with constraints: {}", config.scheduler_constraints);
    }

    if (config.policy == SchedulerPolicy::Topological)
    {
        schedule = run_topological_scheduler(graph);
    }
    else if (config.policy == SchedulerPolicy::ModuleInputsBFS)
    {
        schedule = run_module_by_module_scheduler(config, graph);
    }
    else if (config.policy == SchedulerPolicy::LongestPath)
    {
        schedule = run_longest_path_scheduler(graph);
    }
    else
    {
        log_fatal("providing unknown scheduler policy.");
    }

    if(env_as<bool>("PYBUDA_NEBULA_GALAXY_PLACER"))
    {
        schedule = move_output_ops_to_end(graph, schedule);
    }

    // Remove all already processed nodes.
    //
    if (config.ignored_nodes)
    {
        Schedule final_schedule;
        final_schedule.reserve(schedule.size() - config.ignored_nodes->size());
        for (const std::string& node_name : schedule)
        {
            if (config.ignored_nodes->count(graph->get_node_by_name(node_name)) == 0)
            {
                final_schedule.push_back(node_name);
            }
        }

        TT_ASSERT(final_schedule.size() == schedule.size() - config.ignored_nodes->size());
        schedule.swap(final_schedule);
    }

    log_schedule(schedule);
    return schedule;
}

}  // end namespace tt::scheduler
