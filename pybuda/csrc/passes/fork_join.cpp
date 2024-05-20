// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/fork_join.hpp"

#include <cmath>
#include <string>
#include <queue>
#include <unordered_map>

#include "buda_passes.hpp"
#include "graph_lib/edge.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "passes/eth_stream_reduction.hpp"
#include "post_placer_buda_passes.hpp"
#include "reportify/reportify.hpp"
#include "utils/logger.hpp"
#include "utils/ordered_associative_containers/ordered_map.hpp"

#include "pybind11/pybind11.h"

namespace tt
{

using Node = graphlib::Node;
using Graph = graphlib::Graph;
using NodeId = graphlib::NodeId;
// Trace fork BFS until join
using NodeMap = std::unordered_map<Node *, std::pair<Node *, std::uint32_t>>;

void PyInsertionInstruction::insert(graphlib::Graph *graph)
{
    PYBIND11_OVERRIDE_PURE(
        void,                 /* Return type */
        InsertionInstruction, /* Parent class */
        insert,               /* Name of function in C++ (must match Python name) */
        graph                 /* Argument(s) */
    );
}

InsInstructionUniqueId PyInsertionInstruction::unique_id() const
{
    PYBIND11_OVERRIDE_PURE(
        InsInstructionUniqueId, /* Return type */
        InsertionInstruction,   /* Parent class */
        unique_id,              /* Name of function in C++ (must match Python name) */
    );
}

struct CurrentState
{
    Node *fork;
    std::vector<Node *> current_nodes;
    NodeMap parents;
    std::unordered_map<Node *, std::uint32_t> node_depth;
};

using ForkJoin = std::pair<std::vector<Node *>, std::vector<Node *>>;

std::vector<Node *> remove_duplicates_and_outputs(std::vector<Node *> nodes, graphlib::NodeEpochType epoch_type)
{
    std::vector<Node *> ret;
    for (std::size_t i = 0; i < nodes.size(); i++)
    {
        bool ok = (nodes[i]->get_epoch_type() == epoch_type) && (nodes[i]->node_type() != graphlib::NodeType::kOutput);
        if (ok)
        {
            for (std::size_t j = i + 1; j < nodes.size(); j++)
                if (nodes[i] == nodes[j])
                {
                    ok = false;
                    break;
                }
        }
        if (ok)
            ret.push_back(nodes[i]);
    }
    return ret;
}

void record_fork_join(
    Node *fork,
    Node *join_point,
    Node *parent0,
    Node *parent1,
    const NodeMap &parents,
    std::vector<ForkJoin> &fork_joins)
{
    std::vector<Node *> path0 = {join_point, parent0};
    std::vector<Node *> path1 = {join_point, parent1};

    try
    {
        while (path0.back() != fork)
        {
            path0.push_back(parents.at(path0.back()).first);
        }
        while (path1.back() != fork)
        {
            path1.push_back(parents.at(path1.back()).first);
        }
    }
    catch (std::out_of_range &e)
    {
        TT_THROW("Missing parent when traversing back to fork point.");
    }

    // We'll find sub-forks, which should be thrown out - they will be found later from that fork spot
    std::set<Node *> path0_set = {path0.begin() + 1, path0.end() - 1};
    for (Node *node : path1)
        if (path0_set.count(node) > 0)
            return;

    std::reverse(path0.begin(), path0.end());
    std::reverse(path1.begin(), path1.end());

    fork_joins.push_back(std::make_pair(path0, path1));
}

void print_fork_join(const ForkJoin &fj)
{
    std::cout << "Fork / Join found" << std::endl;
    std::cout << "Fork at: " << fj.first[0]->name() << std::endl;
    std::cout << " Path0: " << std::endl;
    for (Node *node : fj.first) std::cout << "   - " << node->name() << std::endl;
    std::cout << " Path1: " << std::endl;
    for (Node *node : fj.second) std::cout << "   - " << node->name() << std::endl;
}

void trace_fork(const Graph *graph, Node *fork, std::vector<ForkJoin> &fork_joins, graphlib::NodeEpochType epoch_type)
{
    // The strategy is to traverse all forks and count the size of consumed inputs to make fwd progress. The difference
    // when joined is how much we need to buffer to make sure that long side can make full fwd progress and max speed.

    // This needs to be after balancer, so we know input/output blocks and bws per op.

    // Best reference I could find:
    // https://cs.stackexchange.com/questions/57221/efficient-algorithms-for-identifying-the-diamond-forkjoin-vertices-and-the-diam

    // We can probably do something simpler, since forks in graphs are relatively rare. On each fork, do a step-by-step
    // BFS search on both forks until one "dies" (i.e. reaches an output), or we find a common point. Keep track of the
    // path along the way. This is quite inefficient for large graphs with lots of complex fork/joins, but those should
    // be extremely rare in AI models.

    std::vector<Node *> data_users = remove_duplicates_and_outputs(graph->data_users(fork), epoch_type);
    if (data_users.size() == 1)
        return;  // this is not the fork you're looking for

    // Initial state
    NodeMap parents = NodeMap();
    std::unordered_map<Node *, std::uint32_t> node_depth;
    // std::cout << "trace fork: " << fork->name() << std::endl;

    std::vector<CurrentState> states(data_users.size());
    for (std::size_t i = 0; i < data_users.size(); i++)
    {
        Node *user = data_users[i];
        // std::cout << "Initializing child: " << user->name() << " at depth 1" << std::endl;
        states[i].fork = fork;
        states[i].current_nodes = {user};
        states[i].parents[user] = std::make_pair(fork, 1);
        states[i].node_depth[user] = 1;
    }

    std::string indent = " ";
    bool done = false;
    std::unordered_map<std::uint32_t, std::unordered_set<std::uint32_t>> joined;
    while (!done)
    {
        done = true;
        for (std::size_t i = 0; i < states.size(); i++)
        {
            if (states[i].current_nodes.size() == 0)
                continue;

            CurrentState &state = states[i];

            std::vector<Node *> next_children;
            for (Node *node : state.current_nodes)
            {
                if (node->node_type() == graphlib::kQueue && !(node->as<graphlib::QueueNode>()->is_buffering()))
                    continue;  // all queues except for buffering break fork-joins

                std::vector<Node *> children = remove_duplicates_and_outputs(graph->data_users(node), epoch_type);

                for (Node *child : children)
                {
                    if (child->node_type() == graphlib::kQueue && !(child->as<graphlib::QueueNode>()->is_buffering()))
                        continue;  // all queues except for buffering break fork-joins

                    std::uint32_t depth = state.node_depth[node] + 1;
                    if (depth > state.node_depth[child])
                        state.node_depth[child] = depth;

                    // std::cout << indent << " - branch: " << i << " -> child (depth " << depth << "): " <<
                    // child->name() << std::endl;

                    if ((state.parents.count(child) == 0) || (depth > state.parents[child].second))
                    {
                        // std::cout << indent << "     updating parent of  " << child->name() << " to " << node->name()
                        // << " due to depth " << depth << std::endl;
                        state.parents[child] = std::make_pair(node, depth);
                        next_children.push_back(child);
                    }

                    // if any of the children have been visited already in another branch, then we've got a join
                    for (std::size_t j = 0; j < states.size(); j++)
                    {
                        if (i == j)  // same branch
                            continue;

                        if (joined[i].count(j) > 0)  // branches already joined, everything after this will also join
                            continue;

                        if (states[j].parents.count(child) > 0)
                        {
                            // std::cout << indent << " found fork with branch " << j << std::endl;
                            NodeMap common_parents;
                            common_parents.insert(state.parents.begin(), state.parents.end());
                            common_parents.insert(states[j].parents.begin(), states[j].parents.end());
                            record_fork_join(
                                state.fork, child, states[j].parents.at(child).first, node, common_parents, fork_joins);
                            joined[i].insert(j);
                            joined[j].insert(i);
                        }
                    }
                }
            }
            state.current_nodes = next_children;
            indent += "  ";

            if (state.current_nodes.size() > 0)
                done = false;
        }
    }
}

std::vector<ForkJoin> find_fork_joins(Graph *graph)
{
    std::vector<ForkJoin> fork_joins;
    for (Node *node : graphlib::topological_sort(*graph))
    {
        // fork from input can be ignored, as queues can buffer the source at any rate
        if ((node->node_type() == graphlib::kInput) || (node->node_type() == graphlib::kQueue))
            continue;

        if (graph->data_users(node).size() > 1)
        {
            trace_fork(graph, node, fork_joins, node->get_epoch_type());
        }
    }
    return fork_joins;
}

struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &pair) const
    {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

// Recover src node if it's missing, from dest node and input_id
Node *recover_missing_src(graphlib::Graph *graph, Node *dest, std::uint32_t input_id)
{
    for (Edge e : graph->operand_data_edges(dest))
    {
        if (e.consumer_input_port_id == input_id)
            return graph->node_by_id(e.producer_node_id);
    }
    TT_THROW("Unable to find input with given input_id");
    return nullptr;
}

// Return some dest from the given src, if original dest_name node is now missing
Node *recover_missing_dest(graphlib::Graph *graph, Node *src, std::uint32_t fork_id)
{
    auto edges = graph->user_data_edges(src);
    if (fork_id >= edges.size())
    {
        fork_id = 0;  // fall-back, since graph has changed enough that this isn't even a fork any more
    }
    return graph->node_by_id(edges[fork_id].consumer_node_id);
}

void merge_tagged_nops_with_same_src(graphlib::Graph *graph, bool daisy_chain)
{
    // populate a map of src_op -> inserted_buffer_nops(mergeable == True)
    // Make this a map (ordered) to preserve deterministic order of transforms
    std::map<std::string, std::vector<Node *>> src_op_to_mergeable_nops;
    for (Node *node : graph->nodes())
    {
        if (graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
            op and op->op_name() == "nop" and op->has_tag("mergeable") and std::get<bool>(op->tag_value("mergeable")))
        {
            Node *src_op = graph->data_operands(node).at(0);
            src_op_to_mergeable_nops[src_op->name()].push_back(node);
        }
    }

    // check all mergeable nops have the same tms on their edges
    for (const auto &[src_op, nops] : src_op_to_mergeable_nops)
    {
        if (nops.size() > 1)
        {
            Edge edge_between_first_nop_and_src_op = graph->operand_data_edges(nops[0]).at(0);
            auto &tms = graph->get_edge_attributes(edge_between_first_nop_and_src_op)->get_tms();
            for (std::size_t i = 1; i < nops.size(); i++)
            {
                Edge edge_between_nop_and_src_op = graph->operand_data_edges(nops[i]).at(0);
                auto &tms2 = graph->get_edge_attributes(edge_between_nop_and_src_op)->get_tms();
                if (tms != tms2)
                {
                    log_error(
                        "User tried to add a buffering nop from src_op: {}, with (hoist_tms = True)"
                        "between two nodes with different tms. Try setting hoist_tms = False",
                        src_op);
                }
            }
        }
    }

    // check all mergeable nops have the same tms on their edges
    for (const auto &[src_op, nops] : src_op_to_mergeable_nops)
    {
        if (nops.size() > 1)
        {
            if (daisy_chain)
            {
                Node *src_node = graph->get_node_by_name(src_op);
                Node *current_nop = nops[0];

                for (std::size_t i = 1; i < nops.size(); i++)
                {
                    auto edge_to_reattach = graph->get_edges(src_node, nops[i]).at(0);
                    auto edge_attributes = graph->get_edge_attributes(edge_to_reattach);

                    auto new_edge = edge_to_reattach;
                    new_edge.producer_node_id = current_nop->id();

                    graph->remove_edge(edge_to_reattach);
                    graph->add_edge(new_edge, edge_attributes);
                    log_trace(LogGraphCompiler, "Trying to connect a new edge between producer={} and consumer={}", current_nop->name(), nops[i]->name());
                    current_nop = nops[i];
                }

                for (std::size_t i = 0; i < nops.size(); i++) {
                    // Make all merged daisy-chain nops unmergeable to allow for insertion of later daisy-chains
                    Node *current_nop = nops[i];
                    graphlib::OpNode *current_op = dynamic_cast<graphlib::OpNode *>(current_nop);
                    current_op->tag("mergeable", false);
                }
            }
            else
            {
                Node *first_nop = nops[0];
                for (std::size_t i = 1; i < nops.size(); i++)
                {
                    graphlib::replace_node(graph, nops[i], first_nop, true /* skip operands*/);
                }
            }
        }
    }
}

// if path contains fork node and join node of fork-join, then, it contains that fork-join
// this is used when determining if one fork-join is ancestor of another (one wrapping around another)
bool is_fork_join_on_path(const std::vector<Node *> &path, const ForkJoin *fj)
{
    bool contains_fork = false;
    bool contains_join = false;
    for (const Node *node : path)
    {
        if (node->id() == fj->first[0]->id())
        {
            contains_fork = true;
        }

        if (node->id() == fj->first.back()->id())
        {
            contains_join = true;
        }
    }
    return contains_fork && contains_join;
}

// returns true if first fork-join is ancestor of second fork-join
bool is_ancestor(const ForkJoin *fj_1, const ForkJoin *fj_2)
{
    // if path is only 2 nodes then it can't contain fork-join without having same fork and join as descendant.
    if ((fj_1->first.size() > 2 && is_fork_join_on_path(fj_1->first, fj_2)) ||
        (fj_1->second.size() > 2 && is_fork_join_on_path(fj_1->second, fj_2)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

// Graph of fork-joins. Each node is one fork-join, and each edge tells which fork-join is contained inside another
// edge goes from child to parent fork-join.
FJGraph::FJGraph(graphlib::Graph *graph)
{
    fork_joins = find_fork_joins(graph);
    if (fork_joins.size() == 0)
    {
        return;
    }
    // initializing adjacency vector with empty sets. Empty set in vector on index i means that i-th node doesn't
    // have any starting from it. later in this constructor we populate adjacency vector.
    this->adjacency_vector.resize(fork_joins.size());

    for (std::size_t i = 0; i < fork_joins.size(); i++)
    {
        this->fj_ids.push_back(i);
    }

    // create edges that connects fj to its ancestor fork-joins.
    // ancestor fork-join is the one that contains current fork-join one of its paths. we check this by asking if both
    // fork and join of current fj is contained on either of two paths on possible ancestor  fork-join. special case is
    // when two fork-joins share both fork and join node. Then, no one is ancestor so we don't have connection between
    // nodes.
    for (std::size_t i = 0; i < fork_joins.size() - 1; i++)
    {
        for (std::size_t j = i + 1; j < fork_joins.size(); j++)
        {
            const ForkJoin *fj_1 = &fork_joins[i];
            const ForkJoin *fj_2 = &fork_joins[j];
            if (fj_1->first[0] == fj_2->first[0] && fj_1->first.back() == fj_2->first.back())
            {
                // if fork and join from two fork-joins are the same, then this is the special where we can't say which
                // fork-join is ancestor and which descendant.
                continue;
            }
            if (is_ancestor(fj_1, fj_2))
            {
                // fj_1 is ancestor to fj2 so we need edge from fj_2 to fj_1
                this->add_edge(j, i);
            }
            else
            {
                if (is_ancestor(fj_2, fj_1))
                {
                    // fj_2 is ancestor to fj_1 so we need edge from fj_1 to fj_2
                    this->add_edge(i, j);
                }
            }
        }
    }

    this->topological_sort();
    this->create_parents_map();
}

// adds edge from src to dest
void FJGraph::add_edge(std::uint32_t src, std::uint32_t dest) { adjacency_vector[src].insert(dest); }

// Topologically sorts fork-join graph. In the end we get array that is sorted from most inner fork-join to most outer.
void FJGraph::topological_sort()
{
    // initialize cnt of visited nodes to 0
    std::uint32_t cnt_visited_nodes = 0;
    std::queue<std::uint32_t> nodes_to_visit;

    // initialize the vector of number of incoming edges to zeros.
    // for each node in graph we want to know how many incomming edges it has.
    // node that has 0 incomming edges should be the first one in topological order (it does not depend on any other node).
    std::vector<std::uint32_t> num_incomming_edges(fj_ids.size(), 0); 
    for (std::uint32_t src_fj_id : fj_ids)
    {
        for (auto dest_fj_id : adjacency_vector[src_fj_id])
        {
            num_incomming_edges[dest_fj_id]++;
        }
    }

    // add all verticies that have num_incomming_edges 0 to queue nodes_to_visit

    for (std::uint32_t fj_id : fj_ids)
    {
        if (num_incomming_edges[fj_id] == 0)
        {
            // add fj_id into the queue nodes_to_visit
            nodes_to_visit.push(fj_id);
        }
    }

    while (nodes_to_visit.size() != 0)
    {
        // take one element from the begining and pop it
        std::uint32_t current_fj_id = nodes_to_visit.front();
        nodes_to_visit.pop();
        cnt_visited_nodes++;
        // add that current fj to topologically sorted vector.
        topo_sort_fjs.push_back(&fork_joins[current_fj_id]);
        topo_sort_fj_indices.push_back(current_fj_id);

        // decrease num_incomming_edges for all neighbouring nodes of current_fj_id
        for (auto dest_fj_id : adjacency_vector[current_fj_id])
        {
            TT_ASSERT(num_incomming_edges[dest_fj_id] > 0, " It is expected that num_incomming_edges is greater than null, but it is not");
            num_incomming_edges[dest_fj_id]--;
            // if num_incomming_edges of dest_fj_id is reduced to 0, emplace dest_fj_id to nodes_to_visit
            if (num_incomming_edges[dest_fj_id] == 0)
            {
                nodes_to_visit.push(dest_fj_id);
            }
        }
    }
    TT_ASSERT(cnt_visited_nodes == fj_ids.size(), "Number of visited nodes is not equal to number of nodes -> topological sort is not possible for the given graph.");
}

// Fork join FJ_1 is parent to FJ_2 if FJ_1 is the most inner fj that contains FJ_2. We need map that tells us who is the parent for each fork-join.
// This method creates a map of node -> sorted_fj_ind, to track first ForkJoin index that contains node not including most inner fork
// join that contains node. this can be called parent fork-join. We need this structure to handle skipping already
// buffered fork-joins effectively. Most outer fork-join in graph won't have parent fork-join.
void FJGraph::create_parents_map()
{
    for (std::size_t i = 0; i < topo_sort_fj_indices.size(); i++)
    {
        std::size_t fj_child_id = topo_sort_fj_indices[i];

        // By default, fork join is a parent to itself.
        parent_fj_map[topo_sort_fjs[i]] = topo_sort_fjs[i];

        // for current fork-join parent will be on the right in the array of topo_sort_fj_indices
        // and will also have edge from fj_id to parent_id in adjacency_vector
        for (std::size_t j = i + 1; j < topo_sort_fj_indices.size(); j++)
        {
            std::size_t fj_parent_id = topo_sort_fj_indices[j];
            // j is parent to i if adjacent matrix contains edge from fj_child_id to fj_parent_id
            if (adjacency_vector[fj_child_id].find(fj_parent_id) != adjacency_vector[fj_child_id].end())
            {
                parent_fj_map[topo_sort_fjs[i]] = topo_sort_fjs[j];
                break;
            }
        }
    }
}

void FJGraph::add_elem_to_buffered_fjs(
    NodeId fork_id, FJBufferingInfo fj_buff_info)
{
    if (buffered_fjs.count(fork_id))
    {
        buffered_fjs[fork_id].push_back(fj_buff_info);
    }
    else
    {
        // if there is no key fork_id in the map yet.
        buffered_fjs[fork_id] =
            std::vector<FJBufferingInfo>{fj_buff_info};
    }
}

void FJGraph::erase_elem_from_buffered_fjs(NodeId fork_id, std::size_t idx)
{
    buffered_fjs[fork_id].erase(buffered_fjs[fork_id].begin() + idx);
}

/*
Checks if nodes with names this->src this->dest exist. If they do, returns pointers on them.
If one of them doesn't exist in the graph we try to find which node is currently connected to the specified port
on existing node. Therefore, if one of the src, dest exists, we can infer the other one's replacement from PortIds
and still return them.
On the other hand if both src and dest don't exist anymore we return false
*/
std::pair<Node *, Node *> InsertionInstruction::is_instruction_still_valid(graphlib::Graph *graph)
{
    // In some cases, an op will not exist any more -- the known case is when a balancer exception has caused
    // modifications to the graph to be made that might not be needed once NOPs are inserted.
    Node *src, *dest;
    src = graph->get_node_by_name(this->src, false);
    dest = graph->get_node_by_name(this->dest, false);
    if (src == nullptr)  // graph doesn't have node with name this->src
    {
        if (this->user_defined)
        {
            log_error("User constructed Nop Instruction constructed with invalid src-nop: {}", this->src);
        }

        if (dest == nullptr)
        {
            log_debug(
                LogGraphCompiler,
                "Both {} and {} can't be found, re-lowered graph is different. Skipping nop insertion for the pair",
                this->src,
                this->dest);
            return std::make_pair(src, dest);
        }
        TT_ASSERT(this->input_id.has_value(), "Nop Instruction missing input_id attribute populated.");
        src = recover_missing_src(graph, dest, this->input_id.value());
    }

    else if (dest == nullptr)
    {
        if (this->user_defined)
        {
            log_error("User constructed Nop Instruction constructed with invalid dest-nop: {}", this->dest);
        }
        TT_ASSERT(this->input_id.has_value(), "Nop Instruction missing fork_id attribute populated.");
        dest = recover_missing_dest(graph, src, this->fork_id.value());
    }

    this->src = src->name();
    this->dest = dest->name();
    return std::make_pair(src, dest);
}

/*
Inserts nops between src and dest nodes that are specified in this.
*/
void NopInsertionInstruction::insert(graphlib::Graph *graph)
{
    // some reasonable max after which we'll likely change epochs enough to not overdo it
    std::uint32_t max_nops = (std::uint32_t)env_as<int>("PYBUDA_MAX_FORK_NOPS", 2);

    // If this is an user-defined insert instruction (override), don't limit nop count.
    if (this->user_defined)
    {
        max_nops = this->nop_count;
    }

    Node *src, *dest;
    std::tie(src, dest) = this->is_instruction_still_valid(graph);
    // if instruction isn't valid anymore (src or dest is nullptr after calling is_instruction_still_valid) we skip
    // adding nop
    if (src == nullptr || dest == nullptr)
    {
        return;
    }

    // when multiple buffering nops are needed, this string becomes too long and breaks yaml spec when dumped to netlist
    // so if dest name contains buffer_N_src, increment index, and remove buffer_N_src from dest name
    auto op_name = [](Node *src, Node *dest,  graphlib::Graph* graph)
    {
        std::uint32_t buffer_index = 0;
        auto dest_name = dest->name();
        if (dest->name().find("buffer_") != std::string::npos and dest->name().find(src->name()) != std::string::npos)
        {
            buffer_index = std::stoi(dest_name.substr(dest_name.find("buffer_") + 7, dest_name.find(src->name()) - dest_name.find("buffer_") - 7));
            std::string remove = "buffer_" + std::to_string(buffer_index) + "_" + src->name() + "_";
            dest_name.erase(dest_name.find(remove), remove.length());
        }
        std::string op_name;
        do
        {
            op_name = "buffer_" + std::to_string(buffer_index++) + "_" + src->name() + "_" + dest_name;
        } while (graph->has_node_with_name(op_name));
        return op_name;
    };

    if (src->node_type() == graphlib::NodeType::kQueue || dest->node_type() == graphlib::NodeType::kQueue)
    {
        return;  // don't need nop if src or dest are queues
    }
    Node *original_dest = dest;
    // insert min(nop_count,max_nops) nops between src and dest
    for (std::size_t nop_index = 0; nop_index < std::min(this->nop_count, max_nops); nop_index++)
    {
        graphlib::BudaOpNode *buffer_nop = nullptr;

        auto edges = graph->get_edges(src, dest);
        for (graphlib::Edge e : edges)
        {
            if (e.edge_type != graphlib::EdgeType::kData)
            {
                continue;
            }

            if (buffer_nop == nullptr)
            {
                // create new nop BudaOpNode
                buffer_nop = graph->add_node(
                    graphlib::create_node<graphlib::BudaOpNode>(op_name(src, original_dest, graph), "nop"),
                    graph->get_subgraph_id_for_node(src->id()));
                buffer_nop->set_shape(src->shape());
                buffer_nop->set_buffering_op(this->is_fj_buffering);
                buffer_nop->tag("mergeable", this->mergeable);

                graphlib::BudaOpNode* src_op = dynamic_cast<graphlib::BudaOpNode*>(src);
                if (src_op != nullptr and src_op->op_name() != "dequantization")
                {
                    buffer_nop->set_accumulate_df(src_op->accumulate_df());
                    buffer_nop->set_intermediate_df(src_op->intermediate_df());
                    buffer_nop->set_math_fidelity(src_op->math_fidelity());
                }
            }

            // insert new node on edge
            auto [edge0, edge1] = graphlib::insert_node_on_edge(graph, e, buffer_nop, false /* inherit_consumer_attrs */);
            log_trace(
                LogGraphCompiler,
                "Inserted buffer nop node {} between {} and {}",
                buffer_nop->name(),
                src->name(),
                dest->name());

            // Move TMs to edge1
            auto &tms = graph->get_edge_attributes(edge0)->get_tms();
            if (not this->hoist_tms)
            {
                // not hoisting tms, move them to edge1
                graph->get_edge_attributes(edge1)->set_tms(tms);
                graph->get_edge_attributes(edge0)->set_tms(std::vector<graphlib::OpType>{});
            }
            dest = buffer_nop;
        }
    }
    // sometimes we want to connect one src to pultiple consumers but adding one nop between them.
    // This is done buy adding separate nops between every pair of src - dest and then calling this
    // method merge_tagged_nops_with_same_src to merge tagged nops with same source.
    // Tag is "mergeable" and if it is true than nop can be merged
    if (this->request_merge)
    {
        merge_tagged_nops_with_same_src(graph, this->daisy_chain);
    }
}

/*
Inserts buffering queue node between src and dest nodes that are specified in this.
*/
void QueueInsertionInstruction::insert(graphlib::Graph *graph)
{
    Node *src, *dest;
    std::tie(src, dest) = this->is_instruction_still_valid(graph);
    // if instruction isn't valid anymore (src or dest is nullptr after calling is_instruction_still_valid) we skip
    // adding nop
    if (src == nullptr || dest == nullptr)
    {
        return;
    }
    if (src->node_type() == tt::graphlib::kQueue && src->as<graphlib::QueueNode>()->is_buffering())
    {
        // if src is BufferingQueueNode, then just ensure that we have enough buffering by choosing maximum
        // number of entries
        if (src->as<graphlib::QueueNode>()->get_num_entries() < this->num_entries)
            src->as<graphlib::QueueNode>()->set_num_entries(this->num_entries);
        return;
    }
    if (dest->node_type() == tt::graphlib::kQueue && dest->as<graphlib::QueueNode>()->is_buffering())
    {
        // if dest is BufferingQueueNode, then just ensure that we have enough buffering by choosing maximum
        // number of entries
        if (dest->as<graphlib::QueueNode>()->get_num_entries() < this->num_entries)
            dest->as<graphlib::QueueNode>()->set_num_entries(this->num_entries);
        return;
    }
    if (src->node_type() != graphlib::NodeType::kBudaOp || dest->node_type() != graphlib::NodeType::kBudaOp)
    {
        // can put BufferingQueueNode between nodes only if they both are kBudaOp
        return;
    }

    // there has to be an edge between src and dest in order to add queue between them.
    bool has_edge_between_src_dest = false;
    for (Edge e : graph->user_data_edges(src))
    {
        if (e.consumer_node_id == dest->id())
        {
            has_edge_between_src_dest = true;
        }
    }

    if (has_edge_between_src_dest == false)
        return;

    // if there is less than 2 user data edges from src than certainly there is no fork anymore.
    if (graph->user_data_edges(src).size() < 2)
        return;

    // currently we skip adding queue between recompute nodes because fork join paths can reconnect after adding the
    // queue
    if (is_recompute(graph, src) && is_recompute(graph, dest))
        return;

    auto [edge0, queue_node, edge1] = insert_serialized_dram_queue_between_ops(
        graph, src->name(), dest->name(), this->input_id.value(), this->num_entries);

    log_trace(
        LogGraphCompiler,
        "Inserted buffer queue node {} between {} and {}",
        queue_node->name(),
        src->name(),
        dest->name());
}

// Helper method to retrieve OpModel from either the inline or post-placer OpModelMap.
//
balancer::OpModel &get_op_model(balancer::OpModelMap *op_models_post_placer, balancer::OpModels *op_models, Node *node)
{
    return op_models_post_placer == nullptr ? op_models->at(node) : op_models_post_placer->at(node->name());
}

// Calculates next output multiplier from input_multiplier. Tracks how tensor is expanded/contracted from input to
// output of an op.
//  next output multiplier increases when tensor volume decreases form input to output.
float get_output_multiplier(
    Node *node, float input_multiplier, const balancer::OpModel &op_model, tt::graphlib::PortId input_port_id)
{
    tt::balancer::TensorShape in_shape = op_model.op_shape.inputs[input_port_id];
    tt::balancer::TensorShape out_shape = op_model.op_shape.outputs[0];
    // output shape is multiplied with t streaming so its rt and ct represent whole tensor
    // rather than slices, so we have to manually divide it
    float out_shape_volume = out_shape.ct * out_shape.rt / (float)(op_model.t_stream_factor.t());
    float input_shape_volume = in_shape.ct * in_shape.rt;
    // only for sparse matmul in_shape.rt and in_shape.ct calculate in t streaming. That means that
    // in_shape.ct * in_shape.rt represent volume of the whole tensor rather than only one slice (like
    // in the rest of the ops). we are interested in size of the slice that is calculated in one entry,
    // so in this case we have to divide with t stream factor
    if (node->as<graphlib::OpNode>()->is_sparse_matmul())
    {
        input_shape_volume /= (float)(op_model.t_stream_factor.t());
    }
    return input_multiplier * input_shape_volume / out_shape_volume ;
}

// Calculates stack factor between node and consumer node based on op_models, more specifically op shape z dimensions.
float get_stack_factor(
    Graph *graph,
    balancer::OpModelMap *op_models_post_placer,
    balancer::OpModels *op_models,
    Node *node,
    Node *consumer_node)
{
    // stack factor is calculated as ratio of output shape z dimension of producer, and input shape z dimension of
    // consumer.
    float stack_factor = 1;
    // if consumer node is not nullptr and node type is BudaOpNode
    if (consumer_node != nullptr && consumer_node->node_type() == graphlib::NodeType::kBudaOp)
    {
        const balancer::OpModel &op_model = get_op_model(op_models_post_placer, op_models, node);
        const balancer::OpModel &consumer_op_model = get_op_model(op_models_post_placer, op_models, consumer_node);
        std::function<bool(Edge)> edge_filter = [consumer_node](Edge edge)
        { return edge.consumer_node_id == consumer_node->id(); };

        std::vector<Edge> user_edges = graph->user_data_edges(node, edge_filter);
        // If we have multiple edges (user_edges.size() > 1) from node to consumer_node, we find the one with the
        // smallest z dim on input op shape, because that will produce highest stack_factor - we want to buffer the
        // worst case scenario.
        TT_ASSERT(user_edges.size() >= 1, "Expected to have edge between node and consumer_node");
        int min_consumer_z_dim = INT_MAX;
        int consumer_z_dim = INT_MAX;
        for (Edge e : user_edges)
        {
            consumer_z_dim = consumer_op_model.op_shape.inputs[e.consumer_input_port_id].z;
            if (consumer_z_dim < min_consumer_z_dim)
            {
                min_consumer_z_dim = consumer_z_dim;
            }
        }

        TT_ASSERT(min_consumer_z_dim > 0, "input z dim of consumer op model can't be less than 1");
        stack_factor = op_model.get_out_shape().z / (float)(min_consumer_z_dim);
    }
    return stack_factor;
}

// compares two fork-joins node by node
bool is_same_fj(const ForkJoin& fj1, const ForkJoin& fj2)
{
    if (fj1.first.size() != fj2.first.size() || fj1.second.size() != fj2.second.size())
    {
        return false;
    }

    // compare first path
    for (std::size_t i = 0; i < fj1.first.size(); i++)
    {
        if (fj1.first[i]->id() != fj2.first[i]->id())
        {
            return false;
        }
    }

    // compare second path
    for (std::size_t i = 0; i < fj1.second.size(); i++)
    {
        if (fj1.second[i]->id() != fj2.second[i]->id())
        {
            return false;
        }
    }

    return true;
}

// gets current node (fork) and tries to find if there is fork-join starting at that node (fork) and finishing at join
// that belongs to current fork-join (fj) Also, fork-join which we are trying to find has to be already buffered
// (contained in map fj_graph.buffered_fjs).
FJBufferingInfo FJGraph::find_sub_fork_join_from_node(
    const ForkJoin &fj, const std::vector<Node *> &path, Node *fork)
{
    if (buffered_fjs.count(fork->id()))
    {
        // there is some already buffered fork-join that shares the fork with current FJ which one of the paths is path
        for (auto fj_buff_info : buffered_fjs.at(fork->id()))
        {
            const ForkJoin *buff_fj = fj_buff_info.fj;
            // is parent of buffered_fj the same as fj
            if (is_same_fj(*parent_fj_map.at(buff_fj), fj))
            {
                // we found the fj that is contained in current fork-join, and previously buffered.
                // we have to check if this buffered fork-join (buff_fj) belongs to right path of fj
                if (is_fork_join_on_path(path, buff_fj))
                {
                    // we will use its required and available buffering.
                    return fj_buff_info;
                    break;
                }
            }
        }
    }
    return FJBufferingInfo(nullptr, 0, 0, nullptr);
}


void FJGraph::update_buffered_fj_map(const ForkJoin& fj, FJBufferingInfo fj_buff_info)
{
    // ForkJoinId fj_key = std::make_pair(fj.second[0]->id(),fj.second.back()->id());
    NodeId fork_id = fj.second[0]->id();
    if (buffered_fjs.count(fork_id) == 0)
    {
        // if there is no already buffered fork-joins with same fork, just add current fj to buffered_fjs
        this->add_elem_to_buffered_fjs(fork_id, fj_buff_info);
    }
    else
    {
        // there are buffered fork-joins with same fork as current fj. We want to delete buffered fork-joins that have same fork
        // as current fj if that fork-join is their parent.
        std::vector<FJBufferingInfo> already_buff_fj_info =
            buffered_fjs.at(fork_id);
        std::vector<std::size_t> indices_to_delete;
        for (std::size_t i = 0; i < already_buff_fj_info.size(); i++)
        {
            FJBufferingInfo value = already_buff_fj_info[i];
            const ForkJoin *buff_fj = value.fj;

            // check if buff_fj has parent fork-join in map parent_fj_map
            if (parent_fj_map.count(buff_fj))
            {
                const ForkJoin *parent_fj = parent_fj_map.at(buff_fj);
                if (is_same_fj(*parent_fj, fj))
                {
                    indices_to_delete.push_back(i);
                    continue;
                }
            }
            else
            {
                TT_ASSERT(
                    parent_fj_map.count(buff_fj),
                    "It is expected that each join in graph has parebt-fork-join. If a fork-join is not contained in "
                    "other fj, then that fork-join is parent fork-join to itself.");
            }
        }

        this->add_elem_to_buffered_fjs(fork_id, fj_buff_info);
        // delete all fork-joins that share fork with current fj and are contained in current fj, because we only need
        // info on most outer fj that is buffered. It is important that indices_to_delete is sorted in descending order.
        // Then, each index that we delete won't influence index values of next elements we need to delete.
        sort(indices_to_delete.begin(), indices_to_delete.end(), [](int a, int b) { return a > b; });
        for (std::size_t idx : indices_to_delete)
        {
            this->erase_elem_from_buffered_fjs(fork_id, idx);
        }
    }
}

// if available is set, then we're looking for available buffering in the path... if not, then
// we're looking for required buffering
std::tuple<std::uint32_t, bool, int> get_buffering(
    Graph *graph,
    balancer::OpModelMap *op_models_post_placer,
    balancer::OpModels *op_models,
    const std::vector<Node *> &path,
    const ForkJoin &fj,
    bool available,
    FJGraph& fj_graph)
{
    float current_output_multiplier = 1.0;  // keep track of expansions and reductions of fork outputs
    Node *prev_node = path[0];
    int sum_buff_queue_num_entries = 0;

    bool dump_debug_info = false;
    std::stringstream debug_info;

    if (env_as<bool>("PYBUDA_FORK_JOIN_DEBUG_INFO"))
    {
        std::string fork_name = env_as<std::string>("PYBUDA_FORK_JOIN_DEBUG_FORK_NAME");
        std::string join_name = env_as<std::string>("PYBUDA_FORK_JOIN_DEBUG_JOIN_NAME");

        // If user has provided fork/join node names, apply them to filter unneeded debug info.
        dump_debug_info = (fork_name.empty() || fork_name == path[0]->name());
        dump_debug_info &= (join_name.empty() || join_name == path.back()->name());
    }

    std::uint32_t total_buffering = 0;
    bool has_queue_on_path = false;

    auto [join, req, avail, sub_fj] = fj_graph.find_sub_fork_join_from_node(fj, path, path[0]);

    if (join != nullptr)
    {
        // we found sub fork-join from node path[0]
        total_buffering += available ? avail : req;
    }

    for (std::uint32_t path_index = 1; path_index < path.size(); path_index++)
    {
        Node *node = path[path_index];
        // if current node is join that means that we already included its input buffering in total_buffering
        // calculation independent of wether it is available or required buffering.
        bool curr_node_is_join = (join == node && node != nullptr);
        Node *consumer_node = nullptr;

        std::stringstream node_debug_info;

        if (path_index < path.size() - 1)
        {
            consumer_node = path[path_index + 1];

            if (consumer_node->node_type() != graphlib::NodeType::kBudaOp)
            {
                // currently if consumer op is not Buda Op we leave consummer_node to nullptr
                // that causes next_stack_factor to be 1, which is what we want.
                consumer_node = nullptr;
            }
        }

        if (node->node_type() == graphlib::kQueue)
        {
            TT_ASSERT(
                node->as<graphlib::QueueNode>()->is_buffering(),
                "Buffering queues are the only type of queues that are tolerated in fork-joins");
            has_queue_on_path = true;
            sum_buff_queue_num_entries += node->as<graphlib::QueueNode>()->get_num_entries();
            continue;
        }

        const balancer::OpModel &op_model = get_op_model(op_models_post_placer, op_models, node);

        if (dump_debug_info)
        {
            node_debug_info << "node name: " << node->name() << std::endl;
            node_debug_info << "op type: " << node->get_type() << std::endl;
        }

        std::uint32_t input_buffering = 0;  // number of tiles
        float next_output_multiplier = 1.0;
        bool is_join = (path_index == path.size() - 1);
        // if node is buffering op, it should not influence required buffering of the path, since buffering op
        // is just used for storing the data
        if (node->as<graphlib::BudaOpNode>()->is_buffering_op() && !available)
        {
            prev_node = node;
            continue;
        }

        for (Edge e : graph->user_data_edges(prev_node))
        {
            if (e.consumer_node_id != node->id())
                continue;

            auto tms = graph->get_edge_attributes(e)->get_tms();

            float input_multiplier = current_output_multiplier;
            int broadcast_factor = 0;
            for (auto tm : tms)
            {
                // std::cout << "tm: " << tm.op << std::endl;
                // if ((tm.op == "vstack") || (tm.op == "hstack"))
                //    input_multiplier *= (float)std::get<int>(tm.attr[0]);
                // else if ((tm.op == "vslice") || (tm.op == "hslice"))
                //    input_multiplier /= (float)std::get<int>(tm.attr[0]);
                if (tm.op == "broadcast")
                {
                    broadcast_factor = std::get<int>(tm.attr[1]);
                    input_multiplier /= (float)broadcast_factor;
                }
            }

            tt::balancer::TensorShape in_shape = op_model.op_shape.inputs[e.consumer_input_port_id];
            std::uint32_t in_tiles = 0;

            if (available)
            {
                // It's simply input and output buffers together, unless it's the join op, for which only input buffer
                // counts
                in_tiles = op_model.input_buffers.at(e.consumer_input_port_id).l1_size_tiles;
                in_tiles *= op_model.grid_shape.volume();
            }
            else
            {
                // Figure out how much is required for full fwd progress
                if (node->as<graphlib::BudaOpNode>()->op_type().op == "fused_op")
                {
                    // TODO: we can't tell how much of the input fused op needs to produce output, without analyzing and
                    // tracing the fused op For now, we'll estimate by just using the input size.

                    // Fused op can produce output from just one slice (in_shape.ct * in_shape.rt).
                    // However that is not enough if the next op requires more slices to start calculating
                    // (has some stack on next edge)
                    // next_stack_factor handles this.
                    float next_stack_factor =
                        get_stack_factor(graph, op_models_post_placer, op_models, node, consumer_node);
                    // int next_stack_factor = op_model.output_buffers[0].buffer_factor;
                    // If stack_factor is < 1 that means that we have slice on next edge, which doesn't influence
                    // current op required buffering estimation.
                    next_stack_factor = std::max((float)(1), next_stack_factor);

                    // Fused op can produce output from just one slice (in_shape.ct * in_shape.rt).
                    // However that is not enough if the previous op produces output with smaller z dimension. That
                    // means even if current op can move forward only with one slice, it will have to wait the previous
                    // op to form complete output to get one slice. Therefore required buffering doesn't depend only on
                    // slice size but on slice factor on previous edge. We have to take into consideration
                    // prev_slice_factor on previous edge.
                    const balancer::OpModel &previous_op_model =
                        get_op_model(op_models_post_placer, op_models, prev_node);
                    float prev_slice_factor = 1;
                    if (op_model.op_shape.inputs[e.consumer_input_port_id].volume_in_tiles() == previous_op_model.op_shape.outputs[0].volume_in_tiles())
                    {
                        // if volumes are the same, then we can compare z dimensions to infer if it was slicing between prev_node and node.
                        // if volume has changed from output of prev_node to input of node, we had some broadcast
                        prev_slice_factor = op_model.op_shape.inputs[e.consumer_input_port_id].z /
                            (float)(previous_op_model.get_out_shape().z);
                    }

                    // If prev_slice_factor is < 1 that means that we have stack onprevious edge, which doesn't
                    // influence current op required buffering estimation.
                    prev_slice_factor = std::max((float)(1), prev_slice_factor);
                    in_tiles = ceil(in_shape.ct * in_shape.rt * std::max(prev_slice_factor, next_stack_factor) * 2);
                }
                else if (is_join)
                {
                    // We just need to fill the input buffer to make progress, not actually produce a full output
                    in_tiles = op_model.input_buffers.at(e.consumer_input_port_id).block_shape.volume_no_t() * 2; // 2 because of double buffering
                    in_tiles *= op_model.grid_shape.volume();
                }
                else if (node->as<graphlib::BudaOpNode>()->op_type().op == "matmul")
                {
                    // Matmul can produce output from just one slice (in_shape.ct * in_shape.rt).
                    // However that is not enough if the next op requires more slices to start calculating
                    // (has some stack on next edge)
                    // next_stack_factor handles this.
                    float next_stack_factor =
                        get_stack_factor(graph, op_models_post_placer, op_models, node, consumer_node);
                    // int next_stack_factor = op_model.output_buffers[0].buffer_factor;
                    // If stack_factor is < 1 that means that we have slice on next edge, which doesn't influence
                    // current op required buffering estimation.
                    next_stack_factor = std::max((float)(1), next_stack_factor);

                    // Matmul can produce output from just one slice (in_shape.ct * in_shape.rt).
                    // However that is not enough if the previous op produces output with smaller z dimension. That
                    // means even if current op can move forward only with one slice, it will have to wait the previous
                    // op to form complete output to get one slice. Therefore required buffering doesn't depend only on
                    // slice size but on slice factor on previous edge. We have to take into consideration
                    // prev_slice_factor on previous edge.
                    const balancer::OpModel &previous_op_model =
                        get_op_model(op_models_post_placer, op_models, prev_node);
                    float prev_slice_factor = 1;
                    if (op_model.op_shape.inputs[e.consumer_input_port_id].volume_in_tiles() == previous_op_model.op_shape.outputs[0].volume_in_tiles())
                    {
                        // if volumes are the same, then we can compare z dimensions to infer if it was slicing between prev_node and node.
                        // if volume has changed from output of prev_node to input of node, we had some broadcast
                        prev_slice_factor = op_model.op_shape.inputs[e.consumer_input_port_id].z /
                            (float)(previous_op_model.get_out_shape().z);
                    }

                    // If prev_slice_factor is < 1 that means that we have stack onprevious edge, which doesn't
                    // influence current op required buffering estimation.
                    prev_slice_factor = std::max((float)(1), prev_slice_factor);
                    in_tiles = ceil(in_shape.ct * in_shape.rt * std::max(prev_slice_factor, next_stack_factor) * 2);
                    // 2 is beacuse of double buffering.
                    // only for sparse matmul in_shape.rt and in_shape.ct calculate in t streaming. That means that
                    // in_shape.ct * in_shape.rt represent volume of the whole tensor rather than only one slice (like
                    // in the rest of the ops). we are interested in size of the slice that is calculated in one entry,
                    // so in this case we have to divide with t stream factor
                    if (node->as<graphlib::OpNode>()->is_sparse_matmul())
                    {
                        in_tiles /= op_model.t_stream_factor.t();
                    }
                }
                else
                {
                    in_tiles = op_model.output_buffers.at(0).l1_size_tiles;
                    in_tiles *= op_model.grid_shape.volume();
                }
            }

            std::uint32_t in_req = in_tiles * input_multiplier;

            if (in_req > input_buffering)
            {
                input_buffering = in_req;  // largest edge wins if there's more than one edge
                next_output_multiplier =
                    get_output_multiplier(node, input_multiplier, op_model, e.consumer_input_port_id);
            }

            if (dump_debug_info)
            {
                node_debug_info << '\t' << "input port id: " << e.consumer_input_port_id << std::endl;
                if (broadcast_factor)
                    node_debug_info << '\t' << "input edge has broadcast of factor: " << broadcast_factor << std::endl;
                node_debug_info << '\t' << "op grid shape: " << op_model.grid_shape << std::endl;
                node_debug_info << '\t' << "input shape: " << in_shape << std::endl;
                node_debug_info << '\t' << "input buffer block shape: " << op_model.input_buffers.at(e.consumer_input_port_id).block_shape << std::endl;
                node_debug_info << '\t' << "l1 size tiles: " << op_model.input_buffers.at(e.consumer_input_port_id).l1_size_tiles << std::endl;
                node_debug_info << '\t' << "in tiles: " << in_tiles << std::endl;
                node_debug_info << '\t' << "input multiplier: " << input_multiplier << std::endl;
                node_debug_info << '\t' << "total in: " << in_req << std::endl;
                node_debug_info << '\t' << "next output multiplier: " << get_output_multiplier(node, input_multiplier, op_model, e.consumer_input_port_id) << std::endl;

                node_debug_info << std::endl;
            }
        }

        current_output_multiplier = next_output_multiplier;
        std::uint32_t output_buffering =
            (path_index < path.size() - 1)
                ? op_model.output_buffers.at(0).l1_size_tiles * op_model.grid_shape.volume() * current_output_multiplier
                : 0;  // output buffer on the last one doesn't count

        if (join == nullptr)
        {
            if (available)
            {
                total_buffering += input_buffering + output_buffering;
            }
            else
            {
                // Somehwat arbitrary - a better algorithm is needed here
                // std::string op_type = node->as<graphlib::BudaOpNode>()->op_type().op;
                // if ((op_type == "matmul") || (op_type == "sparse_matmul") || (op_type == "exp"))
                total_buffering += input_buffering;  // input buffering is the requirement
            }
        }
        if (join == nullptr || join == node)
        {
            FJBufferingInfo fj_buff_info = fj_graph.find_sub_fork_join_from_node(fj, path, node);
            join = fj_buff_info.join;
            req = fj_buff_info.req;
            avail = fj_buff_info.avail;
            if (join != nullptr)
            {
                // we found sub fork-join from node node
                // we uptade total_buffering with avail or req depending on bool available.
                // We add scalling factor current_output_multiplier that takes into consideration tensor expansion and
                // contraction from the begining of fork-join for which we calculate total buffering.
                total_buffering += available ? avail * current_output_multiplier : req * current_output_multiplier;
            }
            if (curr_node_is_join && available)
            {
                // we are at the end of fj. add output buffer of join node to available buffering
                total_buffering += output_buffering;
            }
        }

        if (dump_debug_info)
        {
            if (available)
            {
                node_debug_info << "output_buffering: " << output_buffering << std::endl;
                node_debug_info << "total_buffering (for node): " << input_buffering + output_buffering << std::endl;
            }
            else
            {
                node_debug_info << "total_buffering (for node): " << input_buffering << std::endl;
            }

            node_debug_info << "--------------------------------------------------->" << std::endl;
            debug_info << node_debug_info.str();
        }

        prev_node = node;
    }

    if (dump_debug_info)
    {
        debug_info << "Total " << (available ? "available" : "required") << " buffering: " << total_buffering << std::endl;
        log_debug(LogGraphCompiler, "Calculating {} buffering between nodes {} and {}\n\n{}", available ? "available" : "required", path[0]->name(), path.back()->name(), debug_info.str());
    }

    return std::make_tuple(total_buffering, has_queue_on_path, sum_buff_queue_num_entries);
}

std::tuple<std::uint32_t, bool, int> get_available_buffering(
    Graph *graph,
    balancer::OpModelMap *op_models_post_placer,
    balancer::OpModels *op_models,
    const std::vector<Node *> &path,
    const ForkJoin &fj,
    FJGraph& fj_graph)
{
    return get_buffering(graph, op_models_post_placer, op_models, path, fj, true, fj_graph);
}

std::tuple<std::uint32_t, bool, int> get_buffering_requirement(
    Graph *graph,
    balancer::OpModelMap *op_models_post_placer,
    balancer::OpModels *op_models,
    const std::vector<Node *> &path,
    const ForkJoin &fj,
    FJGraph& fj_graph)
{
    return get_buffering(graph, op_models_post_placer, op_models, path, fj, false, fj_graph);
}

/*
Calculates how much dram memory buffering queue nodes consume in bytes. If this number exceeds some threshold
we can stop producing queues and continue only using nops for fork-join buffering
*/
int buffering_queues_mem_consumption(
    const tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        &instructions)
{
    int buff_queue_memory_consumption = 0;
    for (auto instruction : instructions)
    {
        if (instruction.second->instr_type == InstructionType::QueueInstruction)
        {
            QueueInsertionInstruction *que_instr = static_cast<QueueInsertionInstruction *>(instruction.second.get());
            buff_queue_memory_consumption += que_instr->queue_size;
        }
    }
    return buff_queue_memory_consumption;
}
// Inserts new queue instruction to map of instructions. 
void insert_queue_ins_to_instructions(
    tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        &instructions,
    InsInstructionUniqueId key,
    std::shared_ptr<InsertionInstruction> new_ins)
{
    TT_ASSERT(
        new_ins.get()->instr_type == InstructionType::QueueInstruction,
        "Instruction has to be of type InstructionType::QueueInstruction");
    if (instructions.count(key) > 0)
    {
        if (instructions[key].get()->instr_type == InstructionType::NopInstruction)
        {
            // if instructions contains element with key equal to key and current instruction is NopInstruction,
            // we replace it with queue instruction. This is because if we add queue on the path, we don't need nops on that path.
            instructions[key] = new_ins;
        }
        else if (instructions[key].get()->instr_type == InstructionType::QueueInstruction)
        {
            // if instructions contains element with key equal to key and current instruction is QueueInstruction,
            // we update num entries of the queue to the maximum of the num_entries of two instructions.
            QueueInsertionInstruction *instr = static_cast<QueueInsertionInstruction *>(instructions[key].get());
            QueueInsertionInstruction *new_queue_ins = static_cast<QueueInsertionInstruction *>(new_ins.get());
            if (instr->num_entries < new_queue_ins->num_entries)
            {
                instr->set_num_entries(new_queue_ins->num_entries);
            }
        }
        else
        {
            log_error("Unsupported instruction type");
        }
    }
    else
    {
        instructions[key] = new_ins;
    }
}

uint32_t expand_output_buffer(
    const Graph *graph,
    const Node *node,
    balancer::OpModel& op_model,
    float scale_usable_l1_size,
    uint32_t usable_l1_size,
    balancer::OpModelMap *op_models_post_placer,
    balancer::OpModels *op_models
)
{
    if (scale_usable_l1_size * usable_l1_size <= op_model.get_l1_memory_usage())
    {
        return 0;
    }

    uint32_t added_tiles = 0;
    balancer::BufferModel& output_buffer = op_model.output_buffers[0];
    const uint32_t tile_size_bytes = balancer::tile_size_bytes(output_buffer.data_format);
    const uint32_t available_space_tiles = (scale_usable_l1_size * usable_l1_size - op_model.get_l1_memory_usage()) / tile_size_bytes;
    const uint32_t tiles_per_mb = op_model.block_shape().volume_no_t();
    const uint32_t t_dim = op_model.block_shape().t;
    const uint32_t initial_mb = output_buffer.buffer_factor;

    // We want to expand output buffer to fit at most t macro blocks.
    const uint32_t mb_limit = std::min((uint32_t)(available_space_tiles + output_buffer.l1_size_tiles) / tiles_per_mb / 2, t_dim);

    if (mb_limit <= 1)
    {
        // No space to extend the output buffer, since we can't fit more than 1 macro block (double buffered).
        return 0;
    }

    const balancer::FactorizedInt t_dim_factors = balancer::FactorizedInt(t_dim);

    // Backend constraint is that the size of the output buffer in macro blocks must be divisible by t (or vice versa).
    // Since we will buffer at most t macro blocks (whole output),
    // take nearest factor of t less than or equal to the actual limit.
    uint32_t size_in_mb = t_dim_factors.get_nearest_factor_le(mb_limit);

    if (size_in_mb * 2 == initial_mb)
    {
        // No change.
        return 0;
    }

    // Adjust for some, but not all constrains in budabackend/src/net2pipe/src/tile_maps.cpp::check_phased_stack.
    // One of them is that the stack_factor must be divisible by the product of it's corresponding grid dimension and buf_size_mb / 2 (size_in_mb).
    // There are a lot more and they are not implemented here.
    for (const Edge& edge : graph->user_data_edges(node))
    {
        Node* consumer_node = graph->node_by_id(edge.consumer_node_id);
        if (consumer_node->node_type() != graphlib::NodeType::kBudaOp)
        {
            continue;
        }

        const balancer::OpModel& consumer_op_model = get_op_model(op_models_post_placer, op_models, consumer_node);

        const uint32_t t_stream_r = op_model.t_stream_factor.r;
        const uint32_t consumer_t_stream_r = consumer_op_model.t_stream_factor.r;

        // Need to vstack.
        if (t_stream_r > consumer_t_stream_r)
        {
            if (t_stream_r % consumer_t_stream_r != 0)
            {
                // T stream factor must be a divisible by the consumer t stream factor.
                // Cannot calculate constraints for this case.
                continue;
            }
            const uint32_t vstack_factor = t_stream_r / consumer_t_stream_r;
            const uint32_t consumer_grid_r = consumer_op_model.grid_shape.r;

            if (vstack_factor > size_in_mb)
            {
                size_in_mb = (t_dim_factors & balancer::FactorizedInt(vstack_factor)).get_nearest_factor_le(mb_limit);
            }

            if (vstack_factor > size_in_mb and vstack_factor > consumer_grid_r)
            {
                if (vstack_factor % consumer_grid_r != 0)
                {
                    // VStack factor must be divisible by the grid r of the consumer op.
                    continue;
                }
                const uint32_t vstack_factor_per_core = vstack_factor / consumer_grid_r;

                if (vstack_factor_per_core % size_in_mb != 0)
                {
                    size_in_mb = (t_dim_factors & balancer::FactorizedInt(vstack_factor_per_core)).get_nearest_factor_le(mb_limit);
                }
            }
        }

        const uint32_t t_stream_c = op_model.t_stream_factor.c;
        const uint32_t consumer_t_stream_c = consumer_op_model.t_stream_factor.c;
        
        // Need to hstack.
        if (t_stream_c > consumer_t_stream_c)
        {
            if (t_stream_c % consumer_t_stream_c != 0)
            {
                // T stream factor must be a divisible by the consumer t stream factor.
                // Cannot calculate constraints for this case.
                continue;
            }
            const uint32_t hstack_factor = t_stream_c / consumer_t_stream_c;
            const uint32_t consumer_grid_c = consumer_op_model.grid_shape.c;

            if (hstack_factor > size_in_mb and hstack_factor % size_in_mb != 0)
            {
                size_in_mb = (t_dim_factors & balancer::FactorizedInt(hstack_factor)).get_nearest_factor_le(mb_limit);
            }

            if (hstack_factor > size_in_mb and hstack_factor > consumer_grid_c)
            {
                if (hstack_factor % consumer_grid_c != 0)
                {
                    // HStack factor must be divisible by the grid c of the consumer op.
                    continue;
                }
                const uint32_t hstack_factor_per_core = hstack_factor / consumer_grid_c;

                if (hstack_factor_per_core % size_in_mb != 0)
                {
                    size_in_mb = (t_dim_factors & balancer::FactorizedInt(hstack_factor_per_core)).get_nearest_factor_le(mb_limit);
                }
            }
        }
    }

    output_buffer.buffer_factor = size_in_mb * 2;
    output_buffer.l1_size_tiles = tiles_per_mb * output_buffer.buffer_factor;

    added_tiles = (output_buffer.buffer_factor - initial_mb) * tiles_per_mb;

    return added_tiles;
}


// This function is attempting to add buffering along a given path in a graph, with the goal of minimizing the number of
// nops that need to be inserted. It does this by iterating over the nodes in the path and attempting to add as much
// buffering as possible at each node, using available memory space and respecting certain environment variables and
// input parameters. If it is not possible to add enough buffering, the function will add nop insertion instructions to
// a provided vector.
void add_buffering_on_path(
    const Graph *graph,
    const std::vector<Node *> path,
    std::uint32_t long_path_required,
    std::uint32_t short_path_available,
    tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        &instructions,
    const tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        &previous_ins_instructions,
    balancer::OpModelMap *op_models_post_placer,
    balancer::OpModels *op_models,
    const std::uint32_t usable_l1_size,
    std::function<int(const tt::balancer::OpModel &)> buffering_factor,
    const ForkJoin &fj,
    FJGraph& fj_graph)
{
    // Go along the path and try to add buffering as much as it fits
    std::uint32_t to_add = long_path_required - short_path_available;
    // growth of available buffering is calculated after increasing input buffers and adding nops on path.
    // this value reflects how many more tiles on input path will be able to buffer.
    std::uint32_t additional_available_buff = 0;

    // If enabled we will add buffering queues instead of NOPs to buffer fork-joins.
    const bool add_buffer_queues = env_as<int>("PYBUDA_FORK_JOIN_BUF_QUEUES", 0);

    // currently, we maximize input buffers by default.
    const bool maximize_buffers = env_as<bool>("PYBUDA_MAX_FORK_JOIN_BUF", 1);

    // If enabled we will expand fork node output buffer.
    const bool expand_fork_output_buffer = env_as<bool>("PYBUDA_FORK_JOIN_EXPAND_FORK_OUTPUT_BUF", 1);

    // If enabled we will expand output buffers (instead of input buffers) of the nodes on the path.
    const bool expand_output_buffers = env_as<bool>("PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS", 0);

    // If enabled we skip expanding buffers for regular ops, to force adding NOPs.
    const bool skip_expanding_buffers = env_as<bool>("PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS");

    const int max_queue_mem =
        1024 * 1024 * 1024;  // 1GB, this is ad hoc limit for maximum memory buffering queues can consume on one chip.
    const float scale_usable_l1_size = 0.95;

    Node *prev_node = nullptr;
    // current_output_multiplier keeps track of expansions and reductions of fork outputs
    float current_output_multiplier = 1.0;
    Node *join  = nullptr;

    for (Node *node : path)
    {
        bool curr_node_is_join = (join == node && node != nullptr);

        // join != nullptr is if node is inside of an inner fork-join. join is then pointing to the join of the inner
        // fork-join so we know on which node this inner fork-join is finishing: node == join)
        bool outside_fj = (join == nullptr && !curr_node_is_join);
        if (join == nullptr || join == node)
        {
            FJBufferingInfo fj_buff_info = fj_graph.find_sub_fork_join_from_node(fj, path, node);
            join = fj_buff_info.join;
        }

        if (prev_node == nullptr)
        {
            prev_node = node;

            balancer::OpModel& op_model = get_op_model(op_models_post_placer, op_models, node);
            if (expand_fork_output_buffer && to_add > (uint32_t)op_model.block_shape().volume_no_t())
            {
                // In cases when we need to buffer more than macro block size of tiles, we may end up
                // in a situation where just expanding input buffers on the short path won't be
                // sufficient - due to the fact that backend cannot generate pipes/streams to utilize
                // 100% of the allocated input buffers.
                //
                // To workaround this limitation, we need to additionally expand output buffer of the fork node.
                uint32_t added_tiles = expand_output_buffer(graph, node, op_model, scale_usable_l1_size, usable_l1_size, op_models_post_placer, op_models);

                if (added_tiles > 0)
                {
                    log_debug(LogGraphCompiler, "Expanded fork node ({}) output buffers to a total of {} macro blocks.", node->name(), op_model.output_buffers[0].buffer_factor);
                }
            }

            continue;
        }

        // If PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS is enabled we don't want to expand buffers for regular ops.
        // This will cause the algorithm to resort to adding nops/buffers to buffer the path.
        // NOTE: Here we don't skip buffering ops (nops), because after we add nops, this code will be executed again to expand their input buffers.
        if (skip_expanding_buffers && !node->as<graphlib::BudaOpNode>()->is_buffering_op())
        {
            prev_node = node;
            continue;
        }

        balancer::OpModel &op_model = get_op_model(op_models_post_placer, op_models, node);
        uint32_t output_buffer_tiles_added = 0;

        // If enabled, expand output buffer - except for join node since that doesn't help with fork-join buffering.
        if (expand_output_buffers && path.back() != node)
        {
            output_buffer_tiles_added = expand_output_buffer(graph, node, op_model, scale_usable_l1_size, usable_l1_size, op_models_post_placer, op_models);
        }

        float next_output_multiplier = 1.0;
        for (Edge e : graph->get_edges(prev_node, node))
        {
            if (e.edge_type == graphlib::EdgeType::kData)
            {
                // it turns out that sometimes pipegen requires more l1 space than what we model.
                // that is why we won't use full usable l1 space in nops, but scale it with scale_usable_l1_size
                // Calculate available size in tiles
                std::uint32_t available_space = 0;
                if (scale_usable_l1_size * usable_l1_size > op_model.get_l1_memory_usage())
                {
                    available_space =
                        (scale_usable_l1_size * usable_l1_size - op_model.get_l1_memory_usage()) /
                        balancer::tile_size_bytes(op_model.input_buffers.at(e.consumer_input_port_id).data_format);
                }
                else
                {
                    // if available_space is 0 then we skip to the next op in path. 
                    continue;
                }
                std::uint32_t grid_size = op_model.grid_shape.volume();
                available_space *= grid_size;

                std::uint32_t add_amount = 0;
                std::uint32_t effective_add_amount = 0;
                auto tms = graph->get_edge_attributes(e)->get_tms();
                float input_multiplier = current_output_multiplier;
                for (auto tm : tms)
                {
                    // broadcast changes the volume of the complete tensor. Now if we have some number of tiles free in
                    // L1 to buffer the data, these tiles will effectively buffer less if the size of the tensor
                    // increased. It is not the same if you can buffer 100 tiles when one output is 10 tiles as when
                    // that output passes though the broadcast (with expansion 10) and effectively becomes 10 times
                    // bigger. Then, you can only buffer one tensor with 100 tiles. So we can't blindly add up free L1
                    // space throughout the buffering path without weighting with respect to tensor volume change. This
                    // is because we want to calculate how much tiles can we buffer with respect to the begining of the
                    // path.
                    if (tm.op == "broadcast")
                        input_multiplier /= (float)std::get<int>(tm.attr[1]);
                }
                std::uint32_t effective_available_space = available_space * input_multiplier;
                next_output_multiplier = get_output_multiplier(node, input_multiplier, op_model, e.consumer_input_port_id);

                // Only expand input buffers if expanding output buffers is disabled.
                if (!expand_output_buffers)
                {
                    if (maximize_buffers)
                    {
                        // Take up all available space
                        effective_add_amount = effective_available_space / grid_size;
                        add_amount = available_space / grid_size;
                    }
                    else
                    {
                        effective_add_amount = ceil((float)std::min(to_add, effective_available_space) / (float)grid_size);
                        add_amount = effective_add_amount / input_multiplier;
                    }
                    add_amount -=
                        add_amount % (op_model.input_buffers.at(e.consumer_input_port_id).block_shape.volume_no_t());

                }

                effective_add_amount = add_amount * input_multiplier + output_buffer_tiles_added * next_output_multiplier;
                if (add_amount > 0 && outside_fj)
                {
                    // we only want to increase input l1 buffers to nodes that are not part of already buffered inner
                    // fork-joins thus outside_fj tells if current node is outside of already buffered sub fjs.
                    op_model.input_buffers.at(e.consumer_input_port_id).l1_size_tiles += add_amount;
                    op_model.input_buffers.at(e.consumer_input_port_id).size_tiles_override = true;
                }

                // add_amount is the additional amount of l1 space (in tiles) that we allocated for fork join buffering.
                // Due to contraction and expansion of tensor troughout the path, add_amount has to be scaled with
                // input_multiplier factor. input multiplier because we add this add_amount of tiles to input buffers of
                // the op. If we added it to output buffers we would use next_output_multiplier for scalling. Note that
                // we still use up add_amount space in l1, but effectively for buffering it decreases to_add for
                // add_amount * input_multiplier tiles.

                // for one core. we have to multiply it with grid size.
                // Prevent underflow.
                if (outside_fj)
                {
                    if (effective_add_amount * grid_size > to_add)
                    {
                        additional_available_buff += to_add;
                        to_add = 0;
                    }
                    else
                    {
                        additional_available_buff += effective_add_amount * grid_size;
                        to_add -= effective_add_amount * grid_size;
                    }
                }
                log_trace(
                    LogGraphCompiler,
                    "Available l1 size for {}: {}, grid_size={}, add_amount={}, remaining to_add={}",
                    node->name(),
                    available_space,
                    grid_size,
                    add_amount,
                    to_add);
            }

            if ((to_add == 0) && !maximize_buffers)
                break;
        }
        current_output_multiplier = next_output_multiplier;
        if ((to_add == 0) && !maximize_buffers)
            break;

        prev_node = node;
    }

    // Optionally always add a nop on short path that is a direct connection, as a workaround for the 2K tile limit
    bool always_add = ((path.size() == 2) && env_as<bool>("PYBUDA_NOP_ON_DIRECT_SHORT_PATH"));

    if (always_add || (to_add > 0))
    {
        log_debug(
            LogGraphCompiler, "Fork join long path requires additional buffering of shorter path {} tiles", to_add);
        // insert NOPs or queue if number of tiles exceeds threshold
        balancer::OpModel &op_model = get_op_model(op_models_post_placer, op_models, path[0]);

        std::uint32_t tile_size = balancer::tile_size_bytes(op_model.output_buffers.at(0).data_format);

        int buff_mem_consumption = buffering_queues_mem_consumption(instructions) +
                                   buffering_queues_mem_consumption(previous_ins_instructions);
        Node *src = path[0];
        std::vector<Node*> dests;
        // currently if src is recompute, we skip adding queue
        // because of possible graph change (reconnecting consumers from recompute node)
        // that results in hang.
        bool src_is_recompute = is_recompute(graph, src);

        // if there is sub fork-join from fork of current fj (fj) on path, we have to add nop effectivaly before sub
        // fork-join. we do that by adding instructions for mergeable nops on both paths of sub fork-join. Nops with
        // mergeable tag will be merged in one nop if they have same source (in method merge_tagged_nops_with_same_src)
        auto [join, req, avail, sub_fj] = fj_graph.find_sub_fork_join_from_node(fj, path, src);
        if (join != nullptr)
        {
            dests.push_back(sub_fj->first[1]);
            dests.push_back(sub_fj->second[1]);
        }
        else
        {
            dests.push_back(path[1]);
        }
        bool merge_nops = dests.size() > 1;

        if (add_buffer_queues && buff_mem_consumption < max_queue_mem && !src_is_recompute)
        {
            auto edges = graph->user_data_edges(src);
            for (std::uint32_t fork_id = 0; fork_id < edges.size(); fork_id++)
            {
                for (Node *dest : dests)
                {
                    graphlib::Edge e = edges[fork_id];
                    if (e.consumer_node_id == dest->id())
                    {
                        // number if entries in queue is 2 * microbatch_size at maximum. We take the minimum of that
                        // upper limit and estimation we get from padding requirement to_add (which is number of tiles
                        // we have to padd on path) When one path of fork join is much longer than other this to_add
                        // becomes large. That would increase number of tensors we have to buffer. Luckily for us,
                        // maximum number of tensors that can be inside one queue ,that is inside one epoch, is
                        // microbatch_size.
                        int num_entries = std::min(
                            2 * graph->get_microbatch(),
                            (int)ceil((float)to_add / (float)op_model.op_shape.outputs.at(0).volume_in_tiles()));
                        int queue_size = (int)(ceil(to_add * tile_size));  // in bytes

                        if (num_entries > 0)
                        {
                            log_trace(
                                LogGraphCompiler,
                                "Adding instruction for buffering queue from {} to {} with {} entries",
                                src->name(),
                                dest->name(),
                                num_entries);

                            // if dests have more than one element that means that I want to add queue with source src
                            // but more than 1 destination. Even though I make 2 instructions, later there won't be 2
                            // queues but one that feeds to 2 consumers if dests.size() is 2 for example.
                            std::shared_ptr<InsertionInstruction> ins = std::make_shared<QueueInsertionInstruction>(
                                src->name() /* src */,
                                dest->name() /* dest */,
                                false /* hoist_tms */,
                                num_entries,
                                queue_size,
                                e.consumer_input_port_id /* input_id */,
                                fork_id /* fork_id */);
                            InsInstructionUniqueId key = ins->unique_id();
                            insert_queue_ins_to_instructions(instructions, key, ins);
                        }
                    }
                }
            }

            fj_graph.add_nop_buffered_fj(&fj);
        }
        else
        {
            // Some heuristic to guess how many nops we need
            float nop_buffering = usable_l1_size / (float)tile_size;
            // std::cout << "Expect " << nop_buffering << " tiles per nop, need to add " << to_add << std::endl;
            /*
            add at most a third of nops needed, since disturbance of placement
            will shift epochs and we might not need them any more
            We don't add all necesarry nops in one step. On the contrary, we add fraction of needed nops in each pass of
            pre-placer post-placer loop in compile.py. This is because adding nops can cause current fork-join to span
            across two epochs, thus elliminating further need for adding new nops (because e2e queues act as buffers).
            */
            float nop_base_buffer = to_add / nop_buffering;

            // In case buffering requirements are abysmal skip buffering.
            //
            if (nop_base_buffer < 0.1)
                return;

            // If we are using inline buffering within epoch we can add all NOPs at once.
            // Legacy post placer path is adding one third at a time due to op shifts accross epochs.
            //
            int buffering_step = op_models_post_placer != nullptr ? 3 : 1;
            int buffering_scale = buffering_factor(op_model) * buffering_step;
            std::uint32_t nop_count = (uint32_t)std::ceil(to_add / (nop_buffering * buffering_scale));

            // Check if we are trying to add unreasonable amount of NOPs.
            // Currently, unreasonable is defined as "more that can fit on grayskull (10x12 grid)".
            if (nop_count > 120)
            {
                log_warning(LogGraphCompiler, "Trying to add large number of NOPs for buffering.");
            }

            log_trace(
                LogGraphCompiler,
                "Ask for {} nops from {}, to_add: {}, tile_size: {}",
                nop_count,
                src->name(),
                to_add,
                tile_size);

            auto edges = graph->user_data_edges(src);
            for (std::uint32_t fork_id = 0; fork_id < edges.size(); fork_id++)
            {
                for (Node *dest : dests)
                {
                    graphlib::Edge e = edges[fork_id];
                    if (e.consumer_node_id == dest->id())
                    {
                        static const bool fix_2351 = env_as<bool>("PYBUDA_TEMP_FIX_2351", false);
                        InsInstructionUniqueId key = InsInstructionUniqueId(
                            src->name(),
                            dest->name(),
                            e.consumer_input_port_id,
                            fork_id,
                            merge_nops,
                            !fix_2351 /* is_fj_buffering */);  // this should be set to true, unless the env is applied

                        if (instructions.count(key) > 0)
                        {
                            if (NopInsertionInstruction *nop_instr =
                                    dynamic_cast<NopInsertionInstruction *>(instructions[key].get()))
                            {
                                if (nop_instr->nop_count < nop_count)
                                {
                                    nop_instr->set_nop_count(nop_count);
                                    // we already added
                                    additional_available_buff +=
                                        (nop_count - nop_instr->nop_count) * nop_buffering * buffering_factor(op_model);
                                }
                            }
                        }
                        else
                        {
                            // instruction doesn't exist in map of instructions
                            additional_available_buff += nop_count * nop_buffering * buffering_factor(op_model);
                            std::shared_ptr<InsertionInstruction> ins = std::make_shared<NopInsertionInstruction>(
                                src->name() /* src */,
                                dest->name() /* dest */,
                                false /* hoist_tms */,
                                nop_count /* nop_count */,
                                e.consumer_input_port_id /* input_id */,
                                fork_id /* fork_id */,
                                merge_nops,
                                true /* is_fj_buffering */);
                            instructions[key] = ins;
                        }
                    }
                }
            }

            fj_graph.add_nop_buffered_fj(&fj);
        }
    }

    FJBufferingInfo fj_buff_info =
        FJBufferingInfo(fj.second.back(), long_path_required, short_path_available + additional_available_buff, &fj);
    fj_graph.update_buffered_fj_map(fj, fj_buff_info);
}

/*
Returns std::tuple<bool,int,int>
First variable in output is true if instructions map is true subset of previous_instructions map. This means that for
each key in instructions there is the key in previous_instructions and values match. Then, second and third variables in
output are 0.
*/
std::tuple<bool, int, int> is_subset_of_instructions(
    const tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        &instructions,
    const tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        &previous_instructions)
{
    bool instr_not_updated = true;
    int num_new_nops = 0;
    int num_new_queues = 0;
    for (auto elem : instructions)
    {
        InsInstructionUniqueId key = elem.first;
        InsertionInstruction *instr = elem.second.get();
        if (previous_instructions.count(key) == 0)
        {
            // if new instructions contain insertion instruction key and previous_instructions don't
            if (instr->instr_type == InstructionType::QueueInstruction)
            {
                num_new_queues++;
            }
            else if (instr->instr_type == InstructionType::NopInstruction)
            {
                num_new_nops += static_cast<NopInsertionInstruction *>(instr)->nop_count;
            }
            else
            {
                log_error("Unsupported instruction type");
            }
            instr_not_updated = false;
        }
        else
        {
            // if previous_instructions contain instruction key and that instruction is NopInstruction
            // we stil have to check if nop count is unchanged. If nop count is changed then we still return false
            InsertionInstruction *prev_instr = previous_instructions.at(key).get();
            if (prev_instr->instr_type == InstructionType::NopInstruction)
            {
                if (instr->instr_type == InstructionType::NopInstruction)
                {
                    int prev_nop_count = static_cast<NopInsertionInstruction *>(prev_instr)->nop_count;
                    int curr_nop_count = static_cast<NopInsertionInstruction *>(instr)->nop_count;
                    if (prev_nop_count != curr_nop_count)
                    {
                        instr_not_updated = false;
                        num_new_nops += (curr_nop_count - prev_nop_count);
                    }
                }
                else if (instr->instr_type == InstructionType::QueueInstruction)
                {
                    num_new_queues++;
                }
                else
                {
                    log_error("Unsupported instruction type");
                }
            }
        }
    }
    return std::tuple<bool, int, int>(instr_not_updated, num_new_nops, num_new_queues);
}

/*
Makes new tt::ordered map containing instructions and previous_instructions. If both maps contain the same nop
instruction key we want to add nop_counts from previous_instructions to new nop_count. This hapens all the time because
we don't add all necesarry nops in one step. On the contrary, we add fraction of needed nops in each pass of pre-placer
post-placer loop in compile.py . This is because adding nops can cause fork-join to span across two epochs, thus
elliminating further need for adding new nops (because e2e queues act as buffers).
*/
tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
append_prev_instr(
    tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        &instructions,
    const tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        previous_instructions)
{
    tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        combined_instructions = previous_instructions;
    for (auto elem : instructions)
    {
        // Iterate through current instructions
        InsInstructionUniqueId key = elem.first;
        std::shared_ptr<InsertionInstruction> instr = elem.second;
        if (combined_instructions.count(key) == 0)
        {
            // if combined instructions doesn't contain current key
            combined_instructions[key] = instr;
        }
        else
        {
            // if combined instructions contains current key then we ask if value is instruction of type NopInstruction
            // actually there should not be the case where two Queue instructions share the same key in instructions and
            // previous_instructions maps because if queue instruction is in previous_instructions map then that
            // fork-join is resolved and new instructions won't have that queue. however for future it is better to
            // check.
            if (instr->instr_type == InstructionType::NopInstruction &&
                combined_instructions.at(key)->instr_type == InstructionType::NopInstruction)
            {
                // if we already have instructions for nop insertion on that place, we just update nop_count
                NopInsertionInstruction *nop_instr = static_cast<NopInsertionInstruction *>(instr.get());
                NopInsertionInstruction *instr_to_modify =
                    static_cast<NopInsertionInstruction *>(combined_instructions[key].get());
                instr_to_modify->set_nop_count(instr_to_modify->nop_count + nop_instr->nop_count);
            }
        }
    }
    return combined_instructions;
}

// Creates QueueInsertionInstruction to add queue between first and second node on path "path_to_buffer", and appends
// that instruction to map of current instructions ,"instructions". Num entries for the queue is "buf_queue_num_entries"
void add_queue_instr_based_on_queues_on_other_path(
    Graph *graph,
    std::vector<Node *> path_to_buffer,
    int buf_queue_num_entries,
    balancer::OpModelMap *op_models_post_placer,
    balancer::OpModels *op_models,
    tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        &instructions)
{
    Node *src = path_to_buffer[0];
    Node *dest = path_to_buffer[1];

    auto edges = graph->user_data_edges(src);
    for (std::uint32_t fork_id = 0; fork_id < edges.size(); fork_id++)
    {
        graphlib::Edge e = edges[fork_id];
        if (e.consumer_node_id == dest->id())
        {
            balancer::OpModel &op_model = get_op_model(op_models_post_placer, op_models, path_to_buffer[0]);
            std::uint32_t tile_size = balancer::tile_size_bytes(op_model.output_buffers.at(0).data_format);
            std::uint32_t queue_size = (std::uint32_t)(ceil(
                buf_queue_num_entries * (float)op_model.op_shape.outputs.at(0).volume_in_tiles() * tile_size));

            std::shared_ptr<InsertionInstruction> queue_ins = std::make_shared<QueueInsertionInstruction>(
                src->name() /* src */,
                dest->name() /* dest */,
                false /* hoist_tms */,
                buf_queue_num_entries,
                queue_size,
                e.consumer_input_port_id /* input_id */,
                fork_id /* fork_id */);
            InsInstructionUniqueId key = queue_ins->unique_id();
            insert_queue_ins_to_instructions(instructions, key, queue_ins);
        }
    }
}

// Returns a map of pointers to insertion instructions needed for buffering the graph.
tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
generate_graph_buffering(
    Graph *graph,
    FJGraph &fj_graph,
    balancer::OpModelMap *op_models_post_placer,
    balancer::OpModels *op_models,
    const std::uint32_t usable_l1_size,
    const tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        previous_ins_instructions,
    std::function<int(const tt::balancer::OpModel &)> buffering_factor)
{
    tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        instructions;

    std::vector<const ForkJoin *> sorted_fork_joins = fj_graph.get_topo_sorted_fjs();

    bool dump_debug_info = (env_as<bool>("PYBUDA_FORK_JOIN_DEBUG_INFO")) ? true : false;
    std::stringstream node_debug_info;

    for (std::size_t j = 0; j < sorted_fork_joins.size(); j++)
    {
        const ForkJoin &fj = *sorted_fork_joins[j];
        if (dump_debug_info)
        {
            // we log fork-join that is buffered to track order of buffering in graph
            node_debug_info << "buffering fork-join: fork node name: " << fj.first[0]->name()
                            << " join node name: " << fj.first.back()->name() << std::endl;
        }
        // std::cout << "== FORK JOIN ==" << std::endl;
        // print_fork_join(fj);

        // Figure out if buffering is needed.
        auto [path0_req, path0_has_buff_queue, path0_buf_queue_num_entries] = get_buffering_requirement(
            graph, op_models_post_placer, op_models, fj.first, fj, fj_graph);
        auto [path1_req, path1_has_buff_queue, path1_buf_queue_num_entries] = get_buffering_requirement(
            graph, op_models_post_placer, op_models, fj.second, fj, fj_graph);

        log_trace(LogGraphCompiler, "path0_req = {}, path1_req = {}", path0_req, path1_req);

        if (path0_has_buff_queue != path1_has_buff_queue )
        {
            // one of the paths has buffering queue, and other doesn't. We will add queue to the one that doesn't have
            // queue. after that, we don't need buffering of that fork-join, because all buffering queues have
            // num_entries equal to microbatch size this guaranties that both paths will be able to buffer all tensors
            // that pass through them in one epoch. 
            // These queue instructions don't conform to maximum queue memory
            // consumption threshold (max_queue_mem). This threshold is introduced for buffering queues that replace
            // nops in buffering fork joins that would require many nops. If all queue instructions exceed
            // max_queue_mem, we will still add buffering queues on fork-joins where we have buffering queue in one
            // path, but we will stop buffering regular fork-joins with buffering queues, and transfer to nops regardles
            // of the path difference
            if(path0_has_buff_queue)
            {
                // path0 has buffering queue and path1 doesn't. we want to add buffering queue to path1 to balance out fork-join

                // create instruction for buffering queue between first and second node of path1 (fj.second), and add it to
                // existing instructions.
                add_queue_instr_based_on_queues_on_other_path(graph, fj.second, path0_buf_queue_num_entries, op_models_post_placer, op_models, instructions);
            }
            else
            {
                // path1 has buffering queue and path0 doesn't. we want to add buffering queue to path0 to balance out fork-join
                // create instruction for buffering queue between first and second node of path0 (fj.first), and add it to
                // existing instructions.
                add_queue_instr_based_on_queues_on_other_path(graph, fj.first, path1_buf_queue_num_entries, op_models_post_placer, op_models, instructions);
            }
        }

        // if path0_req < path1_req, then path 0 is faster path, and path 1 is slower.
        if (path0_req < path1_req && !path0_has_buff_queue && !path1_has_buff_queue)
        {
            // Path 0 is the short path
            std::uint32_t available_bufferings = std::get<0>(get_available_buffering(
                graph, op_models_post_placer, op_models, fj.first, fj, fj_graph));
            log_trace(LogGraphCompiler, "path0 available: {}", available_bufferings);
            if (path1_req > available_bufferings)
            {
                add_buffering_on_path(
                    graph,
                    fj.first,
                    path1_req,
                    available_bufferings,
                    instructions,
                    previous_ins_instructions,
                    op_models_post_placer,
                    op_models,
                    usable_l1_size,
                    buffering_factor,
                    fj,
                    fj_graph);
            }
        }
        // if path1_req < path0_req, then path 1 is faster path, and path 0 is slower.
        else if (path1_req < path0_req && !path1_has_buff_queue && !path0_has_buff_queue)
        {
            // Path 1 is the short path
            std::uint32_t available_bufferings = std::get<0>(get_available_buffering(
                graph, op_models_post_placer, op_models, fj.second, fj, fj_graph));
            log_trace(LogGraphCompiler, "path1 available: {}", available_bufferings);
            if (path0_req > available_bufferings)
            {
                add_buffering_on_path(
                    graph,
                    fj.second,
                    path0_req,
                    available_bufferings,
                    instructions,
                    previous_ins_instructions,
                    op_models_post_placer,
                    op_models,
                    usable_l1_size,
                    buffering_factor,
                    fj,
                    fj_graph);
            }
        }
    }

    log_debug(LogGraphCompiler, "Buffering sequence of fork-joins: \n{}", node_debug_info.str());
    // log new instructions. New instructions can be instructions that are completely new, and those that were updated
    // for example, we increased nop_count in nop instruction, or num_entries in queue instruction
    if (instructions.size() != 0)
    {
        log_trace(LogGraphCompiler, " new or updated instructions: ");
        for (const auto &i : instructions)
        {
            log_trace(LogGraphCompiler, " - src: {}, dest: {}", i.second->src, i.second->dest);
        }
    }
    // order of the arguments matters since tt::ordered_map keeps track of the adding order
    // therefore we have to add previos_ins_instructions before we add instructions
    return append_prev_instr(instructions, previous_ins_instructions);
}

// Generate buffering instructions for fork-join buffering.
// op_models_post_placer is passed in if we are in the post-placer phase - legacy path.
// op_models is passed in balancing phase for "inline" buffering - new path.
//
FJBufferingResult insert_fork_join_buffering(
    graphlib::Graph *graph,
    balancer::OpModelMap *op_models_post_placer,
    balancer::OpModels *op_models,
    const std::uint32_t usable_l1_size,
    const tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        &previous_ins_instructions,
    std::function<int(const tt::balancer::OpModel &)> buffering_factor)
{
    // We can't change the graph, only adjust buffer sizes.
    //
    TT_ASSERT(
        (op_models_post_placer or op_models) and !(op_models_post_placer and op_models),
        "op_models_post_placer or op_models must be passed in but not both!");

    // Disable if env variable set
    if (env_as<bool>("PYBUDA_DISABLE_FORK_JOIN_BUF"))
        return {};

    //
    // Find fork-joins
    //
    FJGraph fj_graph = FJGraph(graph);
    if (fj_graph.get_fjs().size() == 0)
        return {};  // nothing to do

    //
    // Find buffering locations due to mismatched paths, and adjust buffers
    //
    tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        instructions = generate_graph_buffering(
            graph,
            fj_graph,
            op_models_post_placer,
            op_models,
            usable_l1_size,
            previous_ins_instructions,
            buffering_factor);

    // if instructions is not subset of previous instructions, then we have some new instructions (nop or queue)

    auto new_instr_tuple = is_subset_of_instructions(instructions, previous_ins_instructions);
    if (!std::get<0>(new_instr_tuple))
    {
        log_debug(
            LogGraphCompiler,
            "Added more buffering instructions. Additional Nops added: {}; Additional Queues added: {} ",
            std::get<1>(new_instr_tuple),
            std::get<2>(new_instr_tuple));
    }

    if (env_as<bool>("PYBUDA_DISABLE_FORK_JOIN_NOPS"))
        instructions.clear();

    FJBufferingResult res;
    res.instructions = instructions;

    for (auto& fj: fj_graph.get_nop_buffered_fjs())
    {
        res.nop_buffered_fjs.push_back(*fj);
    }

    return res;
}

void upsize_dram_input(graphlib::Graph *graph, balancer::OpModelMap &op_models, const std::uint32_t usable_l1_size)
{
    for (Node *node : graph->nodes())
    {
        if (node->node_type() != graphlib::kBudaOp)
            continue;

        auto edges = graph->operand_data_edges(node);
        std::vector<std::size_t> queue_ops;
        for (std::size_t i = 0; i < edges.size(); i++)
        {
            /*Node *op = graph->node_by_id(edges[i].producer_node_id);
            if (op->node_type() == graphlib::kQueue)
            {
                queue_ops.push_back(i);
            }
            else if (op->node_type() == graphlib::kInput)
            {
                if (!op->as<graphlib::InputNode>()->is_prologue())
                    queue_ops.push_back(i);
            }*/
            queue_ops.push_back(i);
        }

        if (queue_ops.size() > 0)
        {
            balancer::OpModel &op_model = op_models.at(node->name());
            std::uint32_t available_space = usable_l1_size - op_model.get_l1_memory_usage();
            log_trace(
                LogGraphCompiler,
                "Upsize dram for {}: usable: {}, usage: {}, available: {}",
                node->name(),
                usable_l1_size,
                op_model.get_l1_memory_usage(),
                available_space);
            std::uint32_t to_add = 1.0 * available_space / (float)queue_ops.size();

            for (std::size_t i : queue_ops)
            {
                // Calculate available size in tiles
                std::uint32_t to_add_tiles =
                    1.0 * to_add / (float)balancer::tile_size_bytes(op_model.input_buffers.at(i).data_format);
                to_add_tiles -= to_add_tiles % (op_model.input_buffers.at(i).block_shape.volume_no_t());
                log_trace(
                    LogGraphCompiler,
                    " - for operand {}, to_add: {}, current: {}",
                    i,
                    to_add_tiles,
                    op_model.input_buffers.at(i).l1_size_tiles);
                op_model.input_buffers.at(i).l1_size_tiles += to_add_tiles;
                op_model.input_buffers.at(i).size_tiles_override = true;
            }
        }
    }
}

}  // namespace tt
