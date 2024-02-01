// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "pattern_matcher/pattern_matcher.hpp"
#include "utils/logger.hpp"

#include <iostream>
#include <string>
#include <unordered_set>

#include <boost/graph/copy.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/vf2_sub_graph_iso.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/variant/get.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "fmt/core.h"

using std::string;

using tt::LogPatternMatcher;

namespace pattern_matcher {

int num_subgraph_pattern_matches(graph_type& small_graph, graph_type& large_graph, int max_matches = INT_MAX) {
    int total_matches = 0;

    std::unordered_map<NodeId, std::unordered_set<NodeId>> unique_matches;

    auto callback = [&](auto bijection, auto) {
        for (auto v : boost::make_iterator_range(vertices(small_graph))) {
            NodeId matched_id = large_graph[get(bijection, v)].node_id;
            if (unique_matches.find(matched_id) == unique_matches.end()) {
                unique_matches[small_graph[v].node_id].insert(matched_id);
            } else {
                return false;
            }
        }

        total_matches += 1;
        if (total_matches > max_matches) {
            return false;
        }
        return true;
    };

    auto edge_predicate = [&](auto edge_a, auto edge_b) {
        return small_graph[edge_a].producer_output_edge_index == large_graph[edge_b].producer_output_edge_index
            and small_graph[edge_a].consumer_input_edge_index == large_graph[edge_b].consumer_input_edge_index ;
    };
    auto vertex_predicate = [&](auto vertex_a, auto vertex_b) {
        if (small_graph[vertex_a].op_type == "*") {
            return true;
        }

        return small_graph[vertex_a].op_type == large_graph[vertex_b].op_type;
    };
    boost::vf2_subgraph_iso(
        small_graph,
        large_graph,
        callback,
        boost::vertex_order_by_mult(small_graph),
        boost::edges_equivalent(edge_predicate).vertices_equivalent(vertex_predicate));

    return total_matches;
}


struct EdgePredicate {
  EdgePredicate() = default;
  EdgePredicate(graph_type* graph, VertexId valid_start, VertexId valid_end)
      : graph_(graph), valid_start_(valid_start), valid_end_(valid_end) {}

  template <typename Edge>
  bool operator()(const Edge& edge) const {
    // Include this edge iff both endpoints belong in the valid range

    auto source_vertex = source(edge, *graph_);
    if (source_vertex < valid_start_ or source_vertex > valid_end_) {
        return false;
    }

    auto target_vertex = target(edge, *graph_);
    if (target_vertex < valid_start_ or target_vertex > valid_end_) {
        return false;
    }

    return true;
  }

  graph_type* graph_;
  VertexId valid_start_;
  VertexId valid_end_;
};

struct VertexPredicate {
  VertexPredicate() = default;
  VertexPredicate(graph_type* graph, VertexId valid_start, VertexId valid_end)
      : graph_(graph), valid_start_(valid_start), valid_end_(valid_end) {}

  template <typename VertexId>
  bool operator()(const VertexId& v) const {
    if (v < valid_start_) return false;
    if (v > valid_end_) return false;

    // Include this vertex iff it has at least one connection to a vertex in
    // the allowable set.
    for (auto in_edge : make_iterator_range(in_edges(v, *graph_))) {
        VertexId source_vertex = source(in_edge, *graph_);
        if (source_vertex >= valid_start_ and source_vertex <= valid_end_) {
            return true;
        }
    }

    for (auto out_edge : make_iterator_range(out_edges(v, *graph_))) {
        VertexId target_vertex = target(out_edge, *graph_);
        if (target_vertex >= valid_start_ and target_vertex <= valid_end_) {
            return true;
        }
    }

    return false;
  }

  graph_type* graph_;
  VertexId valid_start_;
  VertexId valid_end_;
};


graph_type generate_pattern_subgraph(graph_type& graph, VertexId start, VertexId end)
{
    using filtered_graph_type = boost::filtered_graph<graph_type, EdgePredicate, VertexPredicate>;
    EdgePredicate edge_pred = EdgePredicate(&graph, start, end);
    VertexPredicate vert_pred = VertexPredicate(&graph, start, end);
    filtered_graph_type filtered_graph = boost::filtered_graph(graph, edge_pred, vert_pred);

    // Bad design of boost::filtered_graph.. we can't use it directly in vf2
    // We can incrementally build pattern graph for speedup if this ends up being a bottleneck
    graph_type pattern_graph;
    boost::copy_graph(filtered_graph, pattern_graph);
    return pattern_graph;
}

int get_max_vertices_to_include(const graph_type& graph, int num_expected_matches)
{
    // heuristic to limit the subgraphs generated. Given N-expected matches,
    // we roughly know that the pattern subgraph we should search for is roughly in
    // num_nodes() / N plus some preamble nodes.
    return (num_vertices(graph) / num_expected_matches) + 1;
}
static int get_num_input_nodes(graph_type& graph) {
    int num_input_nodes = 0;

    for (auto v : boost::make_iterator_range(vertices(graph))) {
        if (graph[v].op_type == "*") {
            num_input_nodes++;
        }
    }

    return num_input_nodes;
}

std::vector<std::pair<VertexId, VertexId>> get_subgraph_vertex_start_end_pairs(graph_type& graph, int num_expected_matches) {
    int window_size = 5;
    int max_vertices_to_include = get_max_vertices_to_include(graph, num_expected_matches);
    std::vector<std::pair<VertexId, VertexId>> subgraph_vertex_start_end_pairs;

    int first_non_input_vertex_id = get_num_input_nodes(graph);
    for (int start_vertex_id = first_non_input_vertex_id; start_vertex_id < first_non_input_vertex_id + window_size; ++start_vertex_id) {
        int end_vertex_id_begin = std::max(start_vertex_id, start_vertex_id + max_vertices_to_include - window_size);
        int end_vertex_id_end = start_vertex_id + max_vertices_to_include + window_size;

        // NB: The fact that we go in reverse is important. The reason is because we want to start subgraph matching
        // on the largest set on nodes possibly before reducing the range.
        // TODO(jchu): may need to replace return type with std::set and guarantee largest subgraphs across differnt
        // start vertex ids are tried first.
        for (int end_vertex_id = end_vertex_id_end - 1; end_vertex_id >= end_vertex_id_begin; --end_vertex_id) {
            subgraph_vertex_start_end_pairs.emplace_back(start_vertex_id, end_vertex_id);
        }
    }
    auto largest_difference_cmp = [](const std::pair<VertexId, VertexId>& a, const std::pair<VertexId, VertexId>& b) {
        auto [a_start, a_end] = a;
        auto [b_start, b_end] = b;
        if ((a_end - a_start) == (b_end - b_start)) {
            return a_start < b_start;
        }

        return (a_end - a_start) >= (b_end - b_start);
    };
    std::sort(subgraph_vertex_start_end_pairs.begin(), subgraph_vertex_start_end_pairs.end(), largest_difference_cmp);
    return subgraph_vertex_start_end_pairs;
}


bool contains_exactly_n_subgraph_matches(graph_type& graph, int num_expected_matches) {
    for (const auto& [start_vertex_id, end_vertex_id] : get_subgraph_vertex_start_end_pairs(graph, num_expected_matches)) {
        graph_type subgraph_pattern = generate_pattern_subgraph(graph, start_vertex_id, end_vertex_id);
        int matches = num_subgraph_pattern_matches(subgraph_pattern, graph, num_expected_matches);
        if (matches == num_expected_matches) {
            return true;
        }
    }
    return false;
}

void save_dotgraph_to_ostream(std::ostream& stream, const graph_type& graph)
{
    VertexPropertyWriter<graph_type> vertex_writer(graph);
    write_graphviz(stream, graph, vertex_writer);
}

void save_dotgraph_to_file(std::string filename, const graph_type& graph)
{
    std::string dot_graph_filename = "compiled_dot_graph.txt";
    std::ofstream odotfile{filename};
    save_dotgraph_to_ostream(odotfile, graph);
    odotfile.close();
}

void save_graph_to_file(std::string filename, graph_type& graph)
{
    std::ofstream ofile{filename};
    boost::archive::text_oarchive oa{ofile};
    save(oa, graph);
    ofile.close();
}

graph_type load_graph_from_file(std::string filename)
{
    std::ifstream ifile{filename};
    boost::archive::text_iarchive ia{ifile};

    graph_type graph;
    load(ia, graph);

    VertexPropertyWriter<graph_type> vertex_writer(graph);
    return graph;
}

std::vector<NodeId> get_node_ids(graph_type& graph, const SubgraphPatternMatchMappings& subgraph_matches, int match_index)
{
    assert(match_index < (int)subgraph_matches.size());
    std::vector<NodeId> node_ids;

    for (auto v : boost::make_iterator_range(vertices(graph))) {
        NodeId node_id = (match_index == 0) ? graph[v].node_id : subgraph_matches[match_index].at(graph[v].node_id);
        node_ids.push_back(node_id);
    }

    return node_ids;
}

std::vector<NodeId> get_input_activation_node_ids(graph_type& graph, const SubgraphPatternMatchMappings& subgraph_matches, int match_index)
{
    assert(match_index < (int)subgraph_matches.size());
    std::vector<NodeId> input_node_ids;

    const std::unordered_set<std::string> input_type_strings = {
        "accumulator",
        "loss",
        "parameter",
        "constant",
        "optimizer_parameter",
    };

    for (auto v : boost::make_iterator_range(vertices(graph))) {
        const std::string& op_type = graph[v].op_type;

        bool is_input_node = graph[v].op_type == "*";
        bool is_math_op = input_type_strings.find(op_type) == input_type_strings.end();
        bool are_all_operands_inputs = true;

        auto [in_edges_start, in_edges_end] = boost::in_edges(v, graph);
        for (; in_edges_start != in_edges_end; ++in_edges_start) {
            VertexId operand = boost::source(*in_edges_start, graph);
            are_all_operands_inputs &= input_type_strings.find(graph[operand].op_type) != input_type_strings.end();
            if (not are_all_operands_inputs) {
                break;
            }
        }
        if (is_input_node or (is_math_op and are_all_operands_inputs)) {
            NodeId node_id = (match_index == 0) ? graph[v].node_id : subgraph_matches[match_index].at(graph[v].node_id);
            input_node_ids.push_back(node_id);
        }
    }

    return input_node_ids;
}

std::vector<NodeId> get_parameter_node_ids(graph_type& graph, const SubgraphPatternMatchMappings& subgraph_matches, int match_index)
{
    assert(match_index < (int)subgraph_matches.size());
    std::vector<NodeId> parameter_node_ids;

    for (auto v : boost::make_iterator_range(vertices(graph))) {
        if (boost::in_degree(v, graph) == 0 and graph[v].op_type == "parameter") {
            NodeId node_id = (match_index == 0) ? graph[v].node_id : subgraph_matches[match_index].at(graph[v].node_id);
            parameter_node_ids.push_back(node_id);
        }
    }

    return parameter_node_ids;
}

std::vector<NodeId> get_constant_node_ids(graph_type& graph, const SubgraphPatternMatchMappings& subgraph_matches, int match_index)
{
    assert(match_index < (int)subgraph_matches.size());
    std::vector<NodeId> constant_node_ids;

    for (auto v : boost::make_iterator_range(vertices(graph))) {
        if (boost::in_degree(v, graph) == 0 and graph[v].op_type == "constant") {
            NodeId node_id = (match_index == 0) ? graph[v].node_id : subgraph_matches[match_index].at(graph[v].node_id);
            constant_node_ids.push_back(node_id);
        }
    }

    return constant_node_ids;
}

std::vector<NodeId> get_output_node_ids(graph_type& graph, const SubgraphPatternMatchMappings& subgraph_matches, int match_index)
{
    assert(match_index < (int)subgraph_matches.size());
    std::vector<NodeId> output_node_ids;
    for (auto v : boost::make_iterator_range(vertices(graph))) {
        if (boost::out_degree(v, graph) == 0) {
            NodeId node_id = (match_index == 0) ? graph[v].node_id : subgraph_matches[match_index].at(graph[v].node_id);
            output_node_ids.push_back(node_id);
        }
    }

    return output_node_ids;
}

std::vector<NodeId> get_unmatched_node_ids(graph_type& graph, const SubgraphPatternMatchMappings& subgraph_matches)
{
    std::vector<NodeId> unmatched_node_ids;

    for (auto v : boost::make_iterator_range(vertices(graph))) {
        bool matched = false;
        for (size_t match_index = 0; match_index < subgraph_matches.size(); match_index++) {
            for (auto matches : subgraph_matches[match_index]) {
                if ((matches.first == graph[v].node_id) || (matches.second == graph[v].node_id)) {
                    matched = true;
                    break;
                }
            }
            if (matched) break;
        }
        if (!matched) {
            unmatched_node_ids.push_back(graph[v].node_id);
        }
    }

    return unmatched_node_ids;
}


graph_type discover_largest_subgraph_pattern(graph_type& graph, int num_expected_matches)
{
    log_debug(LogPatternMatcher, "discover_largest_subgraph_pattern(num_expected_matches={}).", num_expected_matches);
    graph_type best_subgraph_pattern;
    int max_num_subgraph_vertices = 0;
    bool is_subgraph_match_found = false;

    for (const auto& [start_vertex_id, end_vertex_id] : get_subgraph_vertex_start_end_pairs(graph, num_expected_matches)) {
        log_trace(LogPatternMatcher, "start_vertex: {}, end_vertex: {}", start_vertex_id, end_vertex_id);
        graph_type subgraph_pattern = generate_pattern_subgraph(graph, start_vertex_id, end_vertex_id);
        //save_dotgraph_to_file(fmt::format("subgraph_pattern_{}_{}.txt", start_vertex_id, end_vertex_id), subgraph_pattern);
        int num_subgraph_vertices = boost::num_vertices(subgraph_pattern);

        if (is_subgraph_match_found and num_subgraph_vertices <= max_num_subgraph_vertices) {
            // we have already found a match with a larger subgraph pattern
            continue;
        }

        int matches = num_subgraph_pattern_matches(subgraph_pattern, graph, num_expected_matches);
        if (matches == num_expected_matches) {
            log_debug(LogPatternMatcher, "\tstart_vertex: {}, end_vertex: {}: Found exactly {} matches.", start_vertex_id, end_vertex_id, num_expected_matches);
            max_num_subgraph_vertices = num_subgraph_vertices;
            is_subgraph_match_found = true;
            best_subgraph_pattern = subgraph_pattern;
        }
    }
    return best_subgraph_pattern;
}

SubgraphPatternMatchMappings subgraph_pattern_match(graph_type& subgraph, graph_type& graph)
{
    int num_subgraph_vertices = boost::num_vertices(subgraph);
    log_info(LogPatternMatcher, "SubgraphPatternMatch found subgraph of {} nodes.", num_subgraph_vertices);

    SubgraphPatternMatchMappings subgraph_match_mappings;

    int total_matches = 0;
    auto callback = [&](auto bijection, auto) {
        SubgraphPatternMatch subgraph_matches;
        subgraph_matches.reserve(num_subgraph_vertices);

        for (auto v : boost::make_iterator_range(vertices(subgraph))) {
            subgraph_matches[subgraph[v].node_id] = graph[get(bijection, v)].node_id;
        }
        subgraph_match_mappings.emplace_back(std::move(subgraph_matches));
        total_matches += 1;
        return true;
    };

    auto edge_predicate = [&](auto edge_a, auto edge_b) {
        return subgraph[edge_a].producer_output_edge_index == graph[edge_b].producer_output_edge_index
            and subgraph[edge_a].consumer_input_edge_index == graph[edge_b].consumer_input_edge_index;
    };
    auto vertex_predicate = [&](auto vertex_a, auto vertex_b) {
        if (subgraph[vertex_a].op_type == "*") {
            return true;
        }
        return subgraph[vertex_a].op_type == graph[vertex_b].op_type;
    };

    boost::vf2_subgraph_iso(
        subgraph,
        graph,
        callback,
        boost::vertex_order_by_mult(subgraph),
        boost::edges_equivalent(edge_predicate).vertices_equivalent(vertex_predicate));

    log_info(LogPatternMatcher, "SubgraphPatternMatch Finished: Recorded {} matches.", total_matches);

    return subgraph_match_mappings;
}

} // namespace pattern_matcher
