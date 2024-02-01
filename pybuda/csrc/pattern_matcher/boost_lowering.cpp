// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "pattern_matcher/boost_lowering.hpp"
#include "pattern_matcher/pattern_matcher.hpp"

#include <sstream>

#include "utils/logger.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"


#include <unordered_map>

// keys := set of nodes corresponding to subgraph pattern
// values := ordered list of matches
// TODO(jchu): verify the matches are presented in order.
using NodeIdToNodeIdMatches = std::unordered_map<NodeId, std::vector<NodeId>>;
using tt::LogPatternMatcher;

using namespace tt::graphlib;

namespace pattern_matcher {

std::string get_input_op_string(Node* node) {
    if (node->as<InputNode>()->is_activation()) {
        return "*";
    }
    return node->as<InputNode>()->input_type_string();
}

std::string get_op_string(Node* node) {
    if (node->node_type() == NodeType::kInput) {
        return get_input_op_string(node);
    } else if (node->node_type() == NodeType::kOutput) {
        return "output";
    }
    return node->as<OpNode>()->op_name();
}

VertexId add_vertex_to_boost_graph(graph_type& graph, Node* node) {

    // log_trace(LogPatternMatcher, "Node: {}, {}, {}", node->id(), get_op_string(node), node->name());
    VertexId vertex_descriptor = add_vertex(
        VertexProperty{
            .name=node->name(),
            .op_type=get_op_string(node),
            .node_id=node->id(),
        },
        graph
    );
    return vertex_descriptor;
}

graph_type convert_graph_to_boost_graph(Graph* graph) {
    graph_type boost_graph;
    std::unordered_map<NodeId, VertexId> node_to_vertex;

    std::vector<Node*> deferred_nodes;

    // add the activation inputs first
    for (Node* node : topological_sort(*graph)) {
        if (node->node_type() == NodeType::kInput and node->as<InputNode>()->is_activation()) {
            node_to_vertex[node->id()] = add_vertex_to_boost_graph(boost_graph, node);
        }
    }

    // copy nodes and record node mapping
    for (Node* node : topological_sort(*graph)) {
        for (Node* operand : graph->data_operands(node)) {
            if (node_to_vertex.find(operand->id()) == node_to_vertex.end()) {
                node_to_vertex[operand->id()] = add_vertex_to_boost_graph(boost_graph, operand);
            }
        }

        if (node->node_type() == NodeType::kPyOp or node->node_type() == NodeType::kBudaOp) {
            node_to_vertex[node->id()] = add_vertex_to_boost_graph(boost_graph, node);
        } else {
            deferred_nodes.push_back(node);
        }
    }

    for (Node* node : deferred_nodes) {
        if (node_to_vertex.find(node->id()) == node_to_vertex.end()) {
            node_to_vertex[node->id()] = add_vertex_to_boost_graph(boost_graph, node);
        }
    }

    // copy nodes and record node mapping
    for (const auto& [node_id, edge_set]: graph->operands_map()) {
        for (const Edge& edge : edge_set) {
            if (edge.edge_type == EdgeType::kData)  {
                // going to assume all of these are just on single output port
                auto [output_port_id, producer_output_edge_index] = graph->output_port_and_index_for_data_user_port(graph->node_by_id(edge.producer_node_id), edge);
                // log_trace(LogPatternMatcher, "Edge: {}, {}, {}", edge.producer_node_id, edge.consumer_node_id, producer_output_edge_index);
                add_edge(
                    node_to_vertex[edge.producer_node_id],
                    node_to_vertex[edge.consumer_node_id],
                    EdgeProperty{
                        .producer_output_edge_index=producer_output_edge_index,
                        .consumer_input_edge_index=(int)edge.consumer_input_port_id
                    },
                    boost_graph
                );
            }
        }
    }

    for (const auto& [node_id, edge_set]: graph->users_map()) {
        for (const Edge& edge : edge_set) {
            if (edge.edge_type == EdgeType::kData)  {
                // going to assume all of these are just on single output port
                auto [output_port_id, producer_output_edge_index] = graph->output_port_and_index_for_data_user_port(graph->node_by_id(edge.producer_node_id), edge);
                // log_trace(LogPatternMatcher, "Edge: {}, {}, {}", edge.producer_node_id, edge.consumer_node_id, producer_output_edge_index);
                add_edge(
                    node_to_vertex[edge.producer_node_id],
                    node_to_vertex[edge.consumer_node_id],
                    EdgeProperty{
                        .producer_output_edge_index=producer_output_edge_index,
                        .consumer_input_edge_index=(int)edge.consumer_input_port_id
                    },
                    boost_graph
                );
            }
        }
    }

    return boost_graph;
}

VertexId add_vertex_to_boost_graph(graph_type& graph, json node) {

    uint node_id = node["nid"];
    log_debug(LogPatternMatcher, "Node: {}, {}, {}", node_id, node["op"], node["buda_name"]);
    VertexId vertex_descriptor = add_vertex(
        VertexProperty{
            .name=node["buda_name"],
            .op_type=node["op"],
            .node_id=node_id,
        },
        graph
    );
    return vertex_descriptor;
}


graph_type convert_json_graph_to_boost_graph(json json_graph) {
    graph_type boost_graph;
    std::unordered_map<int, VertexId> node_to_vertex;
    std::vector<uint> deferred_node_ids; 

    // process input nodes first
    auto nodes = json_graph["nodes"];
    for (auto node : nodes) {
        if (node.contains("op") and node["op"] == "*") {
            node_to_vertex[node["nid"]] = add_vertex_to_boost_graph(boost_graph, node);
        }
    }

    // we want to defer adding constants until they are needed in the graph
    for (auto node : nodes) {
        if (node["attrs"].contains("num_inputs")) {
            std::string num_inputs_str = node["attrs"]["num_inputs"];
            uint num_inputs = std::stoi(num_inputs_str);
            for (uint input_index = 0; input_index < num_inputs; input_index++) {
                uint input_nid = node["inputs"][input_index][0];
                if (node_to_vertex.find(input_nid) == node_to_vertex.end()) {
                    node_to_vertex[input_nid] = add_vertex_to_boost_graph(boost_graph, json_graph["nodes"][input_nid]);
                }
            }

            node_to_vertex[node["nid"]] = add_vertex_to_boost_graph(boost_graph, node);
        }
        else {
            deferred_node_ids.push_back(node["nid"]);
        }
    }
    for (uint nid : deferred_node_ids) {
        if (node_to_vertex.find(nid) == node_to_vertex.end()) {
            std::cout << "Shouldn't be the case!!!!!!!" << std::endl;
            node_to_vertex[nid] = add_vertex_to_boost_graph(boost_graph, json_graph["nodes"][nid]);
        }
    }

    // construct users map:
    std::unordered_map<VertexId, std::vector<VertexId>> users_map;
    std::map<std::pair<VertexId, VertexId>, int> producer_output_index_map;
    std::map<std::pair<VertexId, VertexId>, int> consumer_input_index_map;

    for (auto node: nodes) {
        if (node["attrs"].contains("num_inputs")) {
            std::string num_inputs_str = node["attrs"]["num_inputs"];
            uint num_inputs = std::stoi(num_inputs_str);
            for (uint input_index = 0; input_index < num_inputs; input_index++) {
                uint input_nid = node["inputs"][input_index][0];
                uint node_id = node["nid"];

                int producer_output_index = users_map[input_nid].size();
                users_map[input_nid].push_back(node_id);
                std::pair<VertexId, VertexId> key = {input_nid, node_id};

                producer_output_index_map[key] = producer_output_index;
                consumer_input_index_map[key] = input_index;
            }
        }
    }

    for (auto node: nodes) {
        if (node["attrs"].contains("num_inputs")) {
            std::string num_inputs_str = node["attrs"]["num_inputs"];
            uint num_inputs = std::stoi(num_inputs_str);
            for (uint input_index = 0; input_index < num_inputs; input_index++) {
                uint input_nid = node["inputs"][input_index][0];
                uint node_id = node["nid"];
                // log_trace(LogPatternMatcher, "Edge: {}, {}, {}", input_nid, node_id, producer_output_edge_index);

                std::pair<VertexId, VertexId> key = {input_nid, node_id};
                int producer_output_index = producer_output_index_map[key];
                int consumer_input_index = consumer_input_index_map[key];

                add_edge(
                    node_to_vertex[input_nid],
                    node_to_vertex[node_id],
                    EdgeProperty{.producer_output_edge_index=producer_output_index, .consumer_input_edge_index=consumer_input_index },
                    boost_graph
                );
            }
        }
    }

    return boost_graph;
}


void print_nodes(json graph, std::string type, const std::vector<NodeId>& node_ids) {
    std::stringstream ss;
    for (auto node_id : node_ids) {
        ss << graph["nodes"][node_id]["buda_name"] << ", ";
    }
    log_debug(LogPatternMatcher, "{}: [{}]", type, ss.str());
}

void print_subgraph_pattern_matches(json graph, graph_type& boost_braph, graph_type& subgraph, const SubgraphPatternMatchMappings& subgraph_matches) {
    for (NodeId node_id : get_node_ids(subgraph, subgraph_matches)) {
        std::stringstream ss;
        for (size_t match_idx = 0; match_idx < subgraph_matches.size(); ++match_idx) {
            NodeId matched_node_id = subgraph_matches[match_idx].at(node_id);
            ss << graph["nodes"][matched_node_id]["buda_name"] << ", ";
        }
        log_debug(LogPatternMatcher, "{} -> [{}]", graph["nodes"][node_id]["buda_name"], ss.str());
    }

    std::vector<NodeId> input_activation_node_ids = get_input_activation_node_ids(subgraph, subgraph_matches);
    std::vector<NodeId> parameter_node_ids = get_parameter_node_ids(subgraph, subgraph_matches);
    std::vector<NodeId> constant_node_ids = get_constant_node_ids(subgraph, subgraph_matches);
    std::vector<NodeId> output_node_ids = get_output_node_ids(subgraph, subgraph_matches);

    print_nodes(graph, "input", input_activation_node_ids );
    print_nodes(graph, "parameters", parameter_node_ids);
    print_nodes(graph, "constant", constant_node_ids);
    print_nodes(graph, "output", output_node_ids);

    std::vector<NodeId> unmatched_node_ids = get_unmatched_node_ids(boost_braph, subgraph_matches);
    print_nodes(graph, "unmatched", unmatched_node_ids);
}


void print_nodes(Graph* graph, std::string type, const std::vector<NodeId>& node_ids) {
    std::stringstream ss;
    for (auto node_id : node_ids) {
        ss << graph->node_by_id(node_id)->name() << ", ";
    }
    log_debug(LogPatternMatcher, "{}: [{}]", type, ss.str());
}

void print_subgraph_pattern_matches(Graph* graph, graph_type& boost_graph, graph_type& subgraph, const SubgraphPatternMatchMappings& subgraph_matches) {
    for (NodeId node_id : get_node_ids(subgraph, subgraph_matches)) {
        std::stringstream ss;
        for (size_t match_idx = 0; match_idx < subgraph_matches.size(); ++match_idx) {
            NodeId matched_node_id = subgraph_matches[match_idx].at(node_id);
            ss << (graph->node_by_id(matched_node_id))->name() << ", ";
        }
        log_debug(LogPatternMatcher, "{} -> [{}]", graph->node_by_id(node_id)->name(), ss.str());
    }

    std::vector<NodeId> input_activation_node_ids = get_input_activation_node_ids(subgraph, subgraph_matches);
    std::vector<NodeId> parameter_node_ids = get_parameter_node_ids(subgraph, subgraph_matches);
    std::vector<NodeId> constant_node_ids = get_constant_node_ids(subgraph, subgraph_matches);
    std::vector<NodeId> output_node_ids = get_output_node_ids(subgraph, subgraph_matches);

    log_debug(LogPatternMatcher, "=== Printing Results ===");
    print_nodes(graph, "input", input_activation_node_ids );
    print_nodes(graph, "parameters", parameter_node_ids);
    print_nodes(graph, "constant", constant_node_ids);
    print_nodes(graph, "output", output_node_ids);

    std::vector<NodeId> unmatched_node_ids = get_unmatched_node_ids(boost_graph, subgraph_matches);
    print_nodes(graph, "unmatched", unmatched_node_ids);
}


bool can_subgraph_be_looped(Graph *graph, graph_type& subgraph, const SubgraphPatternMatchMappings& subgraph_matches) {

    // Condition that must be satisfied for subgraph to be looped:
    // - from the discovered subgraph, we visit its output nodes (i.e. leaf nodes). Each output node
    //   must belong in the input_activation_node_ids of the next match instance.
    // - We are required to do a pairwise check for each discovered subgraph pattern to guarantee
    //   that we can continue the loop iteration.
    size_t num_pairwise_checks = subgraph_matches.size() - 1;

    for (size_t match_idx = 0; match_idx < num_pairwise_checks; ++match_idx) {
        // not expecting a lot of input ids.. leave it as a vector instead of converting to a set
        std::vector<NodeId> current_match_output_node_ids = get_output_node_ids(subgraph, subgraph_matches, match_idx);
        std::vector<NodeId> next_match_input_node_ids = get_input_activation_node_ids(subgraph, subgraph_matches, match_idx + 1);

        bool can_continue_looping = true;
        for (NodeId node_id : next_match_input_node_ids) {
            // all operands to the input nodes of the next match should belong in the output node ids of current match
            Node* input_node = graph->node_by_id(node_id);
            for (Node* operand : graph->data_operands(input_node)) {
                // all operands belong to the output-set of current-match
                if (operand->node_type() == NodeType::kInput) {
                    continue;
                }
                can_continue_looping &= std::find(current_match_output_node_ids.begin(), current_match_output_node_ids.end(), operand->id()) !=
                    current_match_output_node_ids.end();
            }
        }
        if (not can_continue_looping) {
            log_debug(LogPatternMatcher, "current match_idx: {}, output_nodes:", match_idx);
            for (NodeId output_id: current_match_output_node_ids ) {
                log_debug(LogPatternMatcher, "\t\t {}", graph->node_by_id(output_id)->name());
            }

            log_debug(LogPatternMatcher, "\t next_match_idx input_ids:" );
            for (NodeId input_id : next_match_input_node_ids) {
                log_debug(LogPatternMatcher, "\t\t {}", graph->node_by_id(input_id)->name());
            }
            return false;
        }
    }

    return true;
}

void loop_over_subgraph(Graph *graph, graph_type& subgraph, const SubgraphPatternMatchMappings& subgraph_matches) {
    std::vector<NodeId> input_activation_node_ids = get_input_activation_node_ids(subgraph, subgraph_matches);
    std::vector<NodeId> output_node_ids = get_output_node_ids(subgraph, subgraph_matches);

    // For now only handle single output
    TT_ASSERT(input_activation_node_ids.size() == 1, "PatternMatcher only supports single output for now.");
    TT_ASSERT(output_node_ids.size() == 1, "PatternMatcher only supports single output for now.");

    std::vector<NodeId> final_match_output_node_ids = get_output_node_ids(subgraph, subgraph_matches, subgraph_matches.size() - 1);
    TT_ASSERT(final_match_output_node_ids.size() == 1, "PatternMatcher only supports single output for now.");
    Node* final_match_output_node = graph->node_by_id(final_match_output_node_ids.at(0));

    int index = 0;
    std::vector<Node*> outputs_of_final_subgraph_match = graph->data_users(final_match_output_node);

    // connect output node to outputs_of_final_subgraph_match
    Node* input_activation_node = graph->node_by_id(input_activation_node_ids.at(index));
    Node* input = graph->data_operands(input_activation_node).at(0);
    Node* primary_output = graph->node_by_id(output_node_ids.at(index));


    int loop_iterations = subgraph_matches.size();
    std::unordered_map<std::string, std::vector<std::string>> parameter_to_matched_parameters;
    std::unordered_set<NodeId> nodes_processed_in_loop;

    for (int match_idx = 0; match_idx < loop_iterations; ++match_idx) {
        for (const auto& [primary_node_id, matched_node_id] : subgraph_matches[match_idx]) {
            Node* primary = graph->node_by_id(primary_node_id);
            Node* matched = graph->node_by_id(matched_node_id);

            if (primary->node_type() == NodeType::kInput and primary->as<InputNode>()->is_parameter()) {
                parameter_to_matched_parameters[primary->name()].push_back(matched->name());
            }

            if (match_idx == 0) {
                nodes_processed_in_loop.insert(primary->id());

            } else {
                // skip match_idx = 0 because we only want to delete matches [1, n)
                if (subgraph_matches[match_idx].find(matched_node_id) == subgraph_matches[match_idx].end()) {
                    // make sure output of primaries aren't deleted
                    graph->remove_node(matched_node_id);
                } else {
                    for (auto edge : graph->user_edges(matched)) {
                        graph->remove_edge(edge);
                    }
                }
            }
        }
    }

    for (const auto& [parameter, mapped_parameters] : parameter_to_matched_parameters) {
        std::stringstream ss;
        for (const auto& mapped_parameter : mapped_parameters) {
            ss << mapped_parameter << ", ";
        }
        log_info(LogPatternMatcher, "Recording Parameter Mapping: {}->[{}]", parameter, ss.str());
    }

    for (auto output : outputs_of_final_subgraph_match) {
        log_info(LogPatternMatcher, "Final subgraph match output users: {}", output->name());
        graph->add_edge(primary_output, output);
    }

    Edge control_loop_edge(primary_output->id(), 0, input->id(), 0 /* consumer_input_port_id */, EdgeType::kControlLoop);
    std::shared_ptr<LoopEdgeAttributes> loop_attributes = std::make_shared<LoopEdgeAttributes>(
            EdgeType::kControlLoop,
            LoopEdgeAttributes::LoopEdgeAttributesInternal{
                .loop_iterations_ = loop_iterations,
                .parameter_to_matched_parameters_ = parameter_to_matched_parameters,
                .nodes_processed_in_loop_= nodes_processed_in_loop
            });
    graph->add_edge(control_loop_edge, loop_attributes);
}

std::pair<Graph*, MatchResult> lower_pybuda_to_pattern_matcher(Graph* graph, int num_matches_to_search) {
    graph_type boost_graph = convert_graph_to_boost_graph(graph);
    graph_type subgraph_pattern = discover_largest_subgraph_pattern(boost_graph, num_matches_to_search);

    // save_dotgraph_to_ostream(std::cout, boost_graph);

    bool is_subgraph_pattern_found = boost::num_vertices(subgraph_pattern) > 0;
    bool is_subgraph_loopable = false;

    SubgraphPatternMatchMappings subgraph_matches = subgraph_pattern_match(subgraph_pattern, boost_graph);
    if (is_subgraph_pattern_found) {
        print_subgraph_pattern_matches(graph, boost_graph, subgraph_pattern, subgraph_matches);

        is_subgraph_loopable = can_subgraph_be_looped(graph, subgraph_pattern, subgraph_matches);
        if (is_subgraph_loopable) {
            loop_over_subgraph(graph, subgraph_pattern, subgraph_matches);
        }
    }


    log_info(LogPatternMatcher, "Subgraph pattern is found: {}", (is_subgraph_pattern_found ? "YES" : "NO"));
    log_info(LogPatternMatcher, "Subgraph can be looped: {}", (is_subgraph_loopable  ? "YES" : "NO"));

    return {
        graph,
        MatchResult{
            .is_subgraph_pattern_found=is_subgraph_pattern_found,
            .is_subgraph_loopable=is_subgraph_loopable,
            .subgraph_matches=subgraph_matches
        }
    };
}

MatchResult lower_json_to_pattern_matcher(json graph, int num_matches_to_search) {
    graph_type boost_graph = convert_json_graph_to_boost_graph(graph);
    graph_type subgraph_pattern = discover_largest_subgraph_pattern(boost_graph, num_matches_to_search);

    //save_dotgraph_to_ostream(std::cout, boost_graph);
    //save_dotgraph_to_file("input_graph.txt", boost_graph);

    bool is_subgraph_pattern_found = boost::num_vertices(subgraph_pattern) > 0;

    SubgraphPatternMatchMappings subgraph_matches = subgraph_pattern_match(subgraph_pattern, boost_graph);
    print_subgraph_pattern_matches(graph, boost_graph, subgraph_pattern, subgraph_matches);

    log_info(LogPatternMatcher, "Subgraph pattern is found: {}", (is_subgraph_pattern_found ? "YES" : "NO"));

    return 
        MatchResult{
            .is_subgraph_pattern_found=is_subgraph_pattern_found,
            .is_subgraph_loopable=false,
            .subgraph_matches=subgraph_matches
        };
}

} // namespace pattern_matcher
