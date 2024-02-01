// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "graph_lib/defines.hpp"

#include <string>
#include <iostream>
#include <vector>
#include <unordered_map>

#include <boost/graph/graph_utility.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adj_list_serialize.hpp>

using tt::graphlib::NodeId;

namespace pattern_matcher {

/*
  ____        _          ____  _                   _
 |  _ \  __ _| |_ __ _  / ___|| |_ _ __ _   _  ___| |_ _   _ _ __ ___  ___
 | | | |/ _` | __/ _` | \___ \| __| '__| | | |/ __| __| | | | '__/ _ \/ __|
 | |_| | (_| | || (_| |  ___) | |_| |  | |_| | (__| |_| |_| | | |  __/\__ \
 |____/ \__,_|\__\__,_| |____/ \__|_|   \__,_|\___|\__|\__,_|_|  \___||___/

*/

struct VertexProperty {
    std::string name;
    std::string op_type;
    NodeId node_id;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        (void)version;
        ar & name;
        ar & op_type;
        ar & node_id;
    }
};

struct EdgeProperty {

    // tag the producer_output_edge_index so that we don't get permuted user edge mappings
    int producer_output_edge_index;
    int consumer_input_edge_index = 0;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        (void)version;
        ar & producer_output_edge_index;
    }
};

template <class Name>
class VertexPropertyWriter {
public:
     VertexPropertyWriter(Name _name) : name(_name) {}
     template <class VertexOrEdge>
     void operator()(std::ostream& out, const VertexOrEdge& v) const
     {
        out << "[label=\"" << name[v].name << "\nid:" << std::to_string(v) << "\"]";
     }
private:
     Name name;
};

typedef boost::adjacency_list< boost::setS, boost::vecS, boost::bidirectionalS, VertexProperty, EdgeProperty> graph_type;
typedef boost::graph_traits<graph_type>::vertex_descriptor VertexId;
typedef boost::graph_traits<graph_type>::edge_descriptor EdgeId;


// keys := set of nodes corresponding to subgraph pattern
// values := ordered list of matches
using SubgraphPatternMatch = std::unordered_map<NodeId, NodeId>;
using SubgraphPatternMatchMappings = std::vector<std::unordered_map<NodeId, NodeId>>;



/*
    _    ____ ___
   / \  |  _ \_ _|___
  / _ \ | |_) | |/ __|
 / ___ \|  __/| |\__ \
/_/   \_\_|  |___|___/
*/


// Utility Methods
void save_dotgraph_to_ostream(std::ostream& stream, const graph_type& graph);
void save_dotgraph_to_file(std::string filename, const graph_type& graph);
void save_graph_to_file(std::string filename, graph_type& graph);
graph_type load_graph_from_file(std::string filename);

// Helper query methods
int num_subgraph_pattern_matches(graph_type& subgraph, graph_type& graph, int num_matches);
bool contains_exactly_n_subgraph_matches(graph_type& graph, int num_matches);

//
// Main Subgraph Pattern Matcher APIs
//
std::vector<NodeId> get_node_ids(graph_type& graph, const SubgraphPatternMatchMappings& subgraph_matches, int match_index = 0);
std::vector<NodeId> get_input_activation_node_ids(graph_type& graph, const SubgraphPatternMatchMappings& subgraph_matches, int match_index = 0);
std::vector<NodeId> get_parameter_node_ids(graph_type& graph, const SubgraphPatternMatchMappings& subgraph_matches, int match_index = 0);
std::vector<NodeId> get_constant_node_ids(graph_type& graph, const SubgraphPatternMatchMappings& subgraph_matches, int match_index = 0);
std::vector<NodeId> get_output_node_ids(graph_type& graph, const SubgraphPatternMatchMappings& subgraph_matches, int match_index = 0);
std::vector<NodeId> get_unmatched_node_ids(graph_type& graph, const SubgraphPatternMatchMappings& subgraph_matches);

// Given a graph, return the largest discovered subgraph which yields exactly
// `num_expected_matches` instances in the input `graph`.
// If subgraph was not discovered, return empty graph.
graph_type discover_largest_subgraph_pattern(graph_type& graph, int num_expected_matches);

// Given a subgraph and a graph, return a BUDA-graph NodeId mapping between
// the subgraph and all instances in the graph.
SubgraphPatternMatchMappings subgraph_pattern_match(graph_type& subgraph, graph_type& graph);

} // namespace pattern_matcher
