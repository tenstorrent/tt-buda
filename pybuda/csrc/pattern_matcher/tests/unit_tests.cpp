// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "gtest/gtest.h"
#include <iostream>
#include <fstream>
#include <experimental/filesystem> // clang6 requires us to use "experimental", g++ 9.3 is fine with just <filesystem>

#include "pattern_matcher/pattern_matcher.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/vf2_sub_graph_iso.hpp>

#include <boost/graph/adj_list_serialize.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/graph/graphviz.hpp>

using std::string;
using namespace tt::graphlib;
using namespace pattern_matcher;


typedef boost::adjacency_list< boost::setS, boost::vecS, boost::bidirectionalS, VertexProperty, EdgeProperty> graph_type;
typedef boost::subgraph<graph_type> subgraph_type;

typedef boost::graph_traits<graph_type>::vertex_descriptor VertexId;
typedef boost::graph_traits<graph_type>::edge_descriptor EdgeId;


graph_type load_mha_matmul_pattern() {
    graph_type pattern_graph;
    auto input = add_vertex(
        VertexProperty{
            .name="input",
            .op_type="*", // used as wildcard
            .node_id=7,
        },
        pattern_graph
    );

    auto reshape= add_vertex(
        VertexProperty{
            .name="reshape_1",
            .op_type="reshape",
            .node_id=0,
        },
        pattern_graph
    );
    auto mm0= add_vertex(
        VertexProperty{
            .name="matmul0",
            .op_type="matmul",
            .node_id=1,
        },
        pattern_graph
    );
    auto mm1= add_vertex(
        VertexProperty{
            .name="matmul1",
            .op_type="matmul",
            .node_id=2,
        },
        pattern_graph
    );
    auto mm2= add_vertex(
        VertexProperty{
            .name="matmul2",
            .op_type="matmul",
            .node_id=3,
        },
        pattern_graph
    );

    add_edge(input, reshape, EdgeProperty{.producer_output_edge_index = 0 }, pattern_graph);
    add_edge(reshape, mm0, EdgeProperty{.producer_output_edge_index = 0 }, pattern_graph);
    add_edge(reshape, mm1, EdgeProperty{.producer_output_edge_index = 1 }, pattern_graph);
    add_edge(reshape, mm2, EdgeProperty{.producer_output_edge_index = 2 }, pattern_graph);

    return pattern_graph;
}


TEST(PatternMatcher, encoders_2)
{
    auto pattern = load_mha_matmul_pattern();
    std::string graph_file_path =
        std::experimental::filesystem::path(__FILE__).parent_path().string() +
        "/boost_test_graphs/2encoder_boost_graph.txt";
    auto graph = load_graph_from_file(graph_file_path);

    int total_matches = num_subgraph_pattern_matches(pattern, graph, 2);
    EXPECT_TRUE(total_matches == 2);
}

TEST(PatternMatcher, encoders_2_discovery)
{
    std::string graph_file_path =
        std::experimental::filesystem::path(__FILE__).parent_path().string() +
        "/boost_test_graphs/2encoder_boost_graph.txt";
    auto large_graph = load_graph_from_file(graph_file_path);
    bool pass = contains_exactly_n_subgraph_matches(large_graph, 2);
    EXPECT_TRUE(pass);
}

TEST(PatternMatcher, encoders_12)
{
    auto pattern = load_mha_matmul_pattern();
    std::string graph_file_path =
        std::experimental::filesystem::path(__FILE__).parent_path().string() +
        "/boost_test_graphs/12encoder_boost_graph.txt";
    auto graph = load_graph_from_file(graph_file_path);

    int total_matches = num_subgraph_pattern_matches(pattern, graph, 12);
    EXPECT_TRUE(total_matches == 12);
}

TEST(PatternMatcher, encoders_12_discovery)
{
    std::string graph_file_path =
        std::experimental::filesystem::path(__FILE__).parent_path().string() +
        "/boost_test_graphs/12encoder_boost_graph.txt";
    auto large_graph = load_graph_from_file(graph_file_path);

    bool pass = contains_exactly_n_subgraph_matches(large_graph, 12);
    EXPECT_TRUE(pass);
}

