// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "test/common.hpp"
#include "passes/pre_placer_buda_passes.hpp"

namespace tt::test
{

struct PastCache : testing::Test
{
    graphlib::Graph *graph;

    PastCache()
    {
        // Initialize graph
        graph = new graphlib::Graph(graphlib::IRLevel::IR_PYBUDA);

        // Graph definition
        auto pcache = create_input(*graph, "pcache", graphlib::Shape::create({1, 1, 416, 384}), graphlib::InputNodeType::Parameter); 
        auto input = create_input(*graph, "input", graphlib::Shape::create({1, 6, 32, 64}));
        
        auto slice0 = add_node<graphlib::PyOpNode>(*graph, "slice0", "hslice", {6}, {pcache});
        auto concat1 = add_node<graphlib::PyOpNode>(*graph, "concat1", "concatenate", {-2}, {slice0, input});
        auto stack2 = add_node<graphlib::PyOpNode>(*graph, "stack2", "hstack", {6}, {concat1});

        auto output = create_output(*graph, "output_0", stack2);

        graphlib::Edge edge(output->id(), (graphlib::PortId)0, pcache->id(), (graphlib::PortId)0, graphlib::EdgeType::kPartialDataCopy);
        graph->add_edge(edge);

        // Set ublock order for pcache producer #1
        auto producer_edge = graph->get_edges(stack2, output)[0];
        graph->get_edge_attributes(producer_edge)->set_ublock_order(graphlib::UBlockOrder::R);


        // Second producer for past cache
        auto slice0_ = add_node<graphlib::PyOpNode>(*graph, "slice0_", "hslice", {6}, {pcache});
        auto concat1_ = add_node<graphlib::PyOpNode>(*graph, "concat1_", "concatenate", {-2}, {slice0_, input});
        auto stack2_ = add_node<graphlib::PyOpNode>(*graph, "stack2_", "hstack", {6}, {concat1_});

        auto output_ = create_output(*graph, "output_0_", stack2_);

        graphlib::Edge edge2(output_->id(), (graphlib::PortId)0, pcache->id(), (graphlib::PortId)0, graphlib::EdgeType::kPartialDataCopy);
        graph->add_edge(edge2);

        // Set ublock order for pcache producer #2
        auto producer_edge2 = graph->get_edges(stack2_, output_)[0];
        graph->get_edge_attributes(producer_edge2)->set_ublock_order(graphlib::UBlockOrder::C);
    }
};


bool check_ublock_order(graphlib::Graph *graph) {
    bool ublock_order_matches = true;
    for (Node * node : graph->nodes())
    {
        std::vector<graphlib::Edge> partial_datacopy_edges = graph->operand_partial_datacopy_edges(node);

        if (partial_datacopy_edges.empty())
            continue;
        
        std::vector<graphlib::Edge> producer_edges;
        for (auto edge : partial_datacopy_edges) {
            auto output_node = graph->node_by_id(edge.producer_node_id);
            TT_ASSERT(graph->operand_edges(output_node).size() == 1, "Output node should only have 1 producer");
            producer_edges.push_back(
                graph->operand_edges(output_node).front()
            );
        }

        // Assert all partial datacopy producer edges have the same ublock order
        ublock_order_matches &=
            std::all_of(
                producer_edges.begin(),
                producer_edges.end(),
                [graph, producer_edges](Edge e)
                { return graph->get_edge_attributes(e)->get_ublock_order() == graph->get_edge_attributes(producer_edges[0])->get_ublock_order(); }
            );

    }
    return ublock_order_matches;

}


TEST_F(PastCache, PastCacheUblockOrder)
{
    // Calculate ublock order
    calculate_ublock_order(graph);

    EXPECT_EQ(check_ublock_order(graph), true);
}

}  // namespace tt::test
