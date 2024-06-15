// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/lower_concat_to_runtime_transform.hpp"


#include "graph_lib/utils.hpp"
#include "graph_lib/node_types.hpp"
#include "passes/passes_utils.hpp"

namespace tt::passes
{


void lower_concat_to_runtime_transform(graphlib::Graph *graph)
{
    bool concat_on_host = env_as<bool>("PYBUDA_CONCAT_ON_HOST");
    if (not concat_on_host)
        return;

    int concat_group = 0;

    for (graphlib::Node *output_node: graph->nodes_by_type(graphlib::NodeType::kOutput))
    {
        // Skip partial data copy edges (past-cache link between producers/consumers)
        std::vector<graphlib::Edge> partial_datacopy_edges = graph->user_partial_datacopy_edges(output_node);
        if (not partial_datacopy_edges.empty())
            continue;
        
        std::vector<graphlib::Node *> producers = graph->data_operands(output_node);
        TT_ASSERT(producers.size() == 1);

        if (not producers[0]->as<graphlib::OpNode>() or producers[0]->as<graphlib::OpNode>()->op_name() != "concatenate")
            continue;

        std::vector<graphlib::OutputNode *> concat_outputs;
        graphlib::Node *concat = producers[0];
        int dim = std::get<int>(concat->as<graphlib::OpNode>()->op_attrs()[0]);
        // The first producer will reuse the output, the remaining will need their own outputs
        std::vector<graphlib::Node *> concat_producers = graph->data_operands(concat);
        if (concat->as<graphlib::TaggedNode>()->has_tag("fracture_bottom"))
        {
            for (graphlib::Node *producer : concat_producers)
            {
                producer->as<graphlib::TaggedNode>()->tag("fracture_bottom", true);
                producer->as<graphlib::TaggedNode>()->tag("dont_remove", true);
            }
        }

        graph->remove_node(concat);

        graph->add_edge(graphlib::Edge(concat_producers[0]->id(), 0, output_node->id(), 0, graphlib::EdgeType::kData));
        output_node->set_shape(concat_producers[0]->shape());

        concat_outputs.push_back(output_node->as<graphlib::OutputNode>());
        unsigned int subgraph_index = graph->get_subgraph_id_for_node(output_node->id());
        for (size_t i = 1; i < concat_producers.size(); i++)
        {
            graphlib::Node *producer = concat_producers[i];
            auto output = graph->add_node(graphlib::create_node<graphlib::OutputNode>(producer->name() + "_" + std::to_string(i)), subgraph_index);
            graphlib::Edge output_edge(producer->id(), (graphlib::PortId)0, output->id(), (graphlib::PortId)0, graphlib::EdgeType::kData);
            graph->add_edge(output_edge);
            graph->add_module_output(output->id());
            concat_outputs.push_back(output->as<graphlib::OutputNode>());
            output->set_shape(producer->shape());
        }
        for(size_t index = 0; index < concat_outputs.size(); index++)
        {
            graphlib::RuntimeTensorTransform transform = graphlib::RuntimeTensorTransform ::ConcatenateOnHost(concat_group, index, dim);
            concat_outputs[index]->set_runtime_tensor_transform(transform);
        }
        concat_group++;
    }


    return;
}
} // namespace tt::passes
