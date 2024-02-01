// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/link_past_cache_ios.hpp"


#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "passes/passes_utils.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{

bool shape_compatible(graphlib::OpNode *output_producer, graphlib::Node *input_consumer)
{
    auto p_shape = output_producer->shape().canonical();
    auto c_shape = input_consumer->shape().canonical();
    if (output_producer->op_type().op == "concatenate")
    {
        int dim = std::get<int>(output_producer->op_type().attr[0]);
        c_shape[dim] = p_shape[dim];
    }
    return (p_shape == c_shape);
}

std::unordered_map<graphlib::Node *, graphlib::Node*> link_cache_outputs_to_parameters(graphlib::Graph *graph)
{
    auto is_matmul = [](graphlib::Node *n)
    {
        return (n->node_type() == graphlib::kPyOp and n->as<graphlib::OpNode>()->is_matmul());
    };

    // Search through output nodes, if producer chain is param-><transpose>->MM-><reshape>-><add>->hslice-><concatonate>->output
    // with the <> nodes being optional, the output is linked with the parameter
    std::unordered_map<graphlib::Node *, graphlib::Node*> param_to_output;
    for (Node *output_node : graph->ordered_module_outputs())
    {
        std::vector<graphlib::Node *> producer_mms = reachable_nodes(graph, output_node, is_matmul, true);

        if (producer_mms.empty())
            continue;

        // producer MMs will be topologically sorted, so last one is the one we want
        graphlib::Node *producer_mm = producer_mms.back();

        // get the parameter, if the producer is a transpose, we need to go one more back
        graphlib::Node *operand = graph->data_operands(producer_mm)[1];
        if (operand->node_type() == graphlib::NodeType::kPyOp and operand->as<graphlib::OpNode>()->op_type().op == "transpose")
        {
            operand = graph->data_operands(operand)[0];
        }
        if (operand->node_type() == graphlib::NodeType::kInput and operand->as<graphlib::InputNode>()->input_type() == graphlib::InputNodeType::Parameter)
        {
            param_to_output[output_node] = operand;
            log_debug(LogGraphCompiler, "Linking present cache output {} to parameter {}", output_node->name(), operand->name());
        }
    }
    return param_to_output;
}


// create a separate program that will rotate the parameters one tile left
void rotate_params(graphlib::Graph *graph, std::vector<graphlib::Node *> params_to_rotate)
{
    unsigned int subgraph_index = graph->num_subgraphs();

    for (graphlib::Node *param : params_to_rotate)
    {
        graphlib::Shape shape = param->shape();
        int end = shape[-2];
        graphlib::OpType index_op_right("index", {-2, 32, end, 1});
        auto index_node_right = graph->add_node(
            graphlib::create_node<graphlib::PyOpNode>(param->name() + "_index_right", index_op_right),
            subgraph_index);
        graphlib::Edge read_edge_right(param->id(), (graphlib::PortId)0, index_node_right->id(), (graphlib::PortId)0, graphlib::EdgeType::kData);
        graph->add_edge(read_edge_right);
        calculate_and_set_node_shape(graph, index_node_right);

        graphlib::OpType index_op_left("index", {-2, 0, 32, 1});
        auto index_node_left = graph->add_node(
            graphlib::create_node<graphlib::PyOpNode>(param->name() + "_index_left", index_op_left),
            subgraph_index);
        graphlib::Edge read_edge_left(param->id(), (graphlib::PortId)0, index_node_left->id(), (graphlib::PortId)0, graphlib::EdgeType::kData);
        graph->add_edge(read_edge_left);
        calculate_and_set_node_shape(graph, index_node_left);

        graphlib::OpType concat_op("concatenate", {-2});
        auto concat_node = graph->add_node(
            graphlib::create_node<graphlib::PyOpNode>(param->name() + "_concat", concat_op),
            subgraph_index);
        graphlib::Edge concat_edge_a(index_node_right->id(), (graphlib::PortId)0, concat_node->id(), (graphlib::PortId)0, graphlib::EdgeType::kData);
        graph->add_edge(concat_edge_a);
        graphlib::Edge concat_edge_b(index_node_left->id(), (graphlib::PortId)0, concat_node->id(), (graphlib::PortId)1, graphlib::EdgeType::kData);
        graph->add_edge(concat_edge_b);
        calculate_and_set_node_shape(graph, concat_node);

        graphlib::OpType nop("nop");
        auto nop_node = graph->add_node(
            graphlib::create_node<graphlib::PyOpNode>(param->name() + "_nop", nop),
            subgraph_index);
        graphlib::Edge nop_edge(concat_node->id(), (graphlib::PortId)0, nop_node->id(), (graphlib::PortId)0, graphlib::EdgeType::kData);
        nop_node->as<graphlib::TaggedNode>()->tag("dont_remove", true);
        graph->add_edge(nop_edge);
        calculate_and_set_node_shape(graph, nop_node);
        

        auto output_node = graph->add_node(graphlib::create_node<graphlib::OutputNode>(param->name() + "_rotate_output"), subgraph_index);
        graphlib::Edge output_edge(nop_node->id(), (graphlib::PortId)0, output_node->id(), (graphlib::PortId)0, graphlib::EdgeType::kData);
        graph->add_edge(output_edge);
        graphlib::Edge write_edge(output_node->id(), (graphlib::PortId)0, param->id(), (graphlib::PortId)0, graphlib::EdgeType::kPartialDataCopy);
        graph->add_edge(write_edge);
    }
}

graphlib::Node* detect_inputs_to_convert(graphlib::Graph *graph, graphlib::Node* producer)
{
    graphlib::Node *input_node = nullptr;
    graphlib::OpNode *output_producer = producer->as<graphlib::OpNode>();
    if (not output_producer or not (is_eltwise(output_producer) or is_eltwise_nary(output_producer)))
        return input_node;

    auto is_input = [](graphlib::Node *n)
    {
        return (n->node_type() == graphlib::NodeType::kInput and n->as<graphlib::InputNode>()->input_type() == graphlib::InputNodeType::Activation);
    };

    // Locate input-node to be replaced with parameter node
    // 1. check the span of last-producer of the output node
    std::string output_producer_span = "";
    if (output_producer->as<graphlib::TaggedNode>()->has_tag("layer"))
        output_producer_span = output_producer->as<graphlib::TaggedNode>()->tag_value<std::string>("layer");

    // 2. get activation nodes in producer chain
    std::vector<graphlib::Node *> producer_input_nodes = reachable_nodes(graph, output_producer, is_input, true);
    if (producer_input_nodes.empty())
        return input_node;

    // 2.5 handle the case that output-producer is 'nop' first, since it does not have span tag
    // currently only accept input -> nop -> output pattern for pt1.x
    if (output_producer->op_type().op == "nop")
    {
        if (producer_input_nodes.size() == 1 and graph->data_users(producer_input_nodes[0])[0] == output_producer)
        {
            input_node = producer_input_nodes[0];
            return input_node;
        }
    }

    // 3. check if the span of any activation nodes match
    for (auto producer_input : producer_input_nodes)
    {
        std::vector<graphlib::Node*> input_consumers = graph->data_users(producer_input);
        std::string input_span_common = "";
        for (auto input_consumer : input_consumers)
        {
            if (input_consumer->as<graphlib::TaggedNode>()->has_tag("layer"))
            {
                std::string input_consumer_span = input_consumer->as<graphlib::TaggedNode>()->tag_value<std::string>("layer");
                if (input_span_common.empty())
                    input_span_common = input_consumer_span;

                // if different input-consumer has different span, the activation node is irrelevant to past-cache
                if (input_span_common != input_consumer_span)
                {
                    input_node = nullptr;
                    break;
                }

                // if span matches and node shape is compatible, the activation node maybe supposed to be linked by past-cache
                if (output_producer_span == input_consumer_span and shape_compatible(output_producer, input_consumer))
                     input_node = producer_input;
            }
        }

        if (input_node)
            break;
    }

    return input_node;
}

std::map<std::string, std::size_t> convert_inputs_to_params(graphlib::Graph *graph, std::unordered_map<graphlib::Node *, graphlib::Node*> param_to_output) 
{
    std::map<std::string, std::size_t> ret;
    std::vector<graphlib::Node *> ordered_inputs = graph->ordered_module_inputs();
    std::vector<graphlib::NodeId> inputs_to_remove;
    std::vector<graphlib::NodeId> outputs_to_remove;
    std::vector<graphlib::NodeId> nodes_to_remove;
    std::vector<graphlib::Node *> params_to_rotate;
    std::vector<graphlib::Node *> processed_outputs;
    std::unordered_map<graphlib::Node *, graphlib::Node *> inputs_to_param;

    for (graphlib::Node *output_node: graph->nodes_by_type(graphlib::NodeType::kOutput))
    {
        // if the input to the node is a concatonate of another input, or a mul/add, assume past cache update
        std::vector<graphlib::Node *> producers = graph->data_operands(output_node);
        TT_ASSERT(producers.size() == 1);
        graphlib::Node *input_node = detect_inputs_to_convert(graph, producers[0]);
        if (input_node == nullptr)
            continue;

        // handle slice-op in producer chain of concatenate op
        bool output_all = (producers[0]->as<graphlib::OpNode>()->op_type().op != "nop");
        bool is_concat = (producers[0]->as<graphlib::OpNode>()->op_type().op == "concatenate");
        int slice_factor = 1;
        std::vector<graphlib::Edge> edges;
        if (is_concat)
        {
            graphlib::Node *slice = nullptr;
            graphlib::Node *concat_operand = nullptr;

            std::vector<graphlib::Node *> concat_producers = graph->data_operands(producers[0]);
            if (concat_producers.size() != 2)
                continue;

            for (auto producer : concat_producers)
            {
                if (dynamic_cast<graphlib::OpNode *>(producer))
                { 
                    concat_operand = producer;
                    if (producer->as<graphlib::OpNode>()->op_type().op == "hslice")
                        slice = producer;
                }
            }

            if (concat_operand == nullptr)
                continue;

            // select out the last tile to stream to dram
            slice_factor = (slice != nullptr) ? std::get<int>(slice->as<graphlib::OpNode>()->op_type().attr[0]) : 1;

            // if we're only producing a single tile, we don't need to select anything
            bool control_edge_needed = false;
            if (concat_operand->shape()[-2] == 32)
            {
                edges = graph->operand_data_edges(output_node);
                graph->remove_edge(edges[0]);
                graphlib::Edge new_output_edge(concat_operand->id(), (graphlib::PortId)0, output_node->id(), (graphlib::PortId)0, graphlib::EdgeType::kData);
                graph->add_edge(new_output_edge);
                control_edge_needed = true;
            }
            else
            {
                graphlib::Shape concat_shape = producers[0]->shape();
                int end = concat_shape[-2];
                graphlib::OpType index_op("index", {-2, -32, end, 1});
                auto index_node = graph->add_node(
                    graphlib::create_node<graphlib::PyOpNode>(producers[0]->name() + "_index", index_op),
                    graph->get_subgraph_id_for_node(producers[0]->id()));
                edges = graph->operand_data_edges(output_node);
                graphlib::insert_node_on_edge(graph, edges[0], index_node);
                calculate_and_set_node_shape(graph, index_node);
            }

            // Update slice-factor for the case hslice does not exist in the producer chain
            slice_factor = (slice == nullptr) ? producers[0]->shape()[-3] : slice_factor;

            // Add control-edge and hstack op if needed
            if (slice_factor > 1)
            {
                graphlib::OpType stack_op("hstack", {slice_factor});
                auto stack_node = graph->add_node(
                    graphlib::create_node<graphlib::PyOpNode>(output_node->name() + "_stack", stack_op),
                    graph->get_subgraph_id_for_node(output_node->id()));
                if (control_edge_needed)
                {
                    graphlib::Edge control_edge(producers[0]->id(), (graphlib::PortId)0, stack_node->id(), (graphlib::PortId)0, graphlib::EdgeType::kControl);
                    graph->add_edge(control_edge); 
                }
                edges = graph->operand_data_edges(output_node);
                graphlib::insert_node_on_edge(graph, edges[0], stack_node);
                calculate_and_set_node_shape(graph, stack_node);
                calculate_and_set_node_shape(graph, output_node);
            }
        } 

        processed_outputs.push_back(output_node);

        // replace input node with parameter if this is a new cache, otherwise remove input node and connect to existing parameter
        graphlib::Node *producer_param = nullptr;
        graphlib::Node *input_param = nullptr;
        graphlib::Node * manual_output_edge = nullptr; 
        if (param_to_output.count(output_node))
        {
            producer_param = param_to_output[output_node];
        }
        else
        {
            // cross-attention cache, no concatination, cache is consumed as is link_cache_outputs_to_parameters couldn't find it for this subgraph
            unsigned int current_subgraph = graph->get_subgraph_id_for_node(output_node->id());
            std::vector<graphlib::Node *> current_subgraph_outputs = graph->ordered_module_outputs_by_subgraph_index(current_subgraph);
            int position = std::find(current_subgraph_outputs.begin(), current_subgraph_outputs.end(), output_node) - current_subgraph_outputs.begin();

            for (size_t i = 0; i < graph->num_subgraphs(); i++)
            {
                if (i == current_subgraph)
                    continue;
                std::vector<graphlib::Node *> subgraph_outputs = graph->ordered_module_outputs_by_subgraph_index(i);
                if (subgraph_outputs.size() == current_subgraph_outputs.size())
                {
                    if (param_to_output.count(subgraph_outputs[position]))
                    {
                        if (subgraph_outputs[position]->shape() == output_node->shape()) 
                        {
                            producer_param = param_to_output[subgraph_outputs[position]];
                            log_warning(LogGraphCompiler, "No paramater found to produce {}, linking to {}", output_node->name(), producer_param->name());
                            manual_output_edge = subgraph_outputs[position];
                            break;
                        }
                    }
                }
            }
        }

        if (producer_param)
        {
            if (inputs_to_param.count(producer_param))
            {   
                input_param = inputs_to_param[producer_param];
            }
        }
        if (not input_param)
        {
            std::string param_name = (producer_param ? producer_param->name() : input_node->name()) + "_cache";
            auto param = graph->add_node(
                graphlib::create_node<graphlib::InputNode>(param_name, graphlib::InputNodeType::Parameter, false),
                graph->get_subgraph_id_for_node(input_node->id()));
    
            graphlib::Shape param_shape = input_node->shape();
            param_shape[-3] /= slice_factor;
            param_shape[-1] *= slice_factor;
            param->set_shape(param_shape);
            param->set_output_df(output_node->output_df());
            param->as<graphlib::TaggedNode>()->tag("dont_consteval", "true");

            auto it = std::find(ordered_inputs.begin(), ordered_inputs.end(), input_node);
            TT_ASSERT(it != ordered_inputs.end());
            std::size_t index =  it - ordered_inputs.begin();
            ret.insert({param->name(), index});
            if (producer_param)
                inputs_to_param[producer_param] = param;
            input_param = param;

            if (is_concat)
            {
                params_to_rotate.push_back(param);
            }
        }

        graphlib::Edge new_input_edge(input_param->id(), (graphlib::PortId)0, graph->data_users(input_node)[0]->id(), (graphlib::PortId)0, graphlib::EdgeType::kData);
        graph->add_edge(new_input_edge);
        for (size_t i = 1; i < graph->data_users(input_node).size(); ++i)
        {
            graphlib::Edge new_input_edge_other(input_param->id(), (graphlib::PortId)i, graph->data_users(input_node)[i]->id(), (graphlib::PortId)0, graphlib::EdgeType::kData);
            graph->add_edge(new_input_edge_other);
        }

        if (output_all)
        {
            graphlib::Edge edge(output_node->id(), (graphlib::PortId)0, input_param->id(), (graphlib::PortId)0, graphlib::EdgeType::kPartialDataCopy);
            graph->add_edge(edge);
        }
        else
        {
            nodes_to_remove.push_back(output_node->id());
        }
        outputs_to_remove.push_back(output_node->id());

        if (manual_output_edge)
        {
            graphlib::Edge edge(manual_output_edge->id(), (graphlib::PortId)0, input_param->id(), (graphlib::PortId)0, graphlib::EdgeType::kPartialDataCopy);
            graph->add_edge(edge);
            outputs_to_remove.push_back(manual_output_edge->id());
            processed_outputs.push_back(manual_output_edge);
        }

        if (is_concat and slice_factor > 1)
        {
            graphlib::OpType slice_op("hslice", {slice_factor});
            auto slice_node = graph->add_node(
                graphlib::create_node<graphlib::PyOpNode>(input_node->name() + "_slice", slice_op),
                graph->get_subgraph_id_for_node(input_node->id()));

            graphlib::insert_node_on_edge(graph, new_input_edge, slice_node);
            calculate_and_set_node_shape(graph, slice_node);
        }

        inputs_to_remove.push_back(input_node->id());
    }
    for (graphlib::Node *output_node: graph->nodes_by_type(graphlib::NodeType::kOutput))
    {
        if (std::find(processed_outputs.begin(), processed_outputs.end(), output_node) != processed_outputs.end() or param_to_output.count(output_node) == 0)
        {
            continue;
        }
        
        graphlib::Node *producer_param = param_to_output[output_node];
        if (inputs_to_param.count(producer_param) == 0)
        {
            continue;
        }

        graphlib::Node *input_param = inputs_to_param[producer_param];
        graphlib::Edge edge(output_node->id(), (graphlib::PortId)0, input_param->id(), (graphlib::PortId)0, graphlib::EdgeType::kPartialDataCopy);
        graph->add_edge(edge);
        outputs_to_remove.push_back(output_node->id());

        // Add hstack if the input-param node and the output node have different z-dim
        int stack_factor = output_node->shape().canonical()[-3] / input_param->shape().canonical()[-3];
        if (stack_factor > 1)
        {
            graphlib::OpType stack_op("hstack", {stack_factor});
            auto stack_node = graph->add_node(
                graphlib::create_node<graphlib::PyOpNode>(output_node->name() + "_stack", stack_op),
            graph->get_subgraph_id_for_node(output_node->id()));
            graphlib::insert_node_on_edge(graph, graph->operand_data_edges(output_node)[0], stack_node);
            calculate_and_set_node_shape(graph, stack_node);
            calculate_and_set_node_shape(graph, output_node);
        }
    }
    for (graphlib::NodeId id : inputs_to_remove)
    {
        log_debug(LogGraphCompiler, "Removing module input {}", graph->node_by_id(id)->name());
        graph->remove_node(graph->node_by_id(id));
        graph->remove_module_input(id);
    }
    for (graphlib::NodeId id : outputs_to_remove)
    {
        log_debug(LogGraphCompiler, "Removing module output {}", graph->node_by_id(id)->name());
        graph->remove_module_output(id);
    }
    for (graphlib::NodeId id : nodes_to_remove)
    {
        log_debug(LogGraphCompiler, "Removing node {}", graph->node_by_id(id)->name());
        graph->remove_node(graph->node_by_id(id));
    }

    if (env_as<bool>("PYBUDA_ROTATE_PAST_CACHE_PARAMS", false))
        rotate_params(graph, params_to_rotate);

    return ret; 
}

std::map<std::string, std::size_t> link_past_cache_ios(graphlib::Graph *graph) 
{
    std::unordered_map<graphlib::Node *, graphlib::Node*> param_to_output = link_cache_outputs_to_parameters(graph);
    std::map<std::string, std::size_t> new_params = convert_inputs_to_params(graph, param_to_output);
    return new_params;
}
}  // namespace tt::passes
