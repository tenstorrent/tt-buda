// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "perf_model/perf_model.hpp"

#include <unordered_map>

#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "perf_model/graph.hpp"
#include "perf_model/simulator.hpp"
#include "placer/placer.hpp"
#include "utils/logger.hpp"

using tt::LogPerfModel;

namespace tt::perf_model
{

// Loop up operand nodes in the map and conver
std::vector<NodeP> get_node_operands(
    graphlib::Graph *g, graphlib::Node *node, const std::unordered_map<graphlib::Node *, NodeP> node_map)
{
    std::vector<NodeP> ret;
    for (graphlib::Node *operand : g->data_operands(node))
    {
        if ((node_map.count(operand)) == 0)
            continue;
        ret.push_back(node_map.at(operand));
    }
    return ret;
}

std::vector<TensorData> get_node_inputs(graphlib::Graph *g, graphlib::Node *node)
{
    std::vector<TensorData> inputs;
    for (graphlib::Node *operand : g->data_operands(node))
    {
        inputs.push_back(
            {.shape = operand->shape(),
             .t = 1,  // TOOD
             .df = operand->output_df()});
    }
    return inputs;
}

std::vector<std::uint32_t> get_node_input_broadcast_multiplier(graphlib::Graph *g, graphlib::Node *node)
{
    auto operand_edges = g->operand_data_edges(node);
    if (node->node_type() != graphlib::kBudaOp)
        return std::vector<std::uint32_t>(operand_edges.size(), 1);

    std::vector<std::uint32_t> m;
    for (Edge e : g->operand_data_edges(node))
    {
        auto tms = g->get_edge_attributes(e)->get_tms();
        std::uint32_t multiplier = 1;
        for (auto tm : tms)
        {
            if (tm.op == "broadcast")
            {
                multiplier *= std::get<int>(tm.attr[1]);
            }
        }
        TT_ASSERT(multiplier >= 1);
        m.push_back(multiplier);
    }

    return m;
}

OpGrid get_op_grid(const placer::OpPlacement &placement)
{
    const auto &p = placement.placed_cores;
    if (placement.grid_transpose)
        return OpGrid{.loc_r = p.start.row, .loc_c = p.start.col, .size_r = p.size_c(), .size_c = p.size_r()};

    return OpGrid{.loc_r = p.start.row, .loc_c = p.start.col, .size_r = p.size_r(), .size_c = p.size_c()};
}

// Generate static perf data for an op
PerfDataP get_op_perf_data(
    graphlib::Graph *g, graphlib::BudaOpNode *op, const std::shared_ptr<balancer::BalancerSolution> balancer_solution)
{
    std::vector<TensorData> inputs = get_node_inputs(g, op);
    balancer::OpModel op_model = balancer_solution->op_models.at(op->name());
    auto ret = std::make_shared<PerfData>(PerfData{
        inputs,
        get_node_input_broadcast_multiplier(g, op),
        TensorData{.shape = op->shape(), .t = 1, .df = op->output_df()},

        OpPerfData(

            get_op_grid(balancer_solution->placer_solution.name_to_op_placement.at(op->name())),
            op_model,
            balancer_solution->placer_solution.temporal_epoch_id(op->name()),
            op->get_epoch_type())});

    if (op->op_type().op == "matmul")
    {
        ret->attr.m_k = std::get<int>(op->op_type().buda_attrs["m_k"]);
        ret->attr.u_kt = std::get<int>(op->op_type().buda_attrs["u_kt"]);
    }
    if (op->op_type().op == "reduce")
    {
        if (std::get<std::string>(op->op_type().buda_attrs["dim"]) == "z")
        {
            ret->attr.m_k = std::get<int>(op->op_type().buda_attrs["z"]);
        }
    }
    return ret;
}

graphlib::QueueNodeType get_queue_type(graphlib::Node *node) { return node->as<graphlib::QueueNode>()->queue_type(); }

// Generate static perf data for a queue
PerfDataP get_queue_perf_data(graphlib::Graph *g, graphlib::Node *node)
{
    return std::make_shared<PerfData>(PerfData{
        get_node_inputs(g, node),
        TensorData{.shape = node->shape(), .t = 1, .df = node->output_df()},
        QueuePerfData{
            .location = "dram",  // TODO
            .dram_channels = {0}}});
}

// Update per-epoch structures
void init_epoch(
    std::uint32_t temporal_epoch_id,
    std::vector<std::unique_ptr<Graph>> &temporal_epoch_graphs,
    std::vector<PerfModel::NodeMap> &epoch_node_map)
{
    while (temporal_epoch_graphs.size() <= temporal_epoch_id)
    {
        temporal_epoch_graphs.push_back(std::make_unique<Graph>());
        epoch_node_map.push_back(PerfModel::NodeMap());
    }
}

void PerfModel::create_op(
    graphlib::Graph *g,
    graphlib::BudaOpNode *op,
    const std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    NodeMap &node_map,
    std::vector<NodeMap> &epoch_node_map)
{
    std::uint32_t temporal_epoch_id = balancer_solution->placer_solution.temporal_epoch_id(op->name());
    init_epoch(temporal_epoch_id, temporal_epoch_graphs, epoch_node_map);

    PerfDataP perf_data = get_op_perf_data(g, op, balancer_solution);

    // Add to global graph
    auto global_operands = get_node_operands(g, op, node_map);
    NodeP new_node =
        graph->add_op(op->name(), op->op_type().op, global_operands, perf_data, global_operands.size() == 0);
    node_map.insert(std::make_pair(op, new_node));

    // Add to epoch graph
    auto epoch_operands = get_node_operands(g, op, epoch_node_map[temporal_epoch_id]);
    std::string op_type = op->is_sparse_matmul() ? "sparse_matmul" : op->op_type().op;
    NodeP epoch_new_node = temporal_epoch_graphs[temporal_epoch_id]->add_op(
        op->name(),
        op_type,
        epoch_operands,
        perf_data,
        epoch_operands.size() == 0);  // TODO: figure out input in constructor instead of here
    epoch_node_map[temporal_epoch_id].insert(std::make_pair(op, epoch_new_node));
}

void PerfModel::create_tm(
    graphlib::Graph *g,
    graphlib::BudaNaryTMNode *tm,
    const std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    NodeMap &node_map,
    std::vector<NodeMap> &epoch_node_map)
{
    graphlib::BudaOpNode *user = g->data_users(tm).at(0)->as<graphlib::BudaOpNode>();
    std::uint32_t temporal_epoch_id = balancer_solution->placer_solution.temporal_epoch_id(user->name());
    init_epoch(temporal_epoch_id, temporal_epoch_graphs, epoch_node_map);

    PerfDataP perf_data = get_op_perf_data(g, user, balancer_solution);

    // Add to global graph
    auto global_operands = get_node_operands(g, tm, node_map);
    NodeP new_node =
        graph->add_op(tm->name(), tm->op_type().op, global_operands, perf_data, global_operands.size() == 0);
    node_map.insert(std::make_pair(tm, new_node));

    // Add to epoch graph
    auto epoch_operands = get_node_operands(g, tm, epoch_node_map[temporal_epoch_id]);
    NodeP epoch_new_node = temporal_epoch_graphs[temporal_epoch_id]->add_op(
        tm->name(),
        tm->op_type().op,
        epoch_operands,
        perf_data,
        epoch_operands.size() == 0);  // TODO: figure out input in constructor instead of here
    epoch_node_map[temporal_epoch_id].insert(std::make_pair(tm, epoch_new_node));
}

void PerfModel::create_queue(
    graphlib::Graph *g,
    graphlib::QueueNode *q,
    const std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    NodeMap &node_map,
    std::vector<NodeMap> &epoch_node_map)
{
    // Queue could be e2e, in which case it's both input and output, on different epoch graphs

    // Check if there's a producer
    std::vector<NodeP> operands = get_node_operands(g, q, node_map);
    NodeP operand = operands.size() > 0 ? operands[0] : nullptr;

    // If there's no producer, it's a global input
    bool is_input = operand == nullptr;

    // Add to global graph
    PerfDataP perf_data = get_queue_perf_data(g, q);
    NodeP new_node = graph->add_queue(q->name(), get_queue_type(q), operand, perf_data, is_input);
    node_map.insert(std::make_pair(q, new_node));

    // Add as output to producer epoch
    std::uint32_t producer_epoch_id = 0;
    if (operand != nullptr)
    {
        producer_epoch_id = balancer_solution->placer_solution.temporal_epoch_id(g->data_operands(q)[0]->name());
        init_epoch(producer_epoch_id, temporal_epoch_graphs, epoch_node_map);

        // Get operand from the temporal epoch
        std::vector<NodeP> operands = get_node_operands(g, q, epoch_node_map[producer_epoch_id]);
        TT_ASSERT(operands.size() > 0);
        operand = operands[0];

        // Make a new copy of perf data, because in each epoch, the queue will be used differently
        PerfDataP perf_data = get_queue_perf_data(g, q);
        NodeP epoch_new_node = temporal_epoch_graphs[producer_epoch_id]->add_queue(
            q->name(), get_queue_type(q), operand, perf_data, false /* epoch input */);
        epoch_node_map[producer_epoch_id].insert(std::make_pair(q, epoch_new_node));
    }

    // Add as input to each epoch that reads it, unless it's also the producer_epoch_id
    std::unordered_set<std::uint32_t> inserted_epochs;
    for (graphlib::Node *user : g->data_users(q))
    {
        std::uint32_t temporal_epoch_id = balancer_solution->placer_solution.temporal_epoch_id(user->name());
        if ((operand != nullptr) && (producer_epoch_id == temporal_epoch_id))
            continue;

        if (inserted_epochs.count(temporal_epoch_id) > 0)
            continue;  // already added to this one

        inserted_epochs.insert(temporal_epoch_id);

        init_epoch(temporal_epoch_id, temporal_epoch_graphs, epoch_node_map);

        // Make a new copy of perf data, because in each epoch, the queue will be used differently
        PerfDataP perf_data = get_queue_perf_data(g, q);
        NodeP epoch_new_node = temporal_epoch_graphs[temporal_epoch_id]->add_queue(
            q->name(), get_queue_type(q), nullptr, perf_data, true /* epoch input */);
        epoch_node_map[temporal_epoch_id].insert(std::make_pair(q, epoch_new_node));
    }
}

void PerfModel::create_graphs(
    graphlib::Graph *g,
    const std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    bool input_queues_on_host,
    bool output_queues_on_host)
{
    graph = std::make_unique<Graph>();
    NodeMap node_map;                     // map of original graph to perf graph nodes
    std::vector<NodeMap> epoch_node_map;  // map of original graph to perf graph nodes

    std::ofstream op_perf;
    std::ofstream balancer_score;
    bool dump_op_perf = env_as<bool>("PYBUDA_OP_PERF");
    if (dump_op_perf)
    {
        op_perf.open("op_perf.csv");
        op_perf << "name, type, epoch, grid, tiles, cycles, limiter_cycles" << std::endl;

        balancer_score.open("balancer_score.csv");
        balancer_score << "epoch, score" << std::endl;
        size_t epoch = 0;
        for (float epoch_score : balancer_solution->balancer_score.epoch_scores)
        {
            balancer_score << epoch << ", " << epoch_score << std::endl;
            epoch++;
        }

        balancer_score << "total, " << balancer_solution->balancer_score.solution_score << std::endl;
    }

    // Convert the graph
    for (graphlib::Node *node : graphlib::topological_sort(*g))
    {
        if (node->node_type() == graphlib::NodeType::kBudaOp)
        {
            create_op(g, node->as<graphlib::BudaOpNode>(), balancer_solution, node_map, epoch_node_map);
            NodeP op = node_map.at(node);
            if (dump_op_perf)
                op_perf << op->get_name() << ", " << op->get_op_type() << ", "
                        << balancer_solution->placer_solution.temporal_epoch_id(op->get_name()) << ","
                        << op->get_perf_data()->op_perf_data.grid.size_r << "x"
                        << op->get_perf_data()->op_perf_data.grid.size_c << ", "
                        << op->get_perf_data()->output.size_in_tiles() << ", "
                        << op->get_perf_data()->op_perf_data.cycle_count_ideal(device_config.arch_name) << ", "
                        << op->get_perf_data()->op_perf_data.cycle_count_bw_limited(
                               device_config, g, input_queues_on_host, output_queues_on_host)
                        << std::endl;
        }
        else if (node->node_type() == graphlib::NodeType::kBudaNaryTM)
            create_tm(g, node->as<graphlib::BudaNaryTMNode>(), balancer_solution, node_map, epoch_node_map);
        else
            create_queue(g, node->as<graphlib::QueueNode>(), balancer_solution, node_map, epoch_node_map);
    }
    if (dump_op_perf)
    {
        op_perf.close();
        balancer_score.close();
    }
}

OpPerfCalculatedData get_op_perf_calculated_data(const PerfDataP perf_data, std::string const &arch_name)
{
    OpPerfCalculatedData ret;
    for (auto input : perf_data->inputs)
    {
        ret.input_bw_needed.push_back(
            1.0 * input.size_in_bytes() / perf_data->op_perf_data.cycle_count_ideal(arch_name));
    }
    ret.output_bw_ideal =
        1.0 * perf_data->output.size_in_bytes() / perf_data->op_perf_data.cycle_count_ideal(arch_name);
    return ret;
}

void PerfModel::calculate_ideal_bws(const SystemSpec &system_spec)
{
    // For each op, calculate ideal input/output bws
    for (NodeP node : graph->get_nodes())
    {
        PerfDataP perf_data = node->get_perf_data();
        if (!perf_data->is_op)
            continue;

        perf_data->op_perf_calculated_data = get_op_perf_calculated_data(perf_data, system_spec.arch_name);
    }

    // For queues, add up input bws for each of its consumers
    for (auto &epoch_graph : temporal_epoch_graphs)
    {
        for (NodeP node : epoch_graph->get_nodes())
        {
            PerfDataP perf_data = node->get_perf_data();
            if (perf_data->is_op)
                continue;

            float total_bw = 0.0;
            for (NodeP user : node->get_outputs())
            {
                std::uint32_t operand_index = user->get_operand_index(node);
                TT_ASSERT(user->is_op());
                total_bw += user->get_perf_data()->op_perf_calculated_data.input_bw_needed[operand_index];
            }
            perf_data->queue_perf_calculated_data.total_read_bw_ideal = total_bw;

            // If it's writen to, record that
            for (NodeP operand : node->get_operands())
            {
                TT_ASSERT(operand->is_op());
                perf_data->queue_perf_calculated_data.write_bw_ideal =
                    operand->get_perf_data()->op_perf_calculated_data.output_bw_ideal;
                break;  // one operand only
            }
        }
    }
}

void propagate_bws(Graph *graph, const SystemSpec &system)
{
    // Figure out how much dram channel bw is available to each queue
    std::vector<float> total_dram_bw_requested(system.dram_bw.size(), 0.0);
    for (NodeP node : graph->get_nodes())
    {
        PerfDataP perf_data = node->get_perf_data();
        if (perf_data->is_op)
            continue;

        std::uint32_t num_channels_used = perf_data->queue_perf_data.dram_channels.size();
        for (std::uint32_t channel : perf_data->queue_perf_data.dram_channels)
        {
            total_dram_bw_requested[channel] +=
                perf_data->queue_perf_calculated_data.total_read_bw_ideal / (float)num_channels_used;

            total_dram_bw_requested[channel] +=
                perf_data->queue_perf_calculated_data.write_bw_ideal / (float)num_channels_used;
        }
    }

    // Figure out how much of the requested bw can we actually give each queue
    std::vector<float> dram_channel_bw_percentage;
    for (std::uint32_t channel = 0; channel < system.dram_bw.size(); channel++)
    {
        dram_channel_bw_percentage.push_back(
            (total_dram_bw_requested[channel] <= system.dram_bw[channel])
                ? 1.0
                : system.dram_bw[channel] / total_dram_bw_requested[channel]);
    }

    for (NodeP node : graph->get_nodes())
    {
        PerfDataP perf_data = node->get_perf_data();
        OpPerfCalculatedData &od = perf_data->op_perf_calculated_data;

        if (perf_data->is_op)
        {
            float worst_input_bw_perc = 1.0;
            for (std::size_t i = 0; i < node->get_operands().size(); i++)
            {
                PerfDataP operand_perf_data = node->get_operands()[i]->get_perf_data();
                float operand_bw;
                if (operand_perf_data->is_op)
                    operand_bw = operand_perf_data->op_perf_calculated_data.output_bw_produced;
                else
                    operand_bw = operand_perf_data->queue_perf_calculated_data.total_read_bw_produced;

                od.input_bw_got.push_back(operand_bw);

                float perc = operand_bw / od.input_bw_needed[i];
                if (perc < worst_input_bw_perc)
                    worst_input_bw_perc = perc;
            }
            TT_ASSERT(worst_input_bw_perc > 0.0);
            od.output_bw_perc = worst_input_bw_perc;
            od.output_bw_produced = worst_input_bw_perc * od.output_bw_ideal;
            od.cycle_count_actual = perf_data->op_perf_data.cycle_count_ideal(system.arch_name) / worst_input_bw_perc;
            od.utilization = worst_input_bw_perc;  // TODO - need baseline utilization first
        }
        else
        {
            // queue
            QueuePerfCalculatedData &qd = perf_data->queue_perf_calculated_data;

            // Figure out total BW available
            qd.total_bw_perc = 0.0;
            for (std::uint32_t dram_channel : perf_data->queue_perf_data.dram_channels)
                qd.total_bw_perc += dram_channel_bw_percentage[dram_channel];
            qd.total_bw_perc /= (float)perf_data->queue_perf_data.dram_channels.size();

            qd.total_read_bw_produced = qd.total_bw_perc * qd.total_read_bw_ideal;
            qd.write_bw_received = qd.total_bw_perc * qd.write_bw_ideal;
        }
    }
}

bool is_matmul(NodeP node)
{
    // Matmul type, and not brcst / reduce
    return (
        node->is_op() && ((node->get_op_type() == "matmul") || (node->get_op_type() == "sparse_matmul")) &&
        (node->get_name().find("_brcst_") == std::string::npos) &&
        (node->get_name().find("_reduce_") == std::string::npos));
}

void PerfModel::calculate_utilization(const SystemSpec &system)
{
    std::ofstream os("utilization.txt");
    std::stringstream epoch_reports;
    std::uint32_t core_count = system.grid_size_c * system.grid_size_r;
    float overall_utilization = 0.0;
    std::vector<float> epoch_utilization;

    std::uint32_t total_matmul_cores = 0, total_other_cores = 0, total_empty_cores = 0, total_cores = 0;
    for (std::uint32_t epoch = 0; epoch < temporal_epoch_graphs.size(); epoch++)
    {
        // Figure out how many cores are using matmuls vs. else vs. unused
        // Figure out matmul utilization vs. max theoretical
        const std::unique_ptr<Graph> &epoch_graph = temporal_epoch_graphs[epoch];

        epoch_reports << "Epoch " << epoch << std::endl;
        epoch_reports << "===============" << std::endl;
        std::stringstream matmul_reports;

        std::uint32_t matmul_cores = 0;
        std::uint32_t other_cores = 0;
        float matmul_util = 0.0;

        for (NodeP node : epoch_graph->get_nodes())
        {
            if (!node->is_op())
                continue;
            std::uint32_t cores = node->get_perf_data()->op_perf_data.grid.size();

            if (is_matmul(node))
            {
                std::uint32_t cycles = node->get_perf_data()->op_perf_data.cycle_count_ideal(system.arch_name);
                std::uint32_t theoretical_cycles =
                    node->get_perf_data()->op_perf_data.theoretical_cycles(system.arch_name);
                float util = (float)theoretical_cycles / cycles;
                std::uint32_t util_p = (util * 100.0);

                matmul_reports << node->get_name() << ": cores " << cores << ", cycles " << cycles << ", theoretical "
                               << theoretical_cycles << ", util " << util_p << "%" << std::endl;

                matmul_util = ((matmul_util * matmul_cores) + (util * cores)) / (matmul_cores + cores);
                matmul_cores += cores;
            }
            else
            {
                other_cores += cores;
            }
        }

        std::uint32_t empty_cores = core_count - matmul_cores - other_cores;
        float matmul_core_util = (float)matmul_cores / core_count;
        std::uint32_t matmul_core_util_p = 100.0 * matmul_core_util;

        epoch_reports << "Matmul cores:     " << matmul_cores << " (" << matmul_core_util_p << "%)" << std::endl;
        epoch_reports << "Non-Matmul cores: " << other_cores << std::endl;
        epoch_reports << "Empty cores:      " << empty_cores << std::endl;
        epoch_reports << std::endl;
        std::uint32_t matmul_util_p = 100.0 * matmul_util;
        epoch_reports << "Matmul math utilization: " << matmul_util_p << "%" << std::endl;
        float epoch_util = matmul_util * matmul_core_util;
        std::uint32_t epoch_util_p = 100.0 * epoch_util;
        epoch_reports << "Overall epoch utilization: " << epoch_util_p << "%" << std::endl;
        epoch_reports << std::endl << "Matmul report: " << std::endl;
        epoch_reports << matmul_reports.str() << std::endl;

        epoch_utilization.push_back(epoch_util);
        overall_utilization = ((epoch * overall_utilization) + epoch_util) / (epoch + 1);

        total_matmul_cores += matmul_cores;
        total_other_cores += other_cores;
        total_empty_cores += empty_cores;
        total_cores += core_count;
    }

    os << "Overall utilization: " << (std::uint32_t)(100.0 * overall_utilization) << "%" << std::endl;
    os << "Total cores:            " << total_cores << std::endl;
    os << "Total matmul cores:     " << total_matmul_cores << " ("
       << (std::uint32_t)(100.0 * total_matmul_cores / total_cores) << "%)" << std::endl;
    os << "Total non-matmul cores: " << total_other_cores << " ("
       << (std::uint32_t)(100.0 * total_other_cores / total_cores) << "%)" << std::endl;
    os << "Total empty cores:      " << total_empty_cores << " ("
       << (std::uint32_t)(100.0 * total_empty_cores / total_cores) << "%)" << std::endl;
    os << std::endl;

    for (std::uint32_t epoch = 0; epoch < epoch_utilization.size(); epoch++)
    {
        os << "Epoch " << epoch << " utilization: " << (std::uint32_t)(100.0 * epoch_utilization[epoch]) << "%"
           << std::endl;
        results["epoch_" + std::to_string(epoch) + "_utilization"] = epoch_utilization[epoch];
    }
    os << std::endl;
    os << epoch_reports.str();

    results["overall_utilization"] = overall_utilization;
    results["total_matmul_cores"] = total_matmul_cores;
    results["total_non_matmul_cores"] = total_other_cores;
    results["total_empty_cores"] = total_empty_cores;

    os.close();
}

// Convert graph to perf model graph, and graphs for each temporal epoch
PerfModel::PerfModel(
    graphlib::Graph *g,
    const std::string &graph_name,
    const DeviceConfig &device_config,
    const std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    bool input_queues_on_host,
    bool output_queues_on_host) :
    graph_name(graph_name), device_config(device_config)
{
    // create main and epoch graphs
    create_graphs(g, balancer_solution, input_queues_on_host, output_queues_on_host);
    SystemSpec system = SystemSpec::get_for_device(device_config);

    // calculate ideal bandwidths for queues and ops
    calculate_ideal_bws(system);

    // calculate utilization
    if (env_as<bool>("PYBUDA_PERF_UTIL"))
        calculate_utilization(system);

    // Propagate BWs
    for (auto &epoch_graph : temporal_epoch_graphs) propagate_bws(epoch_graph.get(), system);

    if (env_as<bool>("PYBUDA_PERF_SIMULATOR"))
    {
        std::uint32_t original_microbatch = g->get_microbatch();
        if (auto sim_mb = env_as_optional<int>("PYBUDA_PERF_SIMULATOR_MICROBATCH"))
            g->set_microbatch(*sim_mb);

        bool sim_log = env_as<bool>("PYBUDA_PERF_SIMULATOR_LOG");
        bool sim_trace = env_as<bool>("PYBUDA_PERF_SIMULATOR_TRACE");

        std::uint32_t total_runtime = 0;
        for (std::uint32_t epoch = 0; epoch < temporal_epoch_graphs.size(); epoch++)
        {
            auto &epoch_graph = temporal_epoch_graphs[epoch];
            auto sim = perf_model::Simulator(epoch_graph.get(), g->get_microbatch(), sim_trace, sim_log);
            bool sim_ok = sim.run(device_config.arch_name, epoch);
            TT_ASSERT(sim_ok);
            std::uint32_t epoch_timestamp = sim.get_timestamp();
            log_debug(tt::LogPerfModel, "Epoch {} expected cycles: {}", epoch, epoch_timestamp);
            results["expected_epoch_" + std::to_string(epoch) + "_cycles"] = epoch_timestamp;
            total_runtime += epoch_timestamp;
        }
        // TBD device config
        float cycles_per_second = 1.2 * 1000000000;
        float expected_perf = round(100.0 * g->get_microbatch() * (cycles_per_second / total_runtime)) / 100.0;
        log_info(
            tt::LogPerfModel,
            "Expected perf: {} samples/s (Total cycles {} for {} inputs)",
            expected_perf,
            total_runtime,
            g->get_microbatch());

        results["total_runtime_cycles"] = total_runtime;
        results["expected_perf"] = expected_perf;

        if (env_as<bool>("PYBUDA_PERF_STOP_AFTER_SIMULATOR"))
            TT_ASSERT(false);  // hacky way to stop

        // revert
        g->set_microbatch(original_microbatch);
    }
}

std::unordered_map<std::string, float> run_performance_model(
    graphlib::Graph *g,
    const std::string &graph_name,
    const DeviceConfig &device_config,
    const std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    bool input_queues_on_host,
    bool output_queues_on_host)
{
    log_info(tt::LogPerfModel, "Running performance model...");
    PerfModel model =
        PerfModel(g, graph_name, device_config, balancer_solution, input_queues_on_host, output_queues_on_host);
    return model.get_results();
}

}  // namespace tt::perf_model
