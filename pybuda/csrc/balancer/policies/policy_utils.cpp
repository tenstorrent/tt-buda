// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/policies/policy_utils.hpp"

#include <experimental/filesystem>
#include <fstream>
#include <unordered_set>

#include "balancer/balancer.hpp"
#include "balancer/data_movement_bw_estimation.hpp"
#include "balancer/legalizer/legalizer.hpp"
#include "balancer/types.hpp"
#include "graph_lib/graph.hpp"
#include "passes/fork_join.hpp"
#include "placer/dram.hpp"
#include "placer/interactive_placer.hpp"
#include "placer/lower_to_placer.hpp"
#include "scheduler/scheduler.hpp"
#include "shared_utils/placement_printer.hpp"
#include "shared_utils/pretty_table.hpp"
#include "utils/assert.hpp"

using Graph = tt::graphlib::Graph;
using Node = tt::graphlib::Node;
using NodeType = tt::graphlib::NodeType;
using Edge = tt::graphlib::Edge;
using DataFormat = tt::DataFormat;

namespace tt::balancer
{

OpModelMap to_op_model_map(OpModels const &selected_op_models)
{
    OpModelMap op_model_map;
    for (auto const &[node, op_model] : selected_op_models)
    {
        op_model_map.insert({node->name(), op_model});
    }
    return op_model_map;
}

placer::PlacerSolution run_placer(
    Graph const *graph, const BalancerConfig &config, OpModelMap const &selected_op_models)
{
    std::unordered_map<std::string, tt::placer::GridShape> op_to_grid_shape;
    std::unordered_map<std::string, tt::placer::GridShape> input_queue_to_grid_shape;
    for (auto [node_name, op_model] : selected_op_models)
    {
        Node *node = graph->get_node_by_name(node_name);
        switch (node->node_type())
        {
            case NodeType::kInput:
            {
                input_queue_to_grid_shape.insert(
                    {node_name,
                     tt::placer::GridShape(
                         (std::uint32_t)op_model.grid_shape.r, (std::uint32_t)op_model.grid_shape.c)});
                break;
            }
            case NodeType::kBudaOp:
            {
                op_to_grid_shape.insert(
                    {node_name,
                     tt::placer::GridShape(
                         (std::uint32_t)op_model.grid_shape.r, (std::uint32_t)op_model.grid_shape.c)});
                break;
            }
            default: break;
        }
    }

    scheduler::Schedule scheduled_ops = run_scheduler(config.scheduler_config, graph);

    placer::PlacerConfig placer_config = {
        .chip_ids = config.chip_ids,
        .chip_placement_policy = config.chip_placement_policy,
        .device_config = config.device_config,
        .device_grid =
            placer::GridShape((uint32_t)config.device_config.grid_size.r, (uint32_t)config.device_config.grid_size.c),
        .contains_recompute = graph->contains_recompute_nodes(),
        .output_queues_on_host = config.output_queues_on_host,
        .strategy = placer::PlacementStrategy::LeftToRight,
        .op_to_grid_shape = op_to_grid_shape,
        .input_queue_to_grid_shape = input_queue_to_grid_shape,
        .op_to_epoch_type = placer::lowering::get_op_to_epoch_type_mapping(graph, scheduled_ops),
        .op_to_grad_op = placer::lowering::get_op_to_grad_op_mapping(graph, scheduled_ops),
        .op_to_recompute_op = placer::lowering::get_op_to_recompute_mapping(graph, scheduled_ops),
        .ops_tagged_for_chip_id_break = placer::lowering::tag_ops_for_chip_break(
            config.device_config.arch_name,
            config.op_names_to_chip_break,
            scheduled_ops,
            graph,
            config.use_interactive_placer),
        .ops_tagged_for_epoch_break = placer::lowering::tag_ops_for_epoch_break(
            config.device_config.arch_name,
            config.op_names_to_epoch_break,
            config.op_names_to_chip_break,
            scheduled_ops,
            graph,
            config.use_interactive_placer),
        .ops_tagged_for_temporal_epoch_break = placer::lowering::tag_ops_for_temporal_epoch_break(
            graph, scheduled_ops, config.op_name_to_placer_overrides),
        .fwd_to_bwd_nodes = placer::lowering::get_fwd_to_bwd_nodes(graph),
        .fwd_to_opt_nodes = placer::lowering::get_fwd_to_opt_nodes(graph, scheduled_ops),
        .output_ops = placer::lowering::get_output_nodes(graph),
        .op_to_chip_id_assignment = config.op_to_chip_id_assignment,
        .op_to_overrides = config.op_name_to_placer_overrides,
        .enable_auto_transposing_placement = config.enable_auto_transposing_placement,
    };

    // NB: We can avoid introducing both core-graph-lib and autograd modules in as dependencies
    // if we move the lowering code (relevant dependencies on both packages) here. Alternatively
    // only have lowering.hpp/cpp files depend on core-graph-lib/autograd
    placer::PlacerSolution solution = placer::placer(placer_config, scheduled_ops);

    // Visualize placement
    if (env_as<bool>("PYBUDA_BALANCER_PLACER_DATA"))
    {
        const std::string placement_dir_path = "bp_data";
        std::experimental::filesystem::create_directory(placement_dir_path);
        std::string file_name = placement_dir_path + "/" + (graph->name().empty() ? "noname" : graph->name()) + "_" +
                                policy_to_string(config.policy_type) + ".txt";
        std::ofstream of(file_name);
        dump_balancer_placer_data(
            graph, config.chip_ids, solution, selected_op_models, of, config.device_config.arch_name);
    }

    return solution;
}

std::vector<uint> get_num_epochs_per_node_epoch_type(Graph const *graph, tt::placer::PlacerSolution placer_solution)
{
    (void)graph;
    constexpr int NUM_EPOCH_TYPES = 3;
    constexpr std::array<NodeEpochType, NUM_EPOCH_TYPES> epoch_types = {
        NodeEpochType::Forward, NodeEpochType::Backward, NodeEpochType::Optimizer};

    std::vector<uint> num_epochs_per_node_type(NUM_EPOCH_TYPES, 0);
    std::unordered_map<uint, std::vector<std::string>> epoch_to_op_names;

    for (uint i = 0; i < placer_solution.num_epochs; i++)
    {
        epoch_to_op_names.emplace(i, std::vector<std::string>());
    }

    for (auto kvp : placer_solution.name_to_op_placement)
    {
        epoch_to_op_names.at(kvp.second.epoch_id()).push_back(kvp.first);
    }

    for (int i = 0; i < NUM_EPOCH_TYPES; ++i)
    {
        num_epochs_per_node_type[i] = placer_solution.num_temporal_epochs(epoch_types[i]);
    }

    // Pop opt and bwd if not training mode

    while (num_epochs_per_node_type.back() == 0)
    {
        num_epochs_per_node_type.pop_back();
    }

    return num_epochs_per_node_type;
}

void dump_balancer_placer_data(
    Graph const *graph,
    std::vector<std::uint32_t> chip_ids,
    tt::placer::PlacerSolution const &placer_solution,
    OpModelMap const &op_model_map,
    std::ostream &of,
    const std::string &arch_name)
{
    if (not env_as<bool>("PYBUDA_BALANCER_PLACER_DATA"))
        return;

    // Create some supporting structures
    std::unordered_map<std::string, int> op_name_to_id_map;
    for (std::pair<const std::string, tt::placer::OpPlacement> kvp : placer_solution.name_to_op_placement)
    {
        op_name_to_id_map.emplace(kvp.first, graph->get_node_by_name(kvp.first)->id());
    }

    std::vector<std::pair<std::string, int>> sorted_op_id_name_pairs;
    std::transform(
        op_name_to_id_map.begin(),
        op_name_to_id_map.end(),
        std::back_inserter(sorted_op_id_name_pairs),
        [](const std::pair<const std::string, int> &kvp) { return kvp; });

    std::sort(
        sorted_op_id_name_pairs.begin(),
        sorted_op_id_name_pairs.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.second < rhs.second; });

    // Create mapping of op id to new set of ids that are in [0, N)
    std::unordered_map<int, int> original_id_to_visualized_id;
    int new_id = 0;
    for (std::pair<std::string, int> kvp : sorted_op_id_name_pairs)
    {
        original_id_to_visualized_id.emplace(kvp.second, new_id);
        new_id++;
    }

    // Placer doesn't have access to graph and PlacerSolution is NodeEpochType-agnostic, so printer will be called here
    // Whether we're training or not, should be read from compiler config, but hack it for now
    uint node_epoch_types_count = graph->contains_bwd_nodes() ? 3 : 1;
    std::vector<uint> epochs_per_epoch_type = get_num_epochs_per_node_epoch_type(graph, placer_solution);

    tt::utils::PlacementPrinter::DeviceType dev_type = (arch_name == "grayskull")
                                                           ? tt::utils::PlacementPrinter::DeviceType::Grayskull
                                                           : tt::utils::PlacementPrinter::DeviceType::Wormhole;

    std::uint32_t max_chip_id = 0;
    for (std::uint32_t chip_id : chip_ids)
    {
        max_chip_id = std::max(max_chip_id, chip_id);
    }

    tt::utils::PlacementPrinter printer(dev_type, node_epoch_types_count, epochs_per_epoch_type, max_chip_id + 1);

    for (auto &kvp : placer_solution.name_to_op_placement)
    {
        std::string name = kvp.first;
        tt::placer::OpPlacement opPlacement = kvp.second;

        auto coords = opPlacement.placed_cores;

        printer.fillRectangle(
            placer_solution.temporal_epoch_id(name),
            opPlacement.chip_id,
            coords.start.row,
            coords.start.col,
            coords.end.row,
            coords.end.col,
            original_id_to_visualized_id.at(op_name_to_id_map[name])  // prints id for visualization
        );
    }

    of << printer.generatePlacementString();

    // Print op data
    tt::utils::PrettyTable table;
    table.add_row(
        {"Visual id",
         "Op id",
         "Op name",
         "Op type",
         "Grid (RxC)",
         "Cores",
         "Cycles",
         "mblock (t)",
         "ublock (u_kt)",
         "Data fmt",
         "Math fdlty",
         "L1 mem (kb)"});

    for (auto &kvp : sorted_op_id_name_pairs)
    {
        const std::string op_name = kvp.first;
        const int op_id = kvp.second;

        // Since op type is of format "BudaOp::matmul", we remove the prefix
        std::string op_type = graph->node_by_id(op_id)->get_type();
        TT_ASSERT(op_type.substr(0, 8) == "BudaOp::", "Op not a buda op!");
        op_type = op_type.substr(8);

        std::string placed_core_shapes;
        int placed_cores_volume = 0;
        tt::placer::CoordRange coord_range = placer_solution.name_to_op_placement.at(op_name).placed_cores;
        placed_core_shapes += " " + std::to_string(coord_range.size_r()) + "x" + std::to_string(coord_range.size_c());
        placed_cores_volume += coord_range.size_r() * coord_range.size_c();

        const OpModel &op_model = op_model_map.at(op_name);

        std::string execution_cycles = std::to_string(op_model.get_execution_cycles(arch_name));
        std::string memory_used_kb = round_float(op_model.get_l1_memory_usage() / 1024.f, 2);
        std::string mblock = std::to_string(op_model.block_shape().mblock_m) + "x" +
                             std::to_string(op_model.block_shape().mblock_n) + " " +
                             std::to_string(op_model.block_shape().t);
        std::string ublock =
            std::to_string(op_model.block_shape().ublock.rt) + "x" + std::to_string(op_model.block_shape().ublock.ct);
        std::string data_format = ((std::stringstream &)(std::stringstream() << op_model.data_format)).str();
        std::string math_fidelity = ((std::stringstream &)(std::stringstream() << op_model.math_fidelity())).str();

        table.add_row({
            std::to_string(original_id_to_visualized_id.at(op_id)),
            std::to_string(op_id),
            op_name,
            op_type,
            placed_core_shapes,
            std::to_string(placed_cores_volume),
            execution_cycles,
            mblock,
            ublock,
            data_format,
            math_fidelity,
            memory_used_kb,
        });
    }

    of << table.generate_table_string(tt::utils::PrettyTable::Format::Pretty) << std::endl;

    int epoch_id = 0;
    int total_cost = 0;
    std::vector<EpochCost> epoch_costs = calculate_epoch_costs(placer_solution, op_model_map, arch_name);
    fmt::print(of, "Epoch costs:\n");
    for (EpochCost epoch_cost : epoch_costs)
    {
        fmt::print(of, "  {}: {} cycles\n", epoch_id++, epoch_cost.setup_cycles + epoch_cost.runtime_cycles);
        total_cost += epoch_cost.setup_cycles + epoch_cost.runtime_cycles;
    }
    fmt::print(of, "  Total: {} cycles\n", total_cost);

    // TODO: print graph of ops to file stream
    // Consider graphviz:
    // -
    // https://stackoverflow.com/questions/9181183/how-to-print-a-boost-graph-in-graphviz-with-one-of-the-properties-displayed
    // - https://stackoverflow.com/questions/33301493/network-graph-visualisation
}

std::vector<EpochCost> calculate_epoch_costs(
    placer::PlacerSolution const &placer_solution, OpModelMap const &selected_op_models, std::string const &arch_name)
{
    std::vector<EpochCost> epoch_costs;
    epoch_costs.resize(placer_solution.num_epochs);
    for (auto const &[node, placement] : placer_solution.name_to_op_placement)
    {
        OpModel const &op_model = selected_op_models.at(node);
        epoch_costs[placement.epoch_id()].runtime_cycles =
            std::max(epoch_costs[placement.epoch_id()].runtime_cycles, op_model.get_execution_cycles(arch_name));
    }
    return epoch_costs;
}

void epoch_or_chip_break_remove_processed_nodes(
    const Graph *graph,
    std::vector<tt::scheduler::Schedule> &op_names_to_epoch_or_chip_break,
    const std::unordered_set<const tt::graphlib::Node *> &processed_nodes)
{
    if (processed_nodes.empty())
    {
        return;
    }

    auto it = op_names_to_epoch_or_chip_break.begin();
    while (it != op_names_to_epoch_or_chip_break.end())
    {
        auto &op_names = *it;
        auto op_names_it = op_names.begin();
        bool delete_op_names = false;
        while (op_names_it != op_names.end())
        {
            auto &op_name = *op_names_it;
            auto node = graph->get_node_by_name(op_name);
            if (processed_nodes.find(node) != processed_nodes.end())
            {
                delete_op_names = true;
                break;
            }
            else
            {
                ++op_names_it;
            }
        }

        if (delete_op_names)
        {
            it = op_names_to_epoch_or_chip_break.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

std::pair<scheduler::Schedule, std::unordered_set<string>> policy_run_scheduler(
    graphlib::Graph const *graph,
    BalancerConfig const &config,
    const std::unordered_set<const tt::graphlib::Node *> &processed_nodes,
    const tt::scheduler::Schedule &processed_schedule,
    std::vector<tt::scheduler::Schedule> &op_names_to_epoch_break)
{
    std::vector<tt::scheduler::Schedule> op_names_to_chip_break;
    const auto [scheduled_ops, epoch_break_ops, chip_break_ops] = policy_run_scheduler(
        graph, config, processed_nodes, processed_schedule, op_names_to_epoch_break, op_names_to_chip_break);
    return make_pair(std::move(scheduled_ops), std::move(epoch_break_ops));
}

std::tuple<scheduler::Schedule, std::unordered_set<string>, std::unordered_set<string>> policy_run_scheduler(
    graphlib::Graph const *graph,
    BalancerConfig const &config,
    const std::unordered_set<const tt::graphlib::Node *> &processed_nodes,
    const tt::scheduler::Schedule &processed_schedule,
    std::vector<tt::scheduler::Schedule> &op_names_to_epoch_break,
    std::vector<tt::scheduler::Schedule> &op_names_to_chip_break)
{
    scheduler::SchedulerConfig scheduler_config = config.scheduler_config;
    if (processed_nodes.size() > 0)
    {
        TT_ASSERT(processed_nodes.size() == processed_schedule.size());
        scheduler_config.ignored_nodes = &processed_nodes;
        scheduler_config.scheduler_constraints.push_back(processed_schedule);
    }

    scheduler::Schedule scheduled_ops = run_scheduler(scheduler_config, graph);

    epoch_or_chip_break_remove_processed_nodes(graph, op_names_to_epoch_break, processed_nodes);
    epoch_or_chip_break_remove_processed_nodes(graph, op_names_to_chip_break, processed_nodes);
    std::unordered_set<string> epoch_break_ops = placer::lowering::tag_ops_for_epoch_break(
        config.device_config.arch_name,
        op_names_to_epoch_break,
        op_names_to_chip_break,
        scheduled_ops,
        graph,
        config.use_interactive_placer);
    std::unordered_set<string> chip_break_ops = placer::lowering::tag_ops_for_chip_break(
        config.device_config.arch_name, op_names_to_chip_break, scheduled_ops, graph, config.use_interactive_placer);

    return make_tuple(std::move(scheduled_ops), std::move(epoch_break_ops), std::move(chip_break_ops));
}

// Cuts OPs in current epoch from rest of the graph.
//
void cut_graph_solver_epoch(
    const graphlib::Graph *graph, placer::InteractivePlacer &placer, legalizer::GraphSolver &graph_solver)
{
    // Only cut edges from ops that have been placed already
    balancer::CutEdges const &already_cut_edges = graph_solver.get_cut_edges();
    std::vector<std::string> const &current_epoch_ops = placer.current_epoch_ops();
    std::vector<graphlib::Edge> edges_to_cut;
    for (auto const &op_name : current_epoch_ops)
    {
        for (auto const &edge : graph->user_data_edges(graph->get_node_by_name(op_name)))
        {
            auto *user = graph->node_by_id(edge.consumer_node_id);
            if (user->node_type() != graphlib::NodeType::kBudaOp)
                continue;

            if (already_cut_edges.find(edge) != already_cut_edges.end())
                continue;

            if (std::find(current_epoch_ops.begin(), current_epoch_ops.end(), user->name()) != current_epoch_ops.end())
                continue;

            edges_to_cut.push_back(edge);
        }
    }

    if (edges_to_cut.size() > 0)
    {
        graph_solver.cut(edges_to_cut, true /*epoch_cut*/);
    }
}

// Validate that all ops in scheduled_ops have been placed in placer_solution.
//
void validate_solution(const scheduler::Schedule &scheduled_ops, const placer::PlacerSolution &placer_solution)
{
    if (placer_solution.name_to_op_placement.size() < scheduled_ops.size())
    {
        log_error(LogBalancer, "Some ops haven't been placed:");
        for (std::size_t i = 0; i < scheduled_ops.size(); i++)
        {
            if (placer_solution.name_to_op_placement.count(scheduled_ops[i]) == 0)
            {
                log_error(LogBalancer, "  - {}", scheduled_ops[i]);
            }
        }
        TT_THROW("Failed to place all ops.");
    }
}

// Merge buffering queues and ops for total current epoch nodes.
// Most balancer policies will track and work with op nodes only
// but for setting proper traversal contexts we need other nodes as well.
//
std::unordered_set<const tt::graphlib::Node *> calculate_current_epoch_nodes(
    const Graph *graph, const std::unordered_set<const tt::graphlib::Node *> &current_epoch_ops)
{
    std::unordered_set<const tt::graphlib::Node *> current_epoch_nodes(current_epoch_ops);

    for (const Node *op_node : current_epoch_ops)
    {
        for (Node *node : graph->data_operands(op_node))
        {
            if (node->node_type() == NodeType::kQueue and current_epoch_ops.count(graph->data_operands(node)[0]) > 0)
            {
                TT_ASSERT(node->as<graphlib::QueueNode>()->is_buffering());
                current_epoch_nodes.insert(node);
            }
        }
    }

    return current_epoch_nodes;
}

// Invoke SET of selected op_model on graphsolver instance for given node.
//
void set_op_model_for_node(
    legalizer::GraphSolver &graph_solver,
    const graphlib::Node *node,
    const OpModel &selected_op_model,
    std::string const &arch_name)
{
    graph_solver.set(node, selected_op_model);
    log_debug(
        LogBalancer,
        "Selected grid for node {} is {}, {}, {}, cycles {}",
        node->name(),
        selected_op_model.grid_shape,
        selected_op_model.t_stream_factor,
        selected_op_model.output_buffers[0].block_shape.ublock,
        selected_op_model.get_execution_cycles(arch_name));
}

void set_op_model_for_node_ribbon(
    legalizer::GraphSolver &graph_solver,
    const graphlib::Node *op,
    const OpModel &selected_op_model,
    std::uint32_t current_ribbon_size)
{
    log_trace(
        LogBalancer,
        "Selected grid for op {}: {}, {}, t-stream: {}, current_ribon={}",
        op->name(),
        selected_op_model.grid_shape.r,
        selected_op_model.grid_shape.c,
        selected_op_model.t_stream_factor,
        current_ribbon_size);
    graph_solver.set(op, selected_op_model);
}

int ribbon_buffering_factor(const OpModel &op_model) { return op_model.grid_shape.r; }

void cut_graph_solver_ribbon(
    const graphlib::Graph *graph,
    const graphlib::Node *op,
    placer::InteractivePlacer &placer,
    legalizer::GraphSolver &graph_solver)
{
    CutEdges pre_cut_edges = graph_solver.get_cut_edges();

    // Only cut edges from ops that have been placed already
    std::vector<graphlib::Edge> edges_to_cut;
    for (auto &edge : graph->operand_data_edges(op))
    {
        if (placer.op_placed(graph->node_by_id(edge.producer_node_id)->name()) && pre_cut_edges.count(edge) == 0)
        {
            edges_to_cut.push_back(edge);
        }
    }

    if (edges_to_cut.size() > 0)
    {
        log_debug(LogBalancer, "Cutting {} edges to {}", edges_to_cut.size(), op->name());
        graph_solver.cut(edges_to_cut);
    }
}

bool is_matmul(const graphlib::BudaOpNode *op)
{
    if (!op->is_matmul_not_sparse())
        return false;

    if (op->has_tag("reduce_r") || op->has_tag("reduce_c"))
        return false;

    return true;
}

bool prologue_ok(const OpModel &op_model)
{
    bool needs_prologue = op_model.buda_op_node->is_matmul();  // others don't matter much, as they are small
    bool has_prologue = false;
    if (needs_prologue)
    {
        if (op_model.buda_op_node->is_sparse_matmul())
        {
            TT_ASSERT(op_model.parameter_buffers.size() == 3);
            has_prologue = op_model.parameter_buffers[0] && op_model.parameter_buffers[2];
        }
        else if (op_model.buda_op_node->is_dense_matmul())
        {
            TT_ASSERT(op_model.parameter_buffers.size() > 1);
            has_prologue = op_model.parameter_buffers[1];
        }
        else
        {
            has_prologue = op_model.parameter_buffers.size() > 1 and op_model.parameter_buffers[1];
        }
    }

    bool prologue_ok = !needs_prologue || has_prologue;

    return prologue_ok;
}

bool ukt_ok(const OpModel &op_model)
{
    if (op_model.buda_op_node->is_matmul_not_sparse())
    {
        return op_model.input_buffers[0].block_shape.ublock.ct >= 4;
    }
    else if (op_model.buda_op_node->is_sparse_matmul())
    {
        return op_model.input_buffers[1].block_shape.ublock.rt >= 4;
    }

    return true;
}

bool mblock_size_ok(const OpModel &op_model)
{
    if (op_model.block_shape().t > 1)
    {
        return op_model.block_shape().volume_no_t() >= 8;
    }

    return true;
}

bool close_to_target_exec_cycles(int kernel_exec_cycles, int limiter_cycles, int target)
{
    return (limiter_cycles < target) && (kernel_exec_cycles > target * 0.8);
}

// Place sparse and dense matmul paired and in the same epoch if possible.
// Used only by RibbonV2 policy.
//
std::optional<placer::CoordRange> place_sparse_dense_pair(
    const graphlib::BudaOpNode *op,
    const OpModel *prefered_op_model,
    const graphlib::BudaOpNode *dense_matmul_op,
    const OpModel *prefered_op_model_dense,
    tt::placer::InteractivePlacer &interactive_placer,
    tt::placer::InteractivePlacer &ip_fittment_tester,
    bool &sparse_dense_pair)
{
    std::optional<placer::CoordRange> op_placement;

    // Place pair atomically in case row size matches and we can fit on a single epoch.
    if (prefered_op_model_dense->grid_shape.r == prefered_op_model->grid_shape.r &&
        interactive_placer.can_fit_on_single_epoch(
            prefered_op_model->grid_shape.r,
            prefered_op_model->grid_shape.c + prefered_op_model_dense->grid_shape.c,
            true /* allow_transpose */))
    {
        sparse_dense_pair = true;
        op_placement = interactive_placer.place_two_ops_rowwise(
            op->name(),
            prefered_op_model->grid_shape,
            dense_matmul_op->name(),
            prefered_op_model_dense->grid_shape,
            true);
    }
    // Row size doesn't match, still try placing them within the same epoch if possible.
    else if (can_fit_on_single_epoch(
                 ip_fittment_tester,
                 op->name(),
                 prefered_op_model->grid_shape,
                 dense_matmul_op->name(),
                 prefered_op_model_dense->grid_shape))
    {
        sparse_dense_pair = true;
        op_placement =
            interactive_placer.place_op(op->name(), prefered_op_model->grid_shape, true /* enable_transpose */);

        if (op_placement.has_value())
        {
            op_placement = interactive_placer.place_op(
                dense_matmul_op->name(), prefered_op_model_dense->grid_shape, true /* enable_transpose */);
        }
    }

    return op_placement;
}

// Iterate over ops and pick "role" op model based on preference function.
// Use "role" op model as an example of a grid size target with regards to the ribbon size.
// For each op, in GS filter out op models which do not match the ribbon size or have larger grid.
// In every step of the way use interactive placer to figure out when is epoch full.
// When epoch is full, target cycle is the slowest "role" op model placed.
//
int calculate_target_cycles_for_ribbon_size(
    const graphlib::Graph *graph,
    const BalancerConfig &config,
    legalizer::GraphSolver &graph_solver,
    tt::placer::InteractivePlacer &interactive_placer,
    tt::placer::InteractivePlacer &ip_fittment_tester,
    const std::uint32_t ribbon_size,
    const std::vector<std::string> &scheduled_ops,
    const std::unordered_set<string> &epoch_break_ops,
    const graphlib::NodeEpochType current_epoch_type,
    const std::uint32_t placed_op_index,
    const int epoch_target_cycles)
{
    TT_ASSERT(interactive_placer.current_epoch_empty());
    int target_exec_cycles = 0;

    // Should we apply filtering on GS search space while producing target cycles.
    //
    static const bool apply_filtering = env_as<bool>("PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES_APPLY_FILTERING", false);

    for (std::uint32_t op_index = placed_op_index; op_index < scheduled_ops.size(); op_index++)
    {
        const graphlib::Node *node = graph->get_node_by_name(scheduled_ops[op_index]);
        if (node->node_type() != NodeType::kBudaOp)
            continue;

        const graphlib::BudaOpNode *op = static_cast<const graphlib::BudaOpNode *>(node);

        // Check if there is a forced break at this op
        //
        bool new_epoch = (op_index > placed_op_index) &&
                         ((epoch_break_ops.count(node->name()) > 0) || (current_epoch_type != op->get_epoch_type()));

        if (new_epoch)
        {
            break;
        }

        bool op_already_set = false;
        const OpModel *prefered_op_model =
            pick_preferred_op_model(graph, config, graph_solver, op, ribbon_size, epoch_target_cycles);

        if (nullptr != prefered_op_model)
        {
            std::optional<placer::CoordRange> op_placement;
            bool sparse_dense_pair = false;

            // Special case for sparse matmuls. Try to pair them with the next op if preferable(sparse-dense
            // like pairs, see should_pair_with_sparse()).
            //
            if (op->is_sparse_matmul() and op_index < scheduled_ops.size() - 1)
            {
                graphlib::Node *next_node = graph->get_node_by_name(scheduled_ops[op_index + 1]);
                if (next_node->node_type() == NodeType::kBudaOp)
                {
                    const graphlib::BudaOpNode *dense_matmul_op = static_cast<const graphlib::BudaOpNode *>(next_node);
                    if (dense_matmul_op->should_pair_with_sparse(op, graph))
                    {
                        if (apply_filtering)
                        {
                            graph_solver.set_filter_grid_size(op, *prefered_op_model);
                        }

                        op_already_set = true;
                        const OpModel *prefered_op_model_dense = pick_preferred_op_model(
                            graph,
                            config,
                            graph_solver,
                            dense_matmul_op,
                            ribbon_size,
                            std::max(epoch_target_cycles, get_limiter_cycles(*prefered_op_model, graph, config)));
                        TT_ASSERT(prefered_op_model_dense != nullptr);

                        op_placement = place_sparse_dense_pair(
                            op,
                            prefered_op_model,
                            dense_matmul_op,
                            prefered_op_model_dense,
                            interactive_placer,
                            ip_fittment_tester,
                            sparse_dense_pair);

                        // Pair has been placed skip next op as it is already selected
                        // and calculate dense matmul cycles.
                        //
                        if (op_placement.has_value() and sparse_dense_pair)
                        {
                            if (apply_filtering)
                            {
                                graph_solver.set_filter_grid_size(dense_matmul_op, *prefered_op_model_dense);
                            }
                            target_exec_cycles = std::max(
                                target_exec_cycles, get_limiter_cycles(*prefered_op_model_dense, graph, config));
                            op_index++;
                        }
                    }
                }
            }

            if (!sparse_dense_pair)
            {
                op_placement =
                    interactive_placer.place_op(op->name(), prefered_op_model->grid_shape, true /* enable_transpose */);
            }

            if (!op_placement.has_value())
            {
                break;
            }

            if (apply_filtering and !op_already_set)
            {
                graph_solver.set_filter_grid_size(op, *prefered_op_model);
            }

            target_exec_cycles = std::max(target_exec_cycles, get_limiter_cycles(*prefered_op_model, graph, config));
        }
        else
        {
            TT_THROW("Failed to find valid op model for op {}", op->name());
        }
    }

    interactive_placer.rewind_epoch();
    return target_exec_cycles;
}

enum TargetProximity
{
    eGood = 0,
    eWeak,
    ePoor,
    eBad,
    eTerrible
};

TargetProximity target_proximity(int target, int cycles)
{
    if (cycles < target * 2)
    {
        return TargetProximity::eGood;
    }
    else if (cycles < target * 5)
    {
        return TargetProximity::eWeak;
    }
    else if (cycles < target * 10)
    {
        return TargetProximity::ePoor;
    }
    else if (cycles < target * 20)
    {
        return TargetProximity::eBad;
    }
    else
    {
        return TargetProximity::eTerrible;
    }
}

enum TargetCloseness
{
    eBelow = 0,
    eCloseB,
    eCloseO,
    eOver
};

TargetCloseness target_closeness(int target, int cycles)
{
    if (cycles < target * 0.8)
    {
        return TargetCloseness::eBelow;
    }
    else if (cycles < target)
    {
        return TargetCloseness::eCloseB;
    }
    else if (cycles < target * 1.2)
    {
        return TargetCloseness::eCloseO;
    }
    else
    {
        return TargetCloseness::eOver;
    }
}

bool bigger_mblock_util(
    const OpModel &current, const OpModel &candidate, const float current_exec_util, const float candidate_exec_util)
{
    if (candidate.block_shape().volume_no_t() > current.block_shape().volume_no_t())
    {
        return true;
    }
    else if (candidate.block_shape().volume_no_t() == current.block_shape().volume_no_t())
    {
        if (candidate_exec_util > current_exec_util)
        {
            return true;
        }
    }

    return false;
}

// OpModel preference comparison function. Returns true if candidate is better than current pick.
//
bool is_candidate_better_than_current(
    const OpModel &current,
    const OpModel &candidate,
    const Graph *graph,
    int ribbon_size,
    int target_exec_cycles,
    const BalancerConfig &balancer_config,
    const OpModels *graph_solver_selected_op_models)
{
    TT_ASSERT(current.buda_op_node == candidate.buda_op_node);

    // Op model compare version. If making major changes increment version and put the newest behaviour under that
    // version.
    //
    int op_model_compare_version = env_as<int>("PYBUDA_OP_MODEL_COMPARE_VERSION", 3);

    if (std::abs(ribbon_size - candidate.grid_shape.r) < std::abs(ribbon_size - current.grid_shape.r))
    {
        return true;
    }
    else if (std::abs(ribbon_size - candidate.grid_shape.r) > std::abs(ribbon_size - current.grid_shape.r))
    {
        return false;
    }

    // If both are same diff from target ribbon size, prefer smaller one.
    // It makes smaller "disturbance" to targeted ribbon and uses smaller number of cores.
    //
    if (candidate.grid_shape.r != current.grid_shape.r)
    {
        return candidate.grid_shape.r < current.grid_shape.r;
    }

    bool candidate_prologue_ok = prologue_ok(candidate);
    bool current_prologue_ok = prologue_ok(current);

    if (candidate_prologue_ok > current_prologue_ok)
    {
        return true;
    }
    else if (candidate_prologue_ok < current_prologue_ok)
    {
        return false;
    }

    int current_cycles = get_limiter_cycles(current, graph, balancer_config, graph_solver_selected_op_models);
    int candidate_cycles = get_limiter_cycles(candidate, graph, balancer_config, graph_solver_selected_op_models);

    // Both op_models are within target. Prefer smaller number of columns.
    //
    if (candidate_cycles <= target_exec_cycles and current_cycles <= target_exec_cycles)
    {
        if (candidate.grid_shape.c < current.grid_shape.c)
        {
            return true;
        }
        else if (candidate.grid_shape.c > current.grid_shape.c)
        {
            return false;
        }
    }

    if (!env_as<bool>("PYBUDA_TEMP_BALANCER_DISABLE_TARGET_PROXIMITY", false))
    {
        TargetProximity candidate_proxmity = target_proximity(target_exec_cycles, candidate_cycles);
        TargetProximity current_proximity = target_proximity(target_exec_cycles, current_cycles);

        if (candidate_proxmity < current_proximity)
        {
            return true;
        }
        else if (candidate_proxmity > current_proximity)
        {
            return false;
        }
    }

    bool ukt_ok_candidate = ukt_ok(candidate);
    bool ukt_ok_current = ukt_ok(current);

    if (ukt_ok_candidate > ukt_ok_current)
    {
        return true;
    }
    else if (ukt_ok_candidate < ukt_ok_current)
    {
        return false;
    }

    bool mblock_size_ok_candidate = mblock_size_ok(candidate);
    bool mblock_size_ok_current = mblock_size_ok(current);
    if (mblock_size_ok_candidate > mblock_size_ok_current)
    {
        return true;
    }
    else if (mblock_size_ok_candidate < mblock_size_ok_current)
    {
        return false;
    }

    // (1) if both are close to target, pick the one with the largest block (volume_no_t)
    // (2) if only one is close to target, pick that one
    // (3) if both are far from target, pick the one that is closer to target (in terms of execution
    // cycles)

    int current_exec_cycles = current.get_execution_cycles(balancer_config.device_config.arch_name);
    int candidate_exec_cycles = candidate.get_execution_cycles(balancer_config.device_config.arch_name);
    float current_exec_util = (float)current_exec_cycles / (float)current_cycles;
    float candidate_exec_util = (float)candidate_exec_cycles / (float)candidate_cycles;

    if (op_model_compare_version == 3)
    {
        // Main logic behind this type of compare is if grid size is the same, still prefer OpModels well below
        // target cycles as this will amortize inaccuracy of kernel estimates, thus reducing risk of slowing down whole
        // epoch because of bad estimate in scenarios where we had faster OpModels to choose from.
        //
        TargetCloseness candidate_closeness = target_closeness(target_exec_cycles, candidate_cycles);
        TargetCloseness current_closeness = target_closeness(target_exec_cycles, current_cycles);
        if (candidate_closeness < current_closeness)
        {
            return true;
        }
        else if (candidate_closeness > current_closeness)
        {
            return false;
        }
        else if (candidate_closeness == TargetCloseness::eOver)
        {
            // Both are well over, pick faster one.
            // Its more important to be faster here than to have larger block.
            //
            return candidate_cycles < current_cycles;
        }
        else
        {
            return bigger_mblock_util(current, candidate, current_exec_util, candidate_exec_util);
        }
    }
    else if (op_model_compare_version == 2)
    {
        if (close_to_target_exec_cycles(current_exec_cycles, current_cycles, target_exec_cycles))
        {
            if (close_to_target_exec_cycles(candidate_exec_cycles, candidate_cycles, target_exec_cycles))
            {
                return bigger_mblock_util(current, candidate, current_exec_util, candidate_exec_util);
            }
        }
        else if (close_to_target_exec_cycles(candidate_exec_cycles, candidate_cycles, target_exec_cycles))
        {
            return true;
        }
        else
        {
            if (candidate_cycles <= target_exec_cycles)
            {
                if (current_cycles > target_exec_cycles)
                {
                    return true;
                }
                else
                {
                    return bigger_mblock_util(current, candidate, current_exec_util, candidate_exec_util);
                }
            }
            else if (candidate_cycles < current_cycles)
            {
                return true;
            }
        }
    }
    else if (op_model_compare_version == 1)
    {
        if (close_to_target(current_cycles, target_exec_cycles))
        {
            if (close_to_target(candidate_cycles, target_exec_cycles))
            {
                if (candidate.block_shape().volume_no_t() > current.block_shape().volume_no_t())
                {
                    return true;
                }
            }
        }
        else if (close_to_target(candidate_cycles, target_exec_cycles))
        {
            return true;
        }
        else if (std::abs(target_exec_cycles - candidate_cycles) < std::abs(target_exec_cycles - current_cycles))
        {
            return true;
        }
    }

    return false;
}

bool can_fit_on_single_epoch(
    tt::placer::InteractivePlacer &ip_fittment_tester,
    const std::string &op_name_1,
    const tt::balancer::GridShape &op_shape_1,
    const std::string &op_name_2,
    const tt::balancer::GridShape &op_shape_2,
    bool enable_transpose)
{
    TT_ASSERT(ip_fittment_tester.current_epoch_empty(), "Test placer epoch must be empty!");
    std::optional<placer::CoordRange> test_placement;

    test_placement = ip_fittment_tester.place_op(op_name_1, op_shape_1, enable_transpose);

    TT_ASSERT(test_placement.has_value(), "Single op must always fit!");

    test_placement = ip_fittment_tester.place_op(op_name_2, op_shape_2, enable_transpose);

    ip_fittment_tester.rewind_epoch();
    return test_placement.has_value();
}

// Pick ribbon size for a given window of ops. The assumption is that all of them have the same r/c image dimension.
//
std::uint32_t pick_ribbon_size(
    std::uint32_t start_index,
    std::uint32_t end_index,  // end is not inclusive
    const Graph *graph,
    const legalizer::GraphSolver &graph_solver,
    const std::vector<std::string> &scheduled_ops,
    std::uint32_t device_rows)
{
    // set some tile limits. Min number ensures big enough blocks to keep perf running reasonably, and max avoids
    // blob sizes from exploding.
    std::uint32_t min_tile_height = env_as<int>("PYBUDA_RIBBON_MIN_TILE_HEIGHT", 1);
    std::uint32_t max_tile_height = env_as<int>("PYBUDA_RIBBON_MAX_TILE_HEIGHT", 200);

    // pick smallest legal ribbon
    bool minimize_ribbon = !env_as<bool>("PYBUDA_RIBBON_MAXIMIZE");

    bool skip_streaming = env_as<bool>("PYBUDA_RIBBON_SKIP_STREAMING");

    // override the max ribon size
    std::uint32_t max_ribbon_size = std::min(env_as<int>("PYBUDA_RIBBON_MAX_HEIGHT", device_rows), (int)device_rows);

    // Try to find a ribbon size that work for all ops in the ribbon
    std::unordered_set<std::uint32_t> candidates;
    std::unordered_map<std::uint32_t, std::unordered_set<std::uint32_t>>
        valid_map;  // map of ribbons that are valid for each op
    for (std::uint32_t i = 1; i <= max_ribbon_size; i++) candidates.insert(i);

    log_trace(LogBalancer, "Starting ribbon size search for {} ops", end_index - start_index);
    for (std::uint32_t i = start_index; i < end_index; i++)
    {
        graphlib::BudaOpNode *op = graph->get_node_by_name(scheduled_ops[i])->as<graphlib::BudaOpNode>();
        log_trace(LogBalancer, "  Checking op {}", op->name());
        for (auto grid : graph_solver.at(op))
        {
            if (skip_streaming && (grid.t_stream_factor.r > 1))
                continue;

            log_trace(
                LogBalancer,
                "    - Grid: {}, t-stream: {}, block shape rt: {}",
                grid.grid_shape,
                grid.t_stream_factor,
                grid.block_shape().rt());
            if (prologue_ok(grid) && ((std::uint32_t)grid.block_shape().rt() >= min_tile_height) &&
                ((std::uint32_t)grid.block_shape().rt() <= max_tile_height))
            {
                log_trace(LogBalancer, "     - valid");
                valid_map[i].insert(grid.grid_shape.r);
            }
        }

        std::unordered_set<std::uint32_t> to_erase;
        for (auto it : candidates)
            if (valid_map[i].count(it) == 0)
                to_erase.insert(it);
        for (auto it : to_erase) candidates.erase(it);

        if (candidates.empty())
            break;  // stop searching, we don't have anything
    }

    // If there are candidates available, pick smallest / largest
    if (!candidates.empty())
    {
        return minimize_ribbon ? *std::min_element(candidates.begin(), candidates.end())
                               : *std::max_element(candidates.begin(), candidates.end());
    }

    // std::cout << "No valid ribbon size found, looking for partials" << std::endl;
    //  TT_THROW("No valid ribbon size found"); // TODO: handle this case... right now, it hangs

    // No candidates available for everything. Need to find the best choice, so that everyone at least fits under
    // some ribbon size and nobody goes beyond it
    std::vector<std::uint32_t> partial_candidates;
    if (minimize_ribbon)
        for (std::uint32_t i = 1; i <= max_ribbon_size; i++) partial_candidates.push_back(i);
    else
        for (std::uint32_t i = max_ribbon_size; i > 0; i--) partial_candidates.push_back(i);

    // For each candidate, find if all ops would fit in something equal or smaller, and then take that.
    for (auto candidate : partial_candidates)
    {
        // At least one op should fit on this ribbon, otherwise it's not a real choice
        bool one_match = false;
        for (std::uint32_t i = start_index; i < end_index; i++)
        {
            if (valid_map[i].count(candidate) > 0)
            {
                one_match = true;
                break;
            }
        }

        if (!one_match)
            continue;

        bool all_ok = true;
        for (std::uint32_t i = start_index; i < end_index; i++)
        {
            bool ok = false;
            for (std::uint32_t ribbon = 1; ribbon <= candidate; ribbon++)
            {
                if (valid_map[i].count(ribbon) > 0)
                {
                    ok = true;
                    break;
                }
            }
            if (!ok)
            {
                all_ok = false;
                break;
            }
        }

        if (all_ok)
            return candidate;
    }

    return 1;  // we couldn't find anything... so we'll just have to pick smallest legal values
}

// Return the index of the next op that should change the ribbon size. It's either matmul or sparse
// matmul feeding it. Size of the array returned if no more changes found.
// In case we are recomputing within current ribbon, pass in current_matmul_dim_r from previous computation.
//
std::pair<std::uint32_t, std::uint32_t> get_next_ribbon_change_op(
    const graphlib::Graph *graph,
    std::uint32_t current_index,
    const std::vector<std::string> &scheduled_ops,
    std::uint32_t current_matmul_dim_r)
{
    for (std::uint32_t i = current_index; i < scheduled_ops.size(); i++)
    {
        graphlib::Node *node = graph->get_node_by_name(scheduled_ops[i]);

        if (node->node_type() != NodeType::kBudaOp)
            continue;

        const graphlib::BudaOpNode *op = node->as<graphlib::BudaOpNode>();
        if (!is_matmul(op))
            continue;

        std::uint32_t dim_r = op->shape().rt();
        if (current_matmul_dim_r == 0)
        {
            current_matmul_dim_r = dim_r;
            continue;
        }

        if (dim_r == current_matmul_dim_r)
            continue;

        // Matmul with different row shape. Let's see if there's a sparse matmul feeding it
        for (Node *operand : graph->data_operands(op))
        {
            // Skip through buffering queue.
            //
            if (operand->node_type() == NodeType::kQueue)
            {
                if (operand->as<graphlib::QueueNode>()->is_buffering())
                {
                    auto data_operands = graph->data_operands(operand);
                    TT_ASSERT(data_operands.size() == 1);
                    operand = data_operands.back();
                }
            }

            if (operand->node_type() != NodeType::kBudaOp)
                continue;

            if (operand->as<graphlib::BudaOpNode>()->is_sparse_matmul())
            {
                // Find the index. Should be a quick search back.
                for (int sparse_i = i - 1; sparse_i >= 0; sparse_i--)
                {
                    if (operand->name() == scheduled_ops[sparse_i])
                    {
                        return std::make_pair(sparse_i, current_matmul_dim_r);
                    }
                }
            }

            // No sparse matmul, switch on matmul itself
            return std::make_pair(i, current_matmul_dim_r);
        }
    }

    // No change until the end
    return std::make_pair(scheduled_ops.size(), current_matmul_dim_r);
}

// Can we bind sparse matmul and matmul and place them atomically together in a single block.
//
bool can_bind_sparse_dense_matmul_pair(
    const Graph *graph,
    const graphlib::BudaOpNode *sparse_op,
    OpModel const &sparse_op_model,
    const graphlib::BudaOpNode *dense_op,
    OpModel const &dense_op_model,
    placer::InteractivePlacer const &interactive_placer,
    bool allow_transpose)
{
    return sparse_op and sparse_op->is_sparse_matmul() and dense_op and
           dense_op->should_pair_with_sparse(sparse_op, graph) and
           sparse_op_model.grid_shape.r == dense_op_model.grid_shape.r and
           interactive_placer.can_fit_on_single_epoch(
               sparse_op_model.grid_shape.r,
               sparse_op_model.grid_shape.c + dense_op_model.grid_shape.c,
               allow_transpose) and
           dense_op == graph->data_users(sparse_op)[0];
}

// Test whether provided value is within specified range from the target execution cycles.
//
bool close_to_target(std::uint32_t test, std::uint32_t target) { return (test < target) && (test > target * 0.8); }

int get_limiter_cycles(
    const OpModel &op_model,
    const Graph *graph,
    const BalancerConfig &balancer_config,
    const OpModels *selected_op_models)
{
    return get_limiter_cycles(
        op_model,
        graph,
        balancer_config,
        0 /* dram_access_core_count */,
        0 /* pcie_access_core_count */,
        nullptr /* current_epoch_nodes */,
        false /* invalidate_cached */,
        selected_op_models);
}

int get_limiter_cycles(
    const OpModel &op_model,
    const Graph *graph,
    const DeviceConfig &device_config,
    const bool input_queues_on_host,
    const bool output_queues_on_host,
    const OpModels *selected_op_models)
{
    return get_limiter_cycles(
        op_model,
        graph,
        device_config,
        input_queues_on_host,
        output_queues_on_host,
        0 /* dram_access_core_count */,
        0 /* pcie_access_core_count */,
        nullptr /* current_epoch_nodes */,
        false /* invalidate_cached */,
        selected_op_models);
}

int get_limiter_cycles(
    const OpModel &op_model,
    const Graph *graph,
    const BalancerConfig &balancer_config,
    const int dram_access_core_count,
    const int pcie_access_core_count,
    const std::unordered_set<const tt::graphlib::Node *> *current_epoch_nodes,
    bool invalidate_cached,
    const OpModels *selected_op_models)
{
    return get_limiter_cycles(
        op_model,
        graph,
        balancer_config.device_config,
        balancer_config.input_queues_on_host,
        balancer_config.output_queues_on_host,
        dram_access_core_count,
        pcie_access_core_count,
        current_epoch_nodes,
        invalidate_cached,
        selected_op_models);
}

int get_limiter_cycles(
    const OpModel &op_model,
    const Graph *graph,
    const DeviceConfig &device_config,
    const bool input_queues_on_host,
    const bool output_queues_on_host,
    const int dram_access_core_count,
    const int pcie_access_core_count,
    const std::unordered_set<const tt::graphlib::Node *> *current_epoch_nodes,
    bool invalidate_cached,
    const OpModels *selected_op_models)
{
    const OpCycleEstimates op_cycle_estimates = get_op_cycles_estimates(
        op_model,
        graph,
        device_config,
        input_queues_on_host,
        output_queues_on_host,
        dram_access_core_count,
        pcie_access_core_count,
        current_epoch_nodes,
        invalidate_cached,
        selected_op_models);

    return op_cycle_estimates.calculate_op_limiter_cycles();
}


// Modelling and rough calculation of limiter cycles for op model.
// Limiter cycles are used to estimate how long would it take to execute the op model on the device.
// There are 3 main types of limiters:
// 1. kernel execution cycles
// 2. memory read cycles
// 3. memory write cycles
// The limiter cycles are the maximum of these 3.
// Memory read and write cycles are calculated based on the bandwidth of the memory and the size of the data(normalized
// per core). There are 3 types of memory with different estimated bandwits:
// 1. NOC bandwidth(for OP to OP connections)
// 2. DRAM bandwidth(read/write from/to DRAM on device - non host queues to OP)
// 3. PCIe bandwidth(read/write from/to host via PCIe - host queues to OP)
//
OpCycleEstimates get_op_cycles_estimates(
    const OpModel &op_model,
    const Graph *graph,
    const DeviceConfig &device_config,
    const bool input_queues_on_host,
    const bool output_queues_on_host,
    const int dram_access_core_count,
    const int pcie_access_core_count,
    const std::unordered_set<const tt::graphlib::Node *> *current_epoch_nodes,
    bool invalidate_cached,
    const OpModels *selected_op_models)
{
    static const bool model_pcie_bw = env_as<bool>("PYBUDA_TEMP_BALANCER_MODEL_PCIE_BW", true);
    static const bool disable_model_kb_prologue_bw = env_as<bool>("PYBUDA_TEMP_DISABLE_MODEL_KB_PROLOGUE_BW", false);

    // Should we use estimates for the NOC bandwidth.
    static const bool use_noc_bw_estimates = env_as<bool>("PYBUDA_BALANCER_USE_NOC_BW_ESTIMATES", false);

    const float inefficency_divider = 2.0;
    const float subchannel_oversub_coeff = 1.5;
    const float pcie_observed_max = 24;
    TT_ASSERT(op_model.buda_op_node);
    int kernel_cycles = op_model.get_execution_cycles(device_config.arch_name, false, invalidate_cached);
    std::vector<Edge> data_operands = graph->operand_data_edges(op_model.buda_op_node);
    std::vector<Edge> data_users = graph->user_data_edges(op_model.buda_op_node);

    std::vector<float> input_bw_estimates(data_operands.size(), 0.0f), output_bw_estimates(data_users.size(), 0.0f);
    std::vector<int> memory_read_cycles(data_operands.size(), 0), memory_write_cycles(data_users.size(), 0);

    // Use half of theoretical max for better average estimate for now.
    //
    float noc_bw = static_cast<float>(device_config.get_noc_bandwidth_bytes_per_cycle()) / inefficency_divider;
    float dram_bw_divider = std::max(
        inefficency_divider,
        std::ceil(
            dram_access_core_count / (device_config.get_dram_num_channels() * device_config.get_dram_num_subchannels() /
                                      subchannel_oversub_coeff)));

    // API is currently returning wrong value for WH
    // tenstorrent/budabackend#2423
    //
    float dram_bw = device_config.is_wormhole_b0()
                        ? 20.4 / dram_bw_divider
                        : static_cast<float>(device_config.get_dram_bandwidth_bytes_per_cycle()) / dram_bw_divider;
    float pcie_bw = 0 == pcie_access_core_count ? pcie_observed_max : pcie_observed_max / pcie_access_core_count;
    if (!model_pcie_bw)
    {
        // Temp fallback to legacy dram calc.
        //
        pcie_bw = dram_bw;
    }

    for (const Edge &edge : data_operands)
    {
        const unsigned int input_idx = edge.consumer_input_port_id;
        const unsigned int input_tensor_size_bytes = op_model.input_buffers[edge.consumer_input_port_id].total_size_bytes();

        Node *producer_node = graph->node_by_id(edge.producer_node_id);
        bool producer_is_queue =
            producer_node->node_type() == NodeType::kQueue || producer_node->node_type() == NodeType::kInput;

        const bool input_is_host_queue =
            producer_is_queue && is_input_host_queue(input_queues_on_host, graph, producer_node);
        const bool input_is_kb = op_model.input_buffers[edge.consumer_input_port_id].kernel_broadcast_tiles;
        const bool input_is_prologue = op_model.parameter_buffers[edge.consumer_input_port_id] &&
                                       op_model.parameter_buffers[edge.consumer_input_port_id].l1_size_tiles > 0;

        bool producer_is_host_input_buffer =
            producer_node->node_type() == tt::graphlib::NodeType::kBudaOp &&
            producer_node->as<tt::graphlib::BudaOpNode>()->has_tag("host_input_buffer");

        if (use_noc_bw_estimates && !producer_is_queue && selected_op_models)
        {
            TT_ASSERT(selected_op_models->count(producer_node) > 0);
            noc_bw = static_cast<float>(
                get_bandwidth_estimation(
                    graph, edge, selected_op_models->at(producer_node), op_model, false /* is_queue */)
                    .get_bandwidth());
        }

        // Legacy path for modelling BW
        //
        if (disable_model_kb_prologue_bw)
        {
            if (producer_is_queue and !op_model.parameter_buffers[edge.consumer_input_port_id])
            {
                if (!input_is_host_queue)
                {

                    input_bw_estimates[input_idx] = dram_bw;
                    memory_read_cycles[input_idx] = static_cast<int>(input_tensor_size_bytes / dram_bw);
                }
                else
                {
                    if (0 == pcie_access_core_count or op_model.buda_op_node->has_tag("host_input_buffer"))
                    {
                        pcie_bw =
                            pcie_observed_max / op_model.get_input_grid_shape(edge.consumer_input_port_id).volume();
                    }

                    input_bw_estimates[input_idx] = pcie_bw;
                    memory_read_cycles[input_idx] = static_cast<int>(input_tensor_size_bytes / pcie_bw);
                }
            }
            else if (producer_is_host_input_buffer)
            {
                if (0 == pcie_access_core_count)
                {
                    pcie_bw = pcie_observed_max / op_model.get_input_grid_shape(edge.consumer_input_port_id).volume();
                }

                input_bw_estimates[input_idx] = pcie_bw;
                memory_read_cycles[input_idx] = static_cast<int>(input_tensor_size_bytes / pcie_bw);
            }
            else
            {

                input_bw_estimates[input_idx] = noc_bw;
                memory_read_cycles[input_idx] = static_cast<int>(input_tensor_size_bytes / noc_bw);
            }
        }
        else
        {
            // Non-legacy path for modelling BW
            //
            // 5 cases:
            //   1. kernel broadcast (from queue)
            //   2. prologue (from queue)
            //   3. streaming (queue -> op)
            //   4. streaming (host -> op)
            //   5. streaming (op -> op) with kernel broadcast
            //   6. streaming (op -> op)
            //
            if (producer_is_queue)
            {
                if (input_is_kb)
                {
                    // kb (queue -> op)
                    //
                    TT_ASSERT(!input_is_prologue);

                    input_bw_estimates[input_idx] = dram_bw;      
                    memory_read_cycles[input_idx] = static_cast<int>(
                            (op_model.input_buffers[edge.consumer_input_port_id].kernel_broadcast_tiles *
                             tile_size_bytes(op_model.input_buffers[edge.consumer_input_port_id].data_format)) /
                            dram_bw / graph->get_microbatch());              
                }
                else if (input_is_prologue)
                {
                    // prologue (queue -> op)
                    //
                    TT_ASSERT(!input_is_kb);

                    input_bw_estimates[input_idx] = dram_bw;
                    memory_read_cycles[input_idx] = static_cast<int>(
                            input_tensor_size_bytes / dram_bw /
                            graph->get_microbatch());  // divide by microbatch as we only transfer data once per input
                                                       // in epoch
                }
                else if (input_is_host_queue)
                {
                    // streaming (host -> op)
                    //
                    TT_ASSERT(!input_is_prologue and !input_is_kb);
                    if (0 == pcie_access_core_count or op_model.buda_op_node->has_tag("host_input_buffer"))
                    {
                        pcie_bw =
                            pcie_observed_max / op_model.get_input_grid_shape(edge.consumer_input_port_id).volume();
                    }

                    input_bw_estimates[input_idx] = pcie_bw;
                    memory_read_cycles[input_idx] = static_cast<int>(input_tensor_size_bytes / pcie_bw);
                }
                else
                {
                    // streaming (queue -> op)

                    input_bw_estimates[input_idx] = dram_bw;
                    memory_read_cycles[input_idx] = static_cast<int>(input_tensor_size_bytes / dram_bw);
                }
            }
            else if (producer_is_host_input_buffer)
            {
                if (0 == pcie_access_core_count)
                {
                    pcie_bw = pcie_observed_max / op_model.get_input_grid_shape(edge.consumer_input_port_id).volume();
                }

                input_bw_estimates[input_idx] = pcie_bw;
                memory_read_cycles[input_idx] = static_cast<int>(input_tensor_size_bytes / pcie_bw);
            }
            else
            {
                if (input_is_kb)
                {
                    // streaming (op -> op) with kernel broadcast
                    //
                    TT_ASSERT(!producer_is_queue and !input_is_prologue);

                    input_bw_estimates[input_idx] = noc_bw;
                    memory_read_cycles[input_idx] = static_cast<int>(
                            (op_model.input_buffers[edge.consumer_input_port_id].kernel_broadcast_tiles *
                             tile_size_bytes(op_model.input_buffers[edge.consumer_input_port_id].data_format)) /
                            noc_bw);
                }
                else
                {
                    // streaming (op -> op)
                    //
                    TT_ASSERT(!producer_is_queue and !input_is_prologue and !input_is_kb);
                    
                    input_bw_estimates[input_idx] = noc_bw;
                    memory_read_cycles[input_idx] = static_cast<int>(input_tensor_size_bytes / noc_bw);
                }
            }
        }
    }

    if (0 == pcie_access_core_count)
    {
        pcie_bw = pcie_observed_max / op_model.grid_shape.volume();
    }

    // Revert noc_bw for output to rough estimation as it is not currently supported by bandwidth estimation.
    //
    noc_bw = static_cast<float>(device_config.get_noc_bandwidth_bytes_per_cycle()) / inefficency_divider;

    for (const Edge &edge : data_users)
    {
        const unsigned int output_idx = edge.producer_output_port_id;
        const unsigned int output_tensor_size_bytes = op_model.output_buffers[edge.producer_output_port_id].total_size_bytes();

        const tt::graphlib::Node *user_node = graph->node_by_id(edge.consumer_node_id);
        bool consumer_is_queue = user_node->node_type() == NodeType::kQueue ||
                                 user_node->node_type() == NodeType::kOutput ||
                                 (nullptr != current_epoch_nodes && current_epoch_nodes->count(user_node) == 0);

        const bool output_is_host_queue = is_output_host_queue(output_queues_on_host, graph, user_node);
        float write_bw;

        if (consumer_is_queue)
        {
            if (!output_is_host_queue)
            {
                write_bw = dram_bw;
            }
            else
            {
                write_bw = pcie_bw;
            }
        }
        else
        {
            write_bw = noc_bw;
        }

        output_bw_estimates[output_idx] = write_bw;
        memory_write_cycles[output_idx] = static_cast<int>(output_tensor_size_bytes / write_bw);
    }

    return OpCycleEstimates{
        .kernel_cycles = kernel_cycles,
        .input_bw_estimates = std::move(input_bw_estimates),
        .memory_read_cycles = std::move(memory_read_cycles),
        .output_bw_estimates = std::move(output_bw_estimates),
        .memory_write_cycles = std::move(memory_write_cycles)
    };
}

bool is_output_write_to_dram_over_target(
    const OpModel &op_model, const DeviceConfig &device_config, const int target_exec_cycles)
{
    int memory_write_cycles = 0;

    // API is currently returning wrong value for WH
    // tenstorrent/budabackend#2423
    //
    float dram_bw = device_config.is_wormhole_b0()
                        ? 20.4 / 2
                        : static_cast<float>(device_config.get_dram_bandwidth_bytes_per_cycle()) / 2;

    for (const BufferModel &output_buffer : op_model.output_buffers)
    {
        memory_write_cycles =
            std::max(memory_write_cycles, static_cast<int>(output_buffer.total_size_bytes() / dram_bw));
    }

    return memory_write_cycles > target_exec_cycles;
}

// Depending on insertion instructions insert NOPs or queues directly into GraphSolver.
//
bool buffer_graph(
    Graph *graph,
    tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash> &inst,
    legalizer::GraphSolver &graph_solver)
{
    vector<legalizer::BufferInfo> buffer_info;
    vector<graphlib::Edge> edges_to_cut;
    bool graph_modified = false;

    for (auto it : inst)
    {
        if (it.second->instr_type == InstructionType::NopInstruction)
        {
            NopInsertionInstruction *nopInsertInst = static_cast<NopInsertionInstruction *>(it.second.get());
            for (graphlib::Edge edge : graph->get_edges(
                     graph->get_node_by_name(nopInsertInst->src), graph->get_node_by_name(nopInsertInst->dest)))
            {
                if (edge.edge_type != graphlib::EdgeType::kData)
                {
                    continue;
                }

                buffer_info.emplace_back(edge, nopInsertInst->nop_count, nopInsertInst->hoist_tms);
            }
        }
        else if (it.second->instr_type == InstructionType::QueueInstruction)
        {
            QueueInsertionInstruction *qInsertInst = static_cast<QueueInsertionInstruction *>(it.second.get());
            std::function<bool(Edge)> edge_filter = [qInsertInst](Edge edge)
            { return edge.consumer_input_port_id == qInsertInst->input_id.value(); };
            std::vector<tt::graphlib::Edge> operand_edges =
                graph->operand_data_edges(graph->get_node_by_name(qInsertInst->dest), edge_filter);
            TT_ASSERT(operand_edges.size() == 1, "Expected exactly one operand edge per queue instruction!");
            edges_to_cut.push_back(operand_edges[0]);
        }
        else
        {
            TT_THROW("Unexpected insertion instruction type!");
        }
    }

    if (buffer_info.size() > 0)
    {
        auto result = graph_solver.buffer(buffer_info);
        graph_modified = true;
        TT_ASSERT(result.size() > 0, "Expected buffering to occur but nothing was buffered!");
    }

    if (edges_to_cut.size() > 0)
    {
        graph_solver.cut(edges_to_cut);
    }

    return graph_modified;
}

void EpochSolution::evaluate() const
{
    static const bool use_legacy_util_eval = env_as<bool>("PYBUDA_TEMP_RIBBON2_LEGACY_UTIL_EVAL", false);
    float pipeline_cycles = 0;
    // Treat non-matmul ops as 8x less efficient than matmul ops.
    // Treat sparse matmuls as least efficient, we want to assign them least amount of cores while at the same time not
    // making them epoch bottleneck.
    //
    const int matmul_penalty = 1;
    const int non_matmul_penalty = use_legacy_util_eval ? 128 : 8;
    const int sparse_matmul_penalty = 128;
    log_trace(LogBalancer, "RIBBON2: Calculating solution score for ribbon size {}", ribbon_size);

    for (const auto &op_model : selected_op_models)
    {
        // We have full epoch candidate. Recalculate impact on data BW.
        //
        int cycles = get_limiter_cycles(
            op_model,
            graph,
            *balancer_config,
            dram_readers_core_count + dram_writers_core_count,
            pcie_readers_core_count + pcie_writers_core_count,
            &current_epoch_nodes,
            false, /* invalidate_cache */
            &current_epoch_op_models);

        if (cycles > pipeline_cycles)
            pipeline_cycles = cycles;
    }

    log_trace(LogBalancer, "RIBBON2: pipeline_cycles = {}", pipeline_cycles);

    int used_cores = 0;
    float utilization = 0;
    for (const auto &op_model : selected_op_models)
    {
        std::uint32_t cores = op_model.grid_shape.volume();
        used_cores += cores;
        int util_penalty = matmul_penalty;

        if (op_model.buda_op_node->is_sparse_matmul())
        {
            util_penalty = sparse_matmul_penalty;
        }
        else if (!op_model.buda_op_node->is_matmul())
        {
            util_penalty = non_matmul_penalty;
        }

        if (!op_model.buda_op_node->is_buffering_op())
        {
            utilization += cores *
                           (op_model.get_execution_cycles(
                                balancer_config->device_config.arch_name, true) /
                            pipeline_cycles) /
                           util_penalty;
        }
    }

    log_trace(
        LogBalancer,
        "RIBBON2: pipeline_cycles = {}, epoch_target_cycles = {}, used_cores = {}, "
        "utilization = {}",
        pipeline_cycles,
        epoch_target_cycles,
        used_cores,
        utilization);

    this->pipeline_cycles = pipeline_cycles;
    this->used_cores = used_cores;
    this->needs_eval = false;
    this->utilization = utilization;
}

void EpochSolution::print() const
{
    for (const auto &op_model : selected_op_models)
    {
        log_trace(
            LogBalancer,
            "RIBBON2: (ribbon={})   {}: {}",
            ribbon_size,
            op_model.buda_op_node->name(),
            get_limiter_cycles(op_model, graph, *balancer_config, &current_epoch_op_models));
    }
}

void EpochSolution::recalc_nodes(bool update_current_epoch_collections)
{
    dram_readers_core_count = 0;
    dram_writers_core_count = 0;
    pcie_readers_core_count = 0;
    pcie_writers_core_count = 0;

    if (update_current_epoch_collections)
    {
        current_epoch_ops.clear();
        current_epoch_op_models.clear();
        current_epoch_nodes.clear();

        for (const auto &op_model : selected_op_models)
        {
            current_epoch_ops.emplace(op_model.buda_op_node);
            current_epoch_op_models.emplace(op_model.buda_op_node, op_model);
        }

        current_epoch_nodes = calculate_current_epoch_nodes(graph, current_epoch_ops);
    }

    for (const auto &op_model : selected_op_models)
    {
        std::vector<Edge> data_operands = graph->operand_data_edges(op_model.buda_op_node);
        std::vector<Edge> data_users = graph->user_data_edges(op_model.buda_op_node);

        for (const Edge &edge : data_operands)
        {
            Node *producer_node = graph->node_by_id(edge.producer_node_id);
            bool producer_is_queue = producer_node->node_type() == tt::graphlib::NodeType::kQueue ||
                                     producer_node->node_type() == tt::graphlib::NodeType::kInput;
            bool producer_is_host_input_buffer =
                producer_node->node_type() == tt::graphlib::NodeType::kBudaOp &&
                producer_node->as<tt::graphlib::BudaOpNode>()->has_tag("host_input_buffer");

            if (producer_is_queue and !op_model.parameter_buffers[edge.consumer_input_port_id])
            {
                if (!is_input_host_queue(balancer_config->input_queues_on_host, graph, producer_node))
                {
                    dram_readers_core_count += op_model.get_input_grid_shape(edge.consumer_input_port_id).volume();
                }
                else if (!graph->node_by_id(edge.consumer_node_id)
                              ->as<tt::graphlib::BudaOpNode>()
                              ->has_tag("host_input_buffer"))
                {
                    pcie_readers_core_count += op_model.get_input_grid_shape(edge.consumer_input_port_id).volume();
                }
            }
            else if (producer_is_host_input_buffer)
            {
                pcie_readers_core_count += op_model.get_input_grid_shape(edge.consumer_input_port_id).volume();
            }
        }

        for (const Edge &edge : data_users)
        {
            const tt::graphlib::Node *user_node = graph->node_by_id(edge.consumer_node_id);
            bool consumer_is_queue = user_node->node_type() == tt::graphlib::NodeType::kQueue ||
                                     user_node->node_type() == tt::graphlib::NodeType::kOutput ||
                                     current_epoch_nodes.count(user_node) == 0;

            if (consumer_is_queue)
            {
                if (!is_output_host_queue(balancer_config->output_queues_on_host, graph, user_node))
                {
                    dram_writers_core_count += op_model.grid_shape.volume();
                }
                else
                {
                    pcie_writers_core_count += op_model.grid_shape.volume();
                }
            }
        }
    }
}

BalancerScore score_solution(const std::vector<EpochSolution> &solutions, const DeviceConfig &device_config)
{
    BalancerScore balancer_score;
    float total_pipeline_cycles = 0;
    for (const auto &solution : solutions)
    {
        balancer_score.epoch_scores.push_back(device_config.get_clock_freq() / (float)solution.get_pipeline_cycles());
        total_pipeline_cycles += solution.get_pipeline_cycles();
    }

    balancer_score.solution_score = device_config.get_clock_freq() / total_pipeline_cycles;

    log_info(LogBalancer, "Balancer perf score : {}", balancer_score.solution_score);

    return balancer_score;
}

}  // namespace tt::balancer
