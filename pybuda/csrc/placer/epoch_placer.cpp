// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placer/epoch_placer.hpp"

#include "balancer/balancer.hpp"
#include "balancer/legalizer/legalizer.hpp"
#include "balancer/policies/policy_nlp.hpp"
#include "graph_lib/defines.hpp"
#include "placer/evaluator.hpp"
#include "placer/grid_placer.hpp"
#include "placer/post_epoch_passes.hpp"
#include "placer/pre_epoch_passes.hpp"
#include "scheduler/interactive_scheduler.hpp"

using namespace tt::balancer;

namespace tt::placer
{

static std::tuple<OpModelMap, BlockShapeMap, OutputHostTMMap, CutEdges> run_balancer(
    Graph* graph,
    BalancerConfig const& config,
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection,
    const LegalOpModels& valid_op_models,
    std::uint32_t target_cycles)
{
    auto graph_solver = get_graph_solver(config, cache_collection, graph, valid_op_models);
    legalizer::GraphSolverSolution graph_solver_solution = run_policy_nlp(graph, config, graph_solver, target_cycles);
    update_ops_on_selected_op_models(graph, graph_solver_solution.selected_op_models);
    return legalizer::resolve_block_shapes(graph, config, graph_solver_solution);
}

void insert_input_queues(PlacerSolution& placer_solution, const Graph* graph, const OpModelMap& op_models)
{
    // Add input queues to the placer solution
    for (auto [node_name, op_model] : op_models)
    {
        Node* node = graph->get_node_by_name(node_name);
        switch (node->node_type())
        {
            case graphlib::NodeType::kInput:
            {
                placer_solution.input_queue_to_grid_shape.insert(
                    {node_name,
                     tt::placer::GridShape(
                         (std::uint32_t)op_model.grid_shape.r, (std::uint32_t)op_model.grid_shape.c)});
                break;
            }
            default: break;
        }
    }
}

PlacerSolution place_epoch(
    std::uint32_t epoch_index,
    graphlib::NodeEpochType epoch_type,
    balancer::BalancerConfig const& config,
    OpModelMap const& selected_op_models,
    scheduler::InteractiveScheduler& ischeduler)
{
    std::optional<Coord> starting_coordinate = std::nullopt;

    unordered_map<string, OpPlacement> name_to_op_placement;
    map<PlacerSolution::EpochId, int> epoch_id_to_chip;
    unordered_map<int, vector<OpPlacement>> epoch_id_to_op_placement;
    EpochIdToDeviceGrid epoch_id_to_device_grid;

    epoch_id_to_device_grid.rows = config.device_config.grid_size.r;  // TODO: get harvested rows
    epoch_id_to_device_grid.columns = config.device_config.grid_size.c;
    epoch_id_to_device_grid.initialize_device_grid(epoch_index);

    std::uint32_t current_row = 0;
    while (true)
    {
        std::vector<std::string> candidate_ops = ischeduler.get_ops();
        if (candidate_ops.size() == 0)
            break;

        std::unordered_map<std::string, tt::placer::GridShape> op_to_grid_shape;

        // Try, in priority order, to place on the same row, then same epoch
        std::optional<std::uint32_t> selected_op = std::nullopt;
        DeviceGridPlacement device_grid_placement;
        for (std::size_t i = 0; i < candidate_ops.size(); i++)
        {
            std::string name = candidate_ops.at(i);
            auto op_model = selected_op_models.at(name);
            op_to_grid_shape.insert(
                {name, GridShape((std::uint32_t)op_model.grid_shape.r, (std::uint32_t)op_model.grid_shape.c)});

            std::optional<DeviceGridPlacement> p = place_one_op(
                name,
                op_to_grid_shape,
                epoch_id_to_device_grid.get_device_grid(epoch_index),
                config.op_name_to_placer_overrides,
                config.enable_auto_transposing_placement,
                starting_coordinate);

            if (!p.has_value())  // doesn't fit
                continue;

            bool placed_on_same_row = (device_grid_placement.placed_cores.start.row == current_row);
            if (!selected_op.has_value() || placed_on_same_row)
            {
                device_grid_placement = p.value();
                selected_op = i;
            }

            if (placed_on_same_row)
                break;  // same row, we can't do any better
        }

        if (!selected_op.has_value())
            break;  // nothing fits in this epoch

        std::string name = candidate_ops.at(selected_op.value());
        ischeduler.accept_op(candidate_ops.at(selected_op.value()));  // placed
        current_row = device_grid_placement.placed_cores.start.row;

        OpPlacement op_placement = OpPlacement{
            .id = 0,
            .name = name,
            .chip_id = 0,
            .global_epoch_id = epoch_index,
            .grid_transpose = device_grid_placement.grid_transpose,
            .placed_cores = device_grid_placement.placed_cores};
        name_to_op_placement[op_placement.name] = op_placement;
        epoch_id_to_op_placement[epoch_index].push_back(op_placement);

        epoch_id_to_device_grid.fill_device_grid_with_placement(
            epoch_index, device_grid_placement.placed_cores.start, op_to_grid_shape.at(name));

        log_debug(
            tt::LogPlacer,
            "\tPlacing {} with grid_shape ({}, {}) onto:",
            op_placement.name,
            op_to_grid_shape.at(op_placement.name).rows,
            op_to_grid_shape.at(op_placement.name).columns);
        log_debug(
            tt::LogPlacer,
            "\t\t chip_id={}, epoch_id={}, inclusive_start: {}, exclusive_end={}",
            op_placement.chip_id,
            op_placement.epoch_id(),
            op_placement.placed_cores.start,
            op_placement.placed_cores.end);
    }

    unordered_map<int, EpochInfo> epoch_id_to_epoch_info;

    epoch_id_to_epoch_info[epoch_index] = EpochInfo{
        .global_epoch_id = epoch_index,
        .temporal_epoch_id = epoch_index,
        .spatial_epoch_id = 0,
        .epoch_type = epoch_type,
    };

    std::uint32_t num_epochs = (name_to_op_placement.size() == 0) ? 0 : 1;
    PlacerSolution placer_solution = PlacerSolution{
        .name_to_op_placement = std::move(name_to_op_placement),
        .input_queue_to_grid_shape = {},
        .name_to_queue_placement = {},
        .epoch_id_to_chip = std::move(epoch_id_to_chip),
        .epoch_id_to_subgraph_index = {},
        .epoch_id_to_op_placement = std::move(epoch_id_to_op_placement),
        .epoch_id_to_device_grid = std::move(epoch_id_to_device_grid),
        .epoch_id_to_epoch_info = std::move(epoch_id_to_epoch_info),
        .num_epochs = num_epochs,
    };

    return placer_solution;
}

std::shared_ptr<balancer::BalancerSolution> run_epoch_placer(
    Graph** graph,
    balancer::BalancerConfig const& config,
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection)
{
    PlacerHistory history;

    Graph* current_graph = *graph;

    PlacerSolution placer_solution;
    scheduler::InteractiveScheduler ischeduler =
        scheduler::InteractiveScheduler(config.scheduler_config, current_graph, graphlib::NodeEpochType::Forward);
    bool first_epoch = true;

    // Final balancer solution, merged over epochs
    OpModelMap selected_op_models;
    BlockShapeMap selected_block_shape_map;
    OutputHostTMMap selected_output_host_tms;

    // Global graph legal op models, generated by the legalizer
    LegalOpModels global_valid_op_models = legalizer::get_legal_op_models(current_graph, config, cache_collection);
    auto graph_solver = get_graph_solver(config, cache_collection, current_graph, global_valid_op_models);

    // Figure out initial global target cycle count
    std::uint32_t global_target_cycles;
    if (auto manual_target = env_as_optional<int>("PYBUDA_NLP_MANUAL_TARGET"))
    {
        global_target_cycles = *manual_target;
        log_info(LogBalancer, "Manual override of target cycles to {}", global_target_cycles);
    }
    else
    {
        global_target_cycles = calculate_target_cycles(current_graph, graph_solver, config.device_config.arch_name);
    }

    // In case of recompile, we can offset the target cycles to get a different solution.
    global_target_cycles += config.target_cycles_offset;

    auto [op_models, block_shape_map, output_host_tms, cut_edges] =
        run_balancer(current_graph, config, cache_collection, global_valid_op_models, global_target_cycles);

    graphlib::NodeEpochType current_epoch_type =
        graphlib::NodeEpochType::Forward;  // TODO: we should do bwd first, figure out recompute, etc.
    while (true)
    {
        // First attempt at placement uses global targets and global valid shapes
        bool global_target_run = true;
        bool epoch_complete = false;

        std::uint32_t eval_attempt = 0;
        EpochPlacement best_placement;  // currently best placement

        std::vector<std::uint32_t> eval_target_cycles;
        std::uint32_t target_cycles_to_try = 10;  // todo, config
        std::uint32_t target_cycles_low = 0.25 * global_target_cycles;
        std::uint32_t target_cycles_high = 4 * global_target_cycles;
        for (std::uint32_t i = 0; i < target_cycles_to_try; i++)
        {
            eval_target_cycles.push_back(
                target_cycles_low + 1.0 * i * (target_cycles_high - target_cycles_low) / (target_cycles_to_try - 1));
        }

        while (!epoch_complete)  // Run the loop until we're satisfied, or enough attempts were done
        {
            // Run pre-epoch passes, potentially modifying the graph
            std::unique_ptr<Graph> modified_graph = run_pre_epoch_passes(current_graph, config, history);

            // Run legalizer/solver
            if (modified_graph)
            {
                current_graph = modified_graph.get();
                global_valid_op_models = legalizer::get_legal_op_models(current_graph, config, cache_collection);
                std::tie(op_models, block_shape_map, output_host_tms, cut_edges) =
                    run_balancer(current_graph, config, cache_collection, global_valid_op_models, global_target_cycles);
            }

            if (!global_target_run)
            {
                std::tie(op_models, block_shape_map, output_host_tms, cut_edges) = run_balancer(
                    current_graph,
                    config,
                    cache_collection,
                    global_valid_op_models,
                    eval_target_cycles.at(eval_attempt));
            }

            // Run op-by-op placer, and place full epoch
            auto checkpoint = ischeduler.save_checkpoint();
            auto epoch_placer_solution = place_epoch(  // TODO on epoch type
                history.current_epoch(),
                current_epoch_type,
                config,
                op_models,
                ischeduler);

            // Run post-epoch passes
            PlacerAttemptSummary summary = run_post_epoch_passes(placer_solution, epoch_placer_solution, history);

            if (summary.fail)
            {
                log_trace(
                    LogPlacer,
                    "Placer attempt {} on epoch {} failed.",
                    history.current_attempt(),
                    history.current_epoch());
                if (history.current_attempt() > 5)
                    TT_THROW("Epoch placer failed to place an epoch more than 5 times. Aborting.");
            }
            else
            {
                log_trace(
                    LogPlacer,
                    "Placer attempt {} on epoch {} passed",
                    history.current_attempt(),
                    history.current_epoch());

                auto balancer_solution = std::make_shared<balancer::BalancerSolution>(
                    epoch_placer_solution, op_models, block_shape_map, output_host_tms, cut_edges);
                EpochPlacement new_placement(
                    balancer_solution,
                    ischeduler.save_checkpoint(),
                    std::move(modified_graph),
                    config.device_config.arch_name);

                if (global_target_run)
                {
                    log_debug(
                        LogPlacer,
                        "Placer initial eval attempt for epoch {} has score of {}, for target cycles {}",
                        history.current_epoch(),
                        new_placement.score(),
                        global_target_cycles);
                }
                else
                {
                    log_debug(
                        LogPlacer,
                        "Placer eval attempt {} of {} for epoch {} has score of {}, for target_cycles {}",
                        eval_attempt + 1,
                        eval_target_cycles.size(),
                        history.current_epoch(),
                        new_placement.score(),
                        eval_target_cycles.at(eval_attempt));
                }

                if (global_target_run || (new_placement.is_better_than(best_placement)))
                    best_placement = std::move(new_placement);

                if (!global_target_run)
                    eval_attempt++;
                global_target_run = false;
                history.reset_attempts();

                epoch_complete = (eval_attempt >= eval_target_cycles.size());
            }

            ischeduler.restore_checkpoint(checkpoint);
        }

        TT_ASSERT(best_placement.valid());

        // Epoch done, record solution
        const auto& best_solution = best_placement.solution();
        for (auto it : best_solution->placer_solution.name_to_op_placement)
        {
            const std::string& name = it.first;
            selected_op_models.insert(std::make_pair(name, best_solution->op_models.at(name)));
            selected_block_shape_map.insert(std::make_pair(name, best_solution->block_shapes.at(name)));

            if (best_solution->output_host_tms.count(name) > 0)
                selected_output_host_tms.insert(std::make_pair(name, best_solution->output_host_tms.at(name)));

            for (graphlib::Node* input : current_graph->data_operands(current_graph->get_node_by_name(name)))
            {
                if (input->node_type() == graphlib::kInput)
                {
                    selected_op_models.insert(
                        std::make_pair(input->name(), best_solution->op_models.at(input->name())));
                    selected_block_shape_map.insert(
                        std::make_pair(input->name(), best_solution->block_shapes.at(input->name())));
                }
            }
            for (graphlib::Node* output : current_graph->data_users(current_graph->get_node_by_name(name)))
            {
                if (output->node_type() == graphlib::kOutput)
                {
                    selected_block_shape_map.insert(
                        std::make_pair(output->name(), best_solution->block_shapes.at(output->name())));
                }
            }
        }

        if (first_epoch)
            placer_solution = best_solution->placer_solution;
        else
            placer_solution.merge(best_solution->placer_solution);  // destroys the original

        // placeholder
        placer_solution.epoch_id_to_chip[history.current_epoch()] = 0;

        history.next_epoch();
        first_epoch = false;

        Graph* modified_graph = best_placement.release_graph();
        if (modified_graph)
        {
            *graph = modified_graph;
            current_graph = modified_graph;
        }

        ischeduler.restore_checkpoint(best_placement.scheduler_checkpoint());

        while (ischeduler.done())
        {
            // TODO: simple progression for now
            if (current_epoch_type == graphlib::NodeEpochType::Forward)
            {
                current_epoch_type = graphlib::NodeEpochType::Backward;
                ischeduler.set_epoch_type(current_epoch_type);
            }
            else if (current_epoch_type == graphlib::NodeEpochType::Backward)
            {
                current_epoch_type = graphlib::NodeEpochType::Optimizer;
                ischeduler.set_epoch_type(current_epoch_type);
            }
            else
            {
                // Add input queues to the placer solution
                insert_input_queues(placer_solution, current_graph, selected_op_models);

                // Assign chips (todo)

                return std::make_shared<balancer::BalancerSolution>(
                    placer_solution, selected_op_models, selected_block_shape_map, selected_output_host_tms, cut_edges);
            }
        }
    }
}

}  // namespace tt::placer
