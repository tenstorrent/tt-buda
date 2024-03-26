// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "balancer/legalizer/graph_solver.hpp"
#include "passes/fork_join.hpp"
#include "placer/interactive_placer.hpp"

namespace tt::balancer
{

class EpochSolution;

// Class that decouples interactive placer and graphsolver from balancer policy logic.
// Op placement, epoch switches and inline fork-join buffering are handled by PolicyManager.
//
class PolicyManager
{
    graphlib::Graph const* graph;
    BalancerConfig config;
    placer::InteractivePlacer interactive_placer;
    placer::InteractivePlacer interactive_placer_tester;
    std::unordered_set<string> epoch_break_ops;
    std::unordered_set<string> chip_break_ops;
    scheduler::Schedule scheduled_ops;
    graphlib::NodeEpochType current_epoch_type = NodeEpochType::Forward;
    std::uint32_t last_epoch_start = 0;
    std::uint32_t op_index = 0;
    std::uint32_t op_nodes_to_process = 0;
    std::unordered_set<const tt::graphlib::Node*> current_epoch_ops;
    std::vector<const tt::graphlib::Node*> pre_buffering_epoch_ops;
    tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash> inst;
    const tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        prev_inst;
    std::unordered_set<const tt::graphlib::Node*> processed_nodes;
    tt::scheduler::Schedule processed_schedule;
    tt::scheduler::Schedule epoch_schedule;
    std::vector<EpochSolution> epoch_solutions;
    std::vector<OpModel> current_epoch_selected_models;
    std::vector<tt::scheduler::Schedule> op_names_to_epoch_break;
    std::vector<tt::scheduler::Schedule> op_names_to_chip_break;
    bool overflow_set_for_epoch = false;
    bool use_interactive_fj_buffering = !env_as<bool>("PYBUDA_DISABLE_INTERACTIVE_FJ_BUFFERING", false);
    std::unique_ptr<legalizer::GraphSolver> graph_solver_main;
    std::unique_ptr<legalizer::GraphSolver> graph_solver_epoch_snapshot;
    std::unique_ptr<legalizer::GraphSolver> graph_solver_buffering_snapshot;
    std::unique_ptr<legalizer::GraphSolver> graph_solver_pairing_checkpoint;
    std::unique_ptr<graphlib::GraphTraversalContext> traversal_context;
    std::optional<OpModel> buffered_op_model;
    const Node* current_op_node = nullptr;
    bool try_transpose_op = true;

    // Section for ribbon like policies.
    //
    bool ribbon_policy = false;

    // Number of rows in the ribbon. Should only change when matmul with a different R dim is encountered.
    // Scheduler needs to ensure that we don't go to lower res and then jump back to bigger, wherever possible to avoid
    // it.
    std::uint32_t current_ribbon_size = 0;
    std::uint32_t current_matmul_dim_r = 0;
    std::uint32_t epoch_start_matmul_dim_r = 0;
    std::uint32_t next_ribbon_change = 0;
    std::uint32_t epoch_start_ribbon_size = 0;
    void update_ribbon_size();

    void register_op_in_current_epoch(const OpModel& selected_op_model);
    bool buffer_epoch();
    bool handle_epoch_buffering_overflow();
    void start_new_epoch(graphlib::NodeEpochType epoch_type);
    void pair_two_ops_if_possible(
        const graphlib::BudaOpNode* op,
        const OpModel& selected_op_model,
        std::optional<placer::CoordRange>& op_placement,
        bool& op_already_set,
        bool& skip_placing);

   public:
    PolicyManager(
        graphlib::Graph const* graph,
        BalancerConfig const& config,
        legalizer::GraphSolver& graph_solver,
        bool ribbon_policy = false);

    // Main interfaces.
    //
    const graphlib::Node* get_next_op();
    std::tuple<bool, bool, bool> commit_op(const OpModel& selected_op_model);
    bool finish_current_epoch();
    BalancerPolicySolution commit_solution();
    void rewind_epoch();
    bool force_current_epoch_break(const std::string& op_name);

    // Graph solver interface.
    //
    legalizer::GraphSolver::RemainingOpModels at(const tt::graphlib::Node* node) const
    {
        return graph_solver_main->at(node);
    }

    void invalidate_suboptimal_op_models(int invalidation_strategy)
    {
        graph_solver_main->invalidate_suboptimal_op_models(invalidation_strategy);
    }

    // Simple getters.
    //
    std::uint32_t get_current_ribbon_size() const { return current_ribbon_size; }
    std::uint32_t get_current_epoch_index() const { return interactive_placer.get_current_epoch_index(); }
    std::uint32_t get_current_epoch_size() const { return interactive_placer.current_epoch_size(); }

    // Currently balanced OP.
    //
    const graphlib::Node* get_current_op() { return current_op_node; }
};
}  // namespace tt::balancer
