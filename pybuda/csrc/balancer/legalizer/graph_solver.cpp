// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/legalizer/graph_solver.hpp"

#include <limits>

#include "balancer/balancer_utils.hpp"
#include "balancer/legalizer/constraints.hpp"
#include "balancer/legalizer/legalizer.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "reportify/reportify.hpp"
#include "utils/assert.hpp"

namespace tt::balancer::legalizer
{
GraphSolver::Bitset GraphSolver::kBitsetAll = ~kBitsetNone;

#ifdef DEBUG
void log_op_model_info(Node* producer_node, Node* consumer_node)
{
    graphlib::BudaOpNode* producer_op_node = dynamic_cast<graphlib::BudaOpNode*>(producer_node);
    graphlib::BudaOpNode* consumer_op_node = dynamic_cast<graphlib::BudaOpNode*>(consumer_node);

    if (producer_op_node)
    {
        log_debug(
            LogGraphSolver,
            "OpModel failure statistics for producer node: {} {} {}",
            producer_op_node->name(),
            producer_op_node->get_type(),
            producer_op_node->shape());
        log_debug(LogGraphSolver, producer_op_node->leg_debug_info->toString().c_str());
    }

    if (consumer_op_node)
    {
        log_debug(
            LogGraphSolver,
            "OpModel failure statistics for consumer node: {} {} {}",
            consumer_node->name(),
            consumer_node->get_type(),
            consumer_node->shape());
        log_debug(LogGraphSolver, consumer_op_node->leg_debug_info->toString().c_str());
    }
}
#endif

void GraphSolver::reset(bool partial_reset_allowed)
{
    path_sets.clear();
    path_set_ids.clear();
    failure_reasons.clear();
    if (!partial_reset_allowed)
    {
        bitsets.clear();
        bitset_ids.clear();
    }

#ifdef DEBUG
    if (env_as<bool>("PYBUDA_LEGALIZER_DETAILED_DEBUGGING"))
    {
        // Cleanup debug data in OpModels.
        //
        for (auto& mapElem : shared_data->legal_op_models)
        {
            std::vector<OpModel>& opModels = mapElem.second;

            for (OpModel& model : opModels)
            {
                model.eliminating_edge = graphlib::EdgeUniqueId();
            }
        }

        for (const auto& verElem : op_model_recompute_version)
        {
            std::vector<OpModel>& opModels = shared_data->recomputed_legal_op_models.at(verElem.first)[verElem.second];

            for (OpModel& model : opModels)
            {
                model.eliminating_edge = graphlib::EdgeUniqueId();
            }
        }
    }
#endif
}

std::pair<EdgeCost, ConstraintFailureReason> cost_fn(
    Constraint* constraint,
    graphlib::Graph const* graph,
    graphlib::Edge const& edge,
    std::vector<OpModel> const& producer_op_models,
    std::vector<OpModel> const& consumer_op_models,
    std::uint64_t producer_id,
    std::uint64_t consumer_id)
{
    graphlib::Node const* producer = graph->node_by_id(edge.producer_node_id);
    graphlib::Node const* consumer = graph->node_by_id(edge.consumer_node_id);
    static OpModel null_op_model;
    if (producer->node_type() != graphlib::NodeType::kBudaOp)
    {
        TT_ASSERT(consumer_id < consumer_op_models.size());
        return constraint->queue_to_op_cost(
            graph,
            edge,
            producer_op_models.size() > 0 ? producer_op_models[producer_id] : std::optional<OpModel>{},
            consumer_op_models[consumer_id]);
    }
    else if (consumer->node_type() != graphlib::NodeType::kBudaOp)
    {
        TT_ASSERT(producer_id < producer_op_models.size());
        return constraint->op_to_queue_cost(
            graph,
            edge,
            producer_op_models[producer_id],
            consumer_op_models.size() > 0 ? consumer_op_models[consumer_id] : std::optional<OpModel>{});
    }
    else
    {
        TT_ASSERT(producer_id < producer_op_models.size());
        TT_ASSERT(consumer_id < consumer_op_models.size());
        return constraint->op_to_op_cost(graph, edge, producer_op_models[producer_id], consumer_op_models[consumer_id]);
    }
}

// Generate unique id of op model pair.
//
static std::uint64_t get_op_model_pair_id(const std::uint64_t prod_om_id, const std::uint64_t cons_om_id)
{
    return (size_t(prod_om_id) << 32llu) | size_t(cons_om_id);
}

bool GraphSolver::resolve_step(const bool self_cut_allowed)
{
#ifdef DEBUG
    EdgeConstraintDebugInfo graph_constraint_debug_info;
    bool enable_legalizer_detailed_debugging = env_as<bool>("PYBUDA_LEGALIZER_DETAILED_DEBUGGING");
    std::string node_name_edge_debug = env_as<std::string>("PYBUDA_LEGALIZER_DEBUG_NODE_NAME");
    bool collect_failure_reasons = env_as<bool>("PYBUDA_COLLECT_CONSTRAINT_INFO");
#endif

    Constraint* constraint = shared_data->constraint.get();
    NodePathsProcessor node_processor;
    std::vector<graphlib::Node*> nodes = graphlib::topological_sort(*graph);
    bitsets.reserve(nodes.size());
    bitset_ids.reserve(nodes.size());
    op_disabled_bitset_cache.reserve(nodes.size());
    selected_op_models.reserve(nodes.size());
    bool fast_cut_used = false;  // Self-cutting is performed in a single graphsolver pass, followed by one more final
                                 // graphsolver resolution.
    bool consumer_op_model_exceeds = false;
    bool producer_op_model_exceeds = false;

    std::vector<int> self_cut_disabled_on_subgraphs = env_as_vector<int>("PYBUDA_DISABLE_SELF_CUT_FOR_SUBGRAPHS");

    for (graphlib::Node* consumer_node : nodes)
    {
        Bitset* consumer_bitset = get_or_insert_bitset(consumer_node->id(), kBitsetAll);
        std::vector<OpModel> const& consumer_op_models = get_legal_op_models(consumer_node);

        for (graphlib::Edge edge : graph->operand_data_edges(consumer_node))
        {
#ifdef DEBUG
            EdgeConstraintDebugInfo edge_constraint_debug_info;
#endif
            // With virtual queue processing on cut edges within GraphTraversalContext,
            // we shouldn't be processing cut edges anymore unless we are in fast cut.
            //
            if (cut_edges.count(edge) > 0)
            {
                TT_ASSERT(edges_to_ignore.count(edge) > 0, "Cut edge must be ignored!");
                TT_ASSERT(fast_cut_used, "If we are processing cut edge this must be fast cut!");

                continue;
            }

            graphlib::Node* producer_node = graph->node_by_id(edge.producer_node_id);
            Bitset* producer_bitset = get_or_insert_bitset(producer_node->id(), kBitsetAll);
            std::vector<OpModel> const& producer_op_models = get_legal_op_models(producer_node);

            TT_ASSERT(not(consumer_op_models.empty() and producer_op_models.empty()));

            if (consumer_op_models.size() > kNumBitsetBits)
            {
                consumer_op_model_exceeds = true;
                log_trace(
                    LogGraphSolver,
                    "Consumer op models [{}] exceeds kNumBitsetBits [{}] node {}",
                    consumer_op_models.size(),
                    kNumBitsetBits,
                    consumer_node->name());
            }

            if (producer_op_models.size() > kNumBitsetBits)
            {
                producer_op_model_exceeds = true;
                log_trace(
                    LogGraphSolver,
                    "Producer op models [{}] exceeds kNumBitsetBits [{}] node {}",
                    producer_op_models.size(),
                    kNumBitsetBits,
                    producer_node->name());
            }

            PathSet::Paths paths;
            Bitset edge_producer_bitset = kBitsetNone;
            Bitset edge_consumer_bitset = kBitsetNone;
            std::uint64_t producer_count = std::min(kNumBitsetBits, std::max(1lu, producer_op_models.size()));
            std::uint64_t consumer_count = std::min(kNumBitsetBits, std::max(1lu, consumer_op_models.size()));
            bool cacheable = producer_node->node_type() == graphlib::NodeType::kBudaOp and
                             consumer_node->node_type() == graphlib::NodeType::kBudaOp;
            for (std::uint64_t producer_id = 0; producer_id < producer_count; ++producer_id)
            {
                // If the producer cannot accomodate this path, continue.
                // Also if this is not the OpModel we selected, continue.
                //
                if (!producer_bitset->test(producer_id))
                    continue;

                for (std::uint64_t consumer_id = 0; consumer_id < consumer_count; ++consumer_id)
                {
                    // If the consumer cannot accomodate this path, continue.
                    // Also if this is not the OpModel we selected, continue.
                    //
                    if (!consumer_bitset->test(consumer_id))
                        continue;

                    // Load constraint check result from cache for Op-Op verification if possible,
                    // otherwise populate cache.
                    //
                    std::uint64_t pair_id = 0;
                    std::unordered_map<std::uint64_t, const std::pair<const EdgeCost, const ConstraintFailureReason>>::
                        const_iterator cache_it;
                    EdgeCost cost;
                    ConstraintFailureReason constraint_failure_reason;

                    if (cacheable)
                    {
                        pair_id = get_op_model_pair_id(
                            producer_op_models[producer_id].id.id, consumer_op_models[consumer_id].id.id);
                        cache_it = shared_data->constraint_result_cache.find(pair_id);
                    }

                    if (!cacheable or cache_it == shared_data->constraint_result_cache.end())
                    {
                        std::tie(cost, constraint_failure_reason) = cost_fn(
                            constraint, graph, edge, producer_op_models, consumer_op_models, producer_id, consumer_id);
                        if (cacheable)
                        {
                            shared_data->constraint_result_cache.try_emplace(pair_id, cost, constraint_failure_reason);
                        }
                    }
                    else
                    {
                        std::tie(cost, constraint_failure_reason) = cache_it->second;
                    }

                    if (NoConstraintFailure == constraint_failure_reason)
                    {
                        if (not cost.exceeded())
                        {
                            TT_ASSERT(producer_id <= std::numeric_limits<decltype(Path::producer_id)>::max());
                            TT_ASSERT(consumer_id <= std::numeric_limits<decltype(Path::consumer_id)>::max());
                            paths.push_back(Path(producer_id, consumer_id, cost));
                            edge_producer_bitset.set(producer_id);
                            edge_consumer_bitset.set(consumer_id);
                        }
                        else
                        {
                            constraint_failure_reason = MaxCostExceeded;
                        }
                    }
#ifdef DEBUG
                    else if (
                        collect_failure_reasons and not producer_op_models.empty() and not consumer_op_models.empty())
                    {
                        std::string key = fmt::format(
                            "{}:{}", producer_op_models[producer_id].id.id, consumer_op_models[consumer_id].id.id);
                        failure_reasons.insert({key, constraint_failure_reason});
                    }

                    edge_constraint_debug_info.recordEdgeConstraintFailure(constraint_failure_reason);
                    graph_constraint_debug_info.recordEdgeConstraintFailure(constraint_failure_reason);
#endif
                }
            }

#ifdef DEBUG
            if (enable_legalizer_detailed_debugging)
            {
                compute_edge_elimination_debug_info(
                    edge,
                    producer_bitset,
                    consumer_bitset,
                    edge_producer_bitset,
                    edge_consumer_bitset,
                    const_cast<std::vector<OpModel>&>(producer_op_models),
                    const_cast<std::vector<OpModel>&>(consumer_op_models),
                    producer_count,
                    consumer_count,
                    edge_constraint_debug_info,
                    graph_constraint_debug_info);
            }
#endif

            if (paths.empty() or ((*producer_bitset & edge_producer_bitset) == 0) or
                ((*consumer_bitset & edge_consumer_bitset) == 0))
            {
#ifdef DEBUG
                // If we fail print whole graph edge constraint statistics, and statistics for this edge.
                // If you enable detailed legalizer debugging "PYBUDA_LEGALIZER_DETAILED_DEBUGGING"
                // you will also get edge elimination stats and stats of OpModels for both nodes on this edge.
                //
                log_debug(LogGraphSolver, "Constraint failure statistics for whole graph:");
                log_debug(LogGraphSolver, graph_constraint_debug_info.toString().c_str());
                log_debug(
                    LogGraphSolver,
                    "Constraint failure statistics for egde: {} -> {}",
                    producer_node->name(),
                    consumer_node->name());
                log_debug(LogGraphSolver, edge_constraint_debug_info.toString(graph).c_str());
                if (enable_legalizer_detailed_debugging)
                {
                    log_op_model_info(producer_node, consumer_node);
                }
#endif

                // No valid paths found for this edge, lets try self-cutting if enabled.
                //
                if (GraphSolverSelfCutType::None != balancer_config.graph_solver_self_cut_type and self_cut_allowed and
                    producer_node->node_type() == graphlib::NodeType::kBudaOp and
                    consumer_node->node_type() == graphlib::NodeType::kBudaOp and
                    (std::find(
                         self_cut_disabled_on_subgraphs.begin(),
                         self_cut_disabled_on_subgraphs.end(),
                         graph->get_subgraph_id_for_node(producer_node->id())) == self_cut_disabled_on_subgraphs.end()))
                {
                    fast_cut_used = self_cut(producer_node, consumer_node);

                    if (fast_cut_used)
                    {
                        *consumer_bitset = kBitsetAll;
                        continue;
                    }
                    else
                    {
                        return false;
                    }
                }

                throw_error_for_edge(edge);
            }

#ifdef DEBUG
            if (enable_legalizer_detailed_debugging)
            {
                if (node_name_edge_debug == producer_node->name() or node_name_edge_debug == consumer_node->name() or
                    node_name_edge_debug.empty())
                {
                    log_debug(
                        LogGraphSolver,
                        "Constraint failure statistics for egde: {} -> {}",
                        producer_node->name(),
                        consumer_node->name());
                    log_debug(LogGraphSolver, edge_constraint_debug_info.toString(graph).c_str());

                    if (!node_name_edge_debug.empty())
                    {
                        log_op_model_info(producer_node, consumer_node);
                    }
                }
            }
#endif
            if (!is_subset(*producer_bitset, edge_producer_bitset) && !fast_cut_used)
            {
                node_processor.add_node(producer_node);
            }

            *producer_bitset &= edge_producer_bitset;
            *consumer_bitset &= edge_consumer_bitset;
            TT_ASSERT(path_set_ids.find(edge) == path_set_ids.end());
            PathSetId path_set_id = (PathSetId)path_sets.size();
            path_sets.emplace_back(
                bitset_ids[producer_node->id()], bitset_ids[consumer_node->id()], producer_node, consumer_node, paths);
            path_set_ids.emplace(edge, path_set_id);
        }

        if (!fast_cut_used)
        {
            node_processor.process(this);
        }
    }

    if (consumer_op_model_exceeds)
    {
        log_warning(
            LogGraphSolver,
            "Consumer op models exceed kNumBitsetBits requirements for some nodes, check trace for detail");
    }

    if (producer_op_model_exceeds)
    {
        log_warning(
            LogGraphSolver,
            "Producer op models exceed kNumBitsetBits requirements for some nodes, check trace for detail");
    }

    if (fast_cut_used)
    {
        return false;
    }

#ifdef DEBUG
    log_debug(LogGraphSolver, "Constraint failure statistics for whole graph:");
    log_debug(LogGraphSolver, graph_constraint_debug_info.toString().c_str());
#endif

    for (graphlib::Node* node : nodes)
    {
        // No need to expand root as we are calling for all nodes anyway.
        //
        update_solver(node, false /* expand_root */);
    }

    resolve_step_postprocess(nodes);

    return true;
}

// Used for tweaking output of graphsolver resolve.
//
void GraphSolver::resolve_step_postprocess(const std::vector<graphlib::Node*>& nodes)
{
    // Invalidate streaming into output if possible(enabled by default).
    //
    invalidate_streaming_into_output(nodes);

    // Invalidate suboptimal op models according to invalidation strategy.
    //
    if (suboptimal_opmodel_invalidation_strategy)
    {
        invalidate_suboptimal_op_models(nodes);
    }
}

// Self-cutting is used when we cannot find valid path for an edge due to constraints. Then as last resort so that we
// can resolve this graph we mark this edge as a virtual one and in later phase we will place a queue in its place
// between producer and consumer. Returns whether fast cut should be used - instead of resolving whole graph after each
// cut, compute all cuts first(approximation which may lead to more cuts).
//
bool GraphSolver::self_cut(graphlib::Node* producer_node, graphlib::Node* consumer_node)
{
    std::unordered_set<graphlib::Node*> nodes_to_legalize;
    bool use_fast_cut = false;

    switch (balancer_config.graph_solver_self_cut_type)
    {
        case FastCut: use_fast_cut = true; [[fallthrough]];
        case ConsumerOperandDataEdgesFirst:
        {
            log_debug(
                LogGraphSolver,
                "Constraint failure - trying to resolve by self-cutting all operand data edges for node {}",
                consumer_node->name());

            for (graphlib::Edge edge : graph->operand_data_edges(consumer_node))
            {
                graphlib::Node* producer_node = graph->node_by_id(edge.producer_node_id);

                if (producer_node->node_type() == graphlib::NodeType::kBudaOp)
                {
                    TT_ASSERT(cut_edges.count(edge) == 0, "Same edge should not be cut twice!");
                    TT_ASSERT(
                        selected_op_models.count(producer_node) == 0 or selected_op_models.count(consumer_node) == 0,
                        "At least one node affected by CUT must not be SET!");
                    if (selected_op_models.count(producer_node) == 0)
                    {
                        nodes_to_legalize.insert(producer_node);
                    }

                    if (selected_op_models.count(consumer_node) == 0)
                    {
                        nodes_to_legalize.insert(consumer_node);
                    }

                    cut_edges.insert(std::make_pair(edge, true /* self cutting edge */));

                    // Insert virtual queue on cut edge.
                    //
                    insert_virtual_queue(edge, producer_node, consumer_node);
                }
            }
        }
        break;

        case ProducerUserDataEdgesFirst:
        {
            log_debug(
                LogGraphSolver,
                "Constraint failure - trying to resolve by self-cutting all user data edges for node {}",
                producer_node->name());

            for (graphlib::Edge edge : graph->user_data_edges(producer_node))
            {
                graphlib::Node* consumer_node = graph->node_by_id(edge.consumer_node_id);

                if (consumer_node->node_type() == graphlib::NodeType::kBudaOp)
                {
                    TT_ASSERT(cut_edges.count(edge) == 0, "Same edge should not be cut twice!");
                    TT_ASSERT(
                        selected_op_models.count(producer_node) == 0 or selected_op_models.count(consumer_node) == 0,
                        "At least one node affected by CUT must not be SET!");
                    if (selected_op_models.count(producer_node) == 0)
                    {
                        nodes_to_legalize.insert(producer_node);
                    }

                    if (selected_op_models.count(consumer_node) == 0)
                    {
                        nodes_to_legalize.insert(consumer_node);
                    }

                    cut_edges.insert(std::make_pair(edge, true /* self cutting edge */));

                    // Insert virtual queue on cut edge.
                    //
                    insert_virtual_queue(edge, producer_node, consumer_node);
                }
            }
        }
        break;

        case None:
        default: TT_ASSERT(false, "Invalid self cut type!"); break;
    }

    // Recalculate OpModels for nodes affected by queue insertion.
    //
    recompute_legal_op_models_on_cut(nodes_to_legalize);
    return use_fast_cut;
}

void GraphSolver::resolve(bool partial_reset_allowed)
{
    PROFILE_SCOPE();
    graphlib::GraphTraversalContext graph_solver_graph_context(graph, &virtual_nodes, &edges_to_ignore);
    int default_resolve_retry_count_self_cutting = 20;
    if (env_as<int>("PYBUDA_MAX_GRAPH_CUT_RETRY", 0))
    {
        default_resolve_retry_count_self_cutting = env_as<int>("PYBUDA_MAX_GRAPH_CUT_RETRY");
    }
    const int max_retry_step = GraphSolverSelfCutType::None != balancer_config.graph_solver_self_cut_type
                                   ? default_resolve_retry_count_self_cutting
                                   : 1;
    int retry_step = 1;
    bool resolved = false;

    do
    {
        // Reset GraphSolver to default state.
        //
        reset(partial_reset_allowed);

        // Try to resolve graph(currently retry is used only by self-cutting mechanism).
        // Self-cutting means graph will cut edge for which it cannot produce valid paths and retry resolve again.
        //
        resolved = resolve_step(retry_step < max_retry_step);
        retry_step++;
    } while (!resolved and retry_step <= max_retry_step);

    if (!resolved and env_as<bool>("PYBUDA_COLLECT_CONSTRAINT_INFO"))
    {
        update_constraint_info();
    }

    TT_ASSERT(resolved, "Graph is either resolved or error is thrown from resolve_step.");

#ifdef DEBUG
    if (cut_edges.size() > 0)
    {
        log_debug(LogGraphSolver, "Graph is resolved with cut edges: ");
        for (auto& it : cut_edges)
        {
            const Edge& edge = it.first;
            std::string producerNodeName = graph->node_by_id(edge.producer_node_id)->name();
            std::string consumerNodeName = graph->node_by_id(edge.consumer_node_id)->name();
            std::string selfCutEdge = it.second ? "Self-cut " : "";
            log_debug(LogGraphSolver, "{}Edge: {} -> {}", selfCutEdge, producerNodeName, consumerNodeName);
        }
    }
#endif
}

std::vector<graphlib::Edge> GraphSolver::get_epoch_type_switch_cut_edges()
{
    std::vector<graphlib::Edge> epoch_to_epoch_cuts;
    for (graphlib::Node* consumer : graphlib::topological_sort(*graph))
    {
        if (consumer->node_type() != graphlib::NodeType::kBudaOp)
            continue;
        for (graphlib::Edge edge : graph->operand_data_edges(consumer))
        {
            graphlib::Node* producer = graph->node_by_id(edge.producer_node_id);
            if (producer->node_type() != graphlib::NodeType::kBudaOp)
                continue;
            if (producer->get_epoch_type() != consumer->get_epoch_type())
            {
                epoch_to_epoch_cuts.push_back(edge);
            }
        }
    }
    return epoch_to_epoch_cuts;
}

GraphSolver::GraphSolver(
    graphlib::Graph* graph,
    std::unique_ptr<Constraint>&& constraint,
    LegalOpModels const& legal_op_models,
    BalancerConfig const& balancer_config,
    std::shared_ptr<balancer::BalancerCacheCollection> balancer_cache_collection,
    std::vector<graphlib::Edge> const& in_cut_edges,
    bool use_op_model_recalculation_on_cut,
    bool resolve_on_create) :
    graph(graph),
    shared_data(std::make_shared<SharedData>(std::move(constraint), legal_op_models)),
    balancer_config(balancer_config),
    balancer_cache_collection(balancer_cache_collection),
    use_op_model_recalculation_on_cut(use_op_model_recalculation_on_cut)
{
    std::size_t num_edges = graph->operands_map().size();
    uint op_model_pairs_per_edge_estimate = 256;
    path_sets.reserve(num_edges);
    path_set_ids.reserve(num_edges);
    shared_data->constraint_result_cache.reserve(num_edges * op_model_pairs_per_edge_estimate);

    single_core_ip_mode =
        balancer_config.device_config.grid_size.r * balancer_config.device_config.grid_size.c == 1 and
        balancer_config.use_interactive_placer and
        (balancer_config.policy_type == PolicyType::NLP || balancer_config.policy_type == PolicyType::Ribbon);

    if (env_as<bool>("PYBUDA_COLLECT_CONSTRAINT_INFO"))
    {
        constraint_info_ptr = std::make_shared<ConstraintInfo>();
    }

    if (resolve_on_create)
    {
        std::vector<graphlib::Edge> initial_cuts = get_epoch_type_switch_cut_edges();
        initial_cuts.insert(initial_cuts.end(), in_cut_edges.begin(), in_cut_edges.end());

        if (initial_cuts.empty())
            resolve();
        else
            cut(initial_cuts, true /*epoch_cut*/);
    }
}

GraphSolver::PathSet& GraphSolver::get_path_set(const graphlib::Edge& edge)
{
    TT_ASSERT(
        path_set_ids.find(edge) != path_set_ids.end(),
        graph->node_by_id(edge.producer_node_id)->name(),
        graph->node_by_id(edge.consumer_node_id)->name());
    return path_sets[path_set_ids.at(edge)];
}

GraphSolver::PathSet const& GraphSolver::get_path_set(const graphlib::Edge& edge) const
{
    return path_sets[path_set_ids.at(edge)];
}

GraphSolver::PathSet* GraphSolver::get_path_set_pt(const graphlib::Edge& edge)
{
    if (path_set_ids.count(edge) > 0)
    {
        return &path_sets[path_set_ids.at(edge)];
    }
    else
    {
        return nullptr;
    }
}

SmallVector<GraphSolver::PathSet*> GraphSolver::get_operand_path_sets_pts(graphlib::Node const* node)
{
    SmallVector<PathSet*> operand_path_sets;
    for (auto edge : graph->operand_data_edges(node))
    {
        PathSet* el = get_path_set_pt(edge);
        if (nullptr != el)
        {
            operand_path_sets.push_back(el);
        }
    }

    return operand_path_sets;
}

SmallVector<GraphSolver::PathSet*> GraphSolver::get_user_path_sets_pts(graphlib::Node const* node)
{
    SmallVector<PathSet*> user_path_sets;
    for (auto edge : graph->user_data_edges(node))
    {
        PathSet* el = get_path_set_pt(edge);
        if (nullptr != el)
        {
            user_path_sets.push_back(el);
        }
    }

    return user_path_sets;
}

void GraphSolver::log_bitset(graphlib::Node const* node, const Bitset& set) const
{
    const std::vector<OpModel>& op_models = get_legal_op_models(node);
    log_debug(
        LogGraphSolver, "    {} {}:{}", node->name(), node->node_type(), !op_models.empty() ? "" : " No op models");

    TT_ASSERT(
        set.count() <= op_models.size(),
        "set.count() = {} is NOT <= op_models.size() = {}",
        set.count(),
        op_models.size());
    for (std::uint64_t i = 0; i < op_models.size() and i < set.size(); i++)
    {
        if (set[i])
            log_debug(
                LogGraphSolver, "      [{}] {} {}", i, op_models.at(i).grid_shape, op_models.at(i).t_stream_factor);
    }
}

template <bool kOperand>
void GraphSolver::handle_cumulative_paths_error(
    PathSet const& path_set,
    const Bitset& debug_snapshot,
    graphlib::Node const* producer,
    graphlib::Node const* consumer)
{
    auto user_data_edges = graph->user_data_edges(producer);
    if (not kOperand and user_data_edges.size() > 2)
    {
        throw BalancerError(
            fmt::format("Node forks {}", producer->name()),
            BalancerError::NodeExceedsMaxOpForks(round_up_div(user_data_edges.size(), std::size_t{2}), producer->id()));
    }

    //
    // Fail with error message
    //
    const std::string msg =
        fmt::format("Could not reconcile constraints: path[{} -> {}]", producer->name(), consumer->name());
    log_debug(LogGraphSolver, "{}", msg);

    if (kOperand)
    {
        log_debug(LogGraphSolver, "  Offending Producer info:");
        log_bitset(producer, path_set.get_producer_set(bitsets));

        log_debug(LogGraphSolver, "  Consumer info:");
        log_bitset(consumer, debug_snapshot);
    }
    else
    {
        log_debug(LogGraphSolver, "  Producer info:");
        log_bitset(producer, debug_snapshot);

        log_debug(LogGraphSolver, "  Offending Consumer info:");
        log_bitset(consumer, path_set.get_consumer_set(bitsets));
    }

    update_constraint_info();
    reportify::dump_constraints(graph->name(), this);

    throw BalancerError(msg, BalancerError::Fatal(msg));
}

template <bool kOperand, typename CostFns>
bool GraphSolver::apply_cumulative_costs(
    SmallVector<tt::balancer::legalizer::GraphSolver::PathSet*> const& path_sets,
    graphlib::Node const* node,
    CostFns cost_fns)
{
    // cull out user paths who's sum exceeds the max cost
    bool path_changed = false;
    SmallVector<Bitset> debug_snapshot;
    for (int i = 0; i < (int)path_sets.size(); ++i)
    {
        debug_snapshot.push_back(
            kOperand ? path_sets[i]->get_consumer_set(bitsets) : path_sets[i]->get_producer_set(bitsets));
    }

    for (auto [sort_fn, sum_fn] : cost_fns())
    {
        for (int i = 0; i < (int)path_sets.size(); ++i)
        {
            auto path_set_i = path_sets[i];

            auto max_path = path_set_i->max_cost(sort_fn);
            if (not max_path)
            {
                if (kOperand)
                    handle_cumulative_paths_error<kOperand>(
                        *path_set_i, debug_snapshot[i], path_set_i->get_producer_node(), node);
                else
                    handle_cumulative_paths_error<kOperand>(
                        *path_set_i, debug_snapshot[i], node, path_set_i->get_consumer_node());
            }

            std::uint16_t max_path_index = kOperand ? max_path->consumer_id : max_path->producer_id;
            EdgeCost total_cost = max_path->cost;
            bool valid = true;

            for (int j = 0; j < (int)path_sets.size(); ++j)
            {
                auto path_set_j = path_sets[j];

                if (i == j)
                    continue;

                auto min_path = path_set_j->min_cost<kOperand>(sort_fn, max_path_index);
                if (not min_path)
                {
                    valid = false;
                    break;
                }

                total_cost = sum_fn(total_cost, min_path->cost);
            }

            if (total_cost.exceeded() or not valid)
            {
                path_changed |= path_set_i->erase(max_path, bitsets);
                --i;  // retry
            }
        }
    }

    return path_changed;
}

void add_operands_and_users(
    const graphlib::Graph* graph,
    const graphlib::Node* node,
    std::vector<graphlib::Node const*>& needs_update,
    const graphlib::Node* ignore_node = nullptr)
{
    for (graphlib::Node* node_o : graph->data_operands(node))
    {
        if (node_o == ignore_node)
            continue;

        needs_update.push_back(node_o);
    }

    for (graphlib::Node* node_u : graph->data_users(node))
    {
        if (node_u == ignore_node)
            continue;

        needs_update.push_back(node_u);
    }
}

void GraphSolver::handle_no_paths_left_on_update(
    bool invoked_by_set, const std::string& root_node_name, const std::string& current_node_name)
{
    // We ended-up in a situation without valid solution due to circular dependency.
    //
    if (invoked_by_set)
    {
        // Invoking resolve again will use self-cut to try to resolve this issue.
        //
        log_debug(
            LogGraphSolver,
            "Update solver failed for root node {} on node {}. Invoking re-resolve!",
            root_node_name,
            current_node_name);
        return resolve();
    }
    else
    {
        // Already in resolve, we can only error out at this point.
        //
        const std::string msg = fmt::format(
            "Update solver failed to reconcile constraints invoked for root node {} on node {}!",
            root_node_name,
            current_node_name);
        throw BalancerError(msg, BalancerError::Fatal(msg));
    }
}

void GraphSolver::update_solver(graphlib::Node const* root, bool expand_root, bool invoked_by_set)
{
    TT_ASSERT(
        graph->virtual_node_count() == 0 or graph->is_graph_traversal_context_set(),
        "We have virtual nodes and no graph traversal context set - this could lead to unexpected results in GS graph "
        "resolution.");
    std::vector<graphlib::Node const*> needs_update = {root};

    if (expand_root)
    {
        auto operand_path_sets = get_operand_path_sets_pts(root);
        auto user_path_sets = get_user_path_sets_pts(root);

        for (auto path_set : operand_path_sets)
        {
            path_set->update(bitsets);
        }

        for (auto path_set : user_path_sets)
        {
            path_set->update(bitsets);
        }

        // When node bitsets are updated(set of valid op models), we need to update paths for all operands and users.
        //
        add_operands_and_users(graph, root, needs_update);
    }

    // Iterate through the nodes that need to be updated and update their operand and user path sets.
    while (not needs_update.empty())
    {
        auto node = needs_update.back();

        // Get path sets for incoming edges
        auto operand_path_sets = get_operand_path_sets_pts(node);
        // Get path sets for outgoing edges
        auto user_path_sets = get_user_path_sets_pts(node);

        bool path_changed = false;
        bool edge_changed = false;

        std::vector<bool> producers_changed(operand_path_sets.size());
        for (size_t i = 0; i < operand_path_sets.size(); i++)
        {
            auto operand_path_set = operand_path_sets[i];
            producers_changed[i] = operand_path_set->update(bitsets);

            if (operand_path_set->empty(bitsets))
            {
                return handle_no_paths_left_on_update(invoked_by_set, root->name(), node->name());
            }
        }

        // Cumulative cost of operand edges coming into this consumer
        path_changed |= apply_cumulative_costs<true>(operand_path_sets, node, EdgeCost::consumer_cost_fns);

        std::vector<bool> consumers_changed(user_path_sets.size());
        for (size_t i = 0; i < user_path_sets.size(); i++)
        {
            auto user_path_set = user_path_sets[i];
            consumers_changed[i] = user_path_set->update(bitsets);

            if (user_path_set->empty(bitsets))
            {
                return handle_no_paths_left_on_update(invoked_by_set, root->name(), node->name());
            }
        }

        // Cumulative cost of user edges going out of this producer
        path_changed |= apply_cumulative_costs<false>(user_path_sets, node, EdgeCost::producer_cost_fns);

        // If any of the paths between producer and this consumer changed, we need to visit producer node and add its operands and users to the needs_update list.
        for(size_t i = 0; i < producers_changed.size(); i++)
        {
            if (path_changed || producers_changed[i])
            {
                const Node* producer_node = operand_path_sets[i]->get_producer_node();
                needs_update.push_back(producer_node);
                if (producers_changed[i])
                {
                    add_operands_and_users(graph, producer_node, needs_update, node);
                }

                edge_changed = true;
            }
        }

        // If any of the paths between this producer and consumer changed, we need to visit consumer node and add its operands and users to the needs_update list.
        for(size_t i = 0; i < consumers_changed.size(); i++)
        {
            if (path_changed || consumers_changed[i])
            {
                const Node* consumer_node = user_path_sets[i]->get_consumer_node();
                needs_update.push_back(consumer_node);
                if (consumers_changed[i])
                {
                    add_operands_and_users(graph, consumer_node, needs_update, node);
                }

                edge_changed = true;
            }
        }

        if (not edge_changed)
            needs_update.pop_back();
    }
}

GraphSolver::Bitset* GraphSolver::get_bitset(graphlib::NodeId node_id) { return &bitsets[bitset_ids.at(node_id)]; }

GraphSolver::Bitset const* GraphSolver::get_bitset(graphlib::NodeId node_id) const
{
    return &bitsets[bitset_ids.at(node_id)];
}

GraphSolver::Bitset* GraphSolver::get_or_insert_bitset(graphlib::NodeId node_id, const Bitset& init)
{
    auto match = bitset_ids.find(node_id);
    if (match == bitset_ids.end())
    {
        BitsetId bitset_id = bitsets.size();
        bitset_ids.insert({node_id, bitset_id});
        auto tmp = bitsets.data();
        const auto disabled_bitset = op_disabled_bitset_cache.find(node_id);
        if (disabled_bitset == op_disabled_bitset_cache.end())
        {
            bitsets.push_back(init);
        }
        else
        {
            bitsets.push_back(init & ~disabled_bitset->second);
        }

        TT_ASSERT(tmp == bitsets.data(), "bitsets reallocated, pointers invalid");
        return &bitsets.back();
    }
    else
    {
        return &bitsets[match->second];
    }
}

void GraphSolver::throw_error_for_edge(graphlib::Edge edge)
{
    graphlib::Node* producer = graph->node_by_id(edge.producer_node_id);
    graphlib::Node* consumer = graph->node_by_id(edge.consumer_node_id);
    if (producer->node_type() == graphlib::NodeType::kInput and producer->shape().rt() == 1 and
        producer->shape().ct() == 1 and producer->shape().z() == 1 and producer->shape().w() == 1 and
        graph->get_edge_attributes(edge)->has_broadcast_dims())
    {
        // Single tile broadcast case
        throw BalancerError(
            fmt::format("Input exceeds max grid forks: {}", producer->name()),
            BalancerError::InputBroadcastExceedsMaxGridForks(producer->id()));
    }
    else
    {
        update_constraint_info();
        reportify::dump_constraints(graph->name(), this);
        throw BalancerError(
            fmt::format("Could not satisfy all constraints for edge: {} -> {}", producer->name(), consumer->name()));
    }
}

// Returns vector of legal OpModels for passed in node by merging legal OpModels of inserted NOP nodes
// and legal OpModels of non-modified nodes from shared data.
//
const std::vector<OpModel>& GraphSolver::get_legal_op_models(graphlib::Node const* node) const
{
    static std::vector<OpModel> null_op_models;

    // For Queue take its producer OpModels.
    //
    if (node->node_type() == graphlib::NodeType::kQueue)
    {
        node = graph->data_operands(node).back();
    }

    const auto recomputed_version_it = op_model_recompute_version.find(node);

    if (recomputed_version_it != op_model_recompute_version.end())
    {
        return shared_data->recomputed_legal_op_models.at(node)[recomputed_version_it->second];
    }

    const auto legal_it = shared_data->legal_op_models.find(node);

    if (legal_it != shared_data->legal_op_models.end())
    {
        return legal_it->second;
    }

    return null_op_models;
}

GraphSolver::RemainingOpModels GraphSolver::at(graphlib::Node const* node) const
{
    auto op_models = RemainingOpModels(get_legal_op_models(node), *get_bitset(node->id()));
    TT_ASSERT(op_models.begin() != op_models.end());
    return op_models;
}

void GraphSolver::set(graphlib::Node const* node, OpModel const& op_model, bool skip_update)
{
    TT_LOG_ASSERT(selected_op_models.count(node) == 0, "OpModel has already been selected for node {}!", node->name());
    graphlib::GraphTraversalContext graph_solver_graph_context(graph, &virtual_nodes, &edges_to_ignore);

    selected_op_models.emplace(node, op_model);
    if (skip_update)  // don't worry about setting legal vs. not, just keep track of what we have in here
        return;

    auto const& op_models = get_legal_op_models(node);
    TT_ASSERT(!op_models.empty());
    std::size_t selection = op_models.size();
    for (std::size_t i = 0; i < op_models.size(); ++i)
    {
        if (op_models[i] == op_model)
        {
            selection = i;
            break;
        }
    }

    Bitset* node_bitset = get_bitset(node->id());

    TT_LOG_ASSERT(selection != op_models.size(), "OpModel not found in legal OpModels for node {}!", node->name());
    TT_LOG_ASSERT((*node_bitset)[selection], "Selection not in legal OpModel set");

    node_bitset->reset();
    node_bitset->set(selection);
    op_disabled_bitset_cache[node->id()] = ~(*node_bitset);

    // If placing on single core grid, don't update the solver as it will overconstraint and waste time modeling op-op
    // connections, since we are cutting and re-resolving anyway after each placed op(op-queue-op).
    //
    if (single_core_ip_mode)
    {
        return;
    }

    update_solver(node, true /*expand_root*/, true /*invoked_by_set*/);
}

// Given current epoch ops, runs the overlay model to determine the amount of memory used for each of the ops. Where
// needed, it adds extra overlay memory to op (in its OpModel) via overlay_size attribute.
//
OpModels* GraphSolver::get_selected_op_models_for_buffering(
    std::unordered_set<const tt::graphlib::Node*> const& current_epoch_ops)
{
    // If fallback (simple) mode is on, we don't need to model ovelay and can just return the selected op models
    //
    if (this->shared_data->constraint->resource_usage_fallback_mode)
    {
        return &selected_op_models;
    }

    // If global overlay blob extra size is set, we don't need to model overlay blob memory footprint as it's accounted
    // for by BBE reserved space
    //
    if (this->shared_data->constraint->device_config.get_overlay_blob_extra_size())
    {
        return &selected_op_models;
    }

    // TODO: Read these value from device config
    // tenstorrent/budabackend#2345
    //
    static constexpr int kPhases32kb = 32 * 1024 / 38;  // 862
    static constexpr int kPhases64kb = 64 * 1024 / 38;  // 1724

    for (const tt::graphlib::Node* node : current_epoch_ops)
    {
        TT_ASSERT(node->node_type() == graphlib::NodeType::kBudaOp);

        std::vector<graphlib::Edge> data_operand_edges = graph->operand_data_edges(node);
        std::vector<graphlib::Edge> data_user_edges = graph->user_data_edges(node);

        // Check usage with producers
        int total_producer_phases = 0;
        for (tt::graphlib::Edge e : data_operand_edges)
        {
            Node* producer = graph->node_by_id(e.producer_node_id);
            if (producer->node_type() != graphlib::NodeType::kBudaOp or not current_epoch_ops.count(producer))
            {
                continue;
            }

            ResourceUsage ru = get_edge_resource_usage(
                graph,
                balancer_cache_collection->pipe_to_resource_usage_cache,
                e,
                selected_op_models.at(producer),
                selected_op_models.at(node));
            // From the perspective of the edge, we're interested in the consumer phases, but from the perspective of
            // the current node, those are producer-side phases
            total_producer_phases += ru.consumer_phases;
        }

        // Check usage with consumers
        int total_consumer_phases = 0;
        for (Edge e : data_user_edges)
        {
            Node* consumer = graph->node_by_id(e.consumer_node_id);
            if (consumer->node_type() != graphlib::NodeType::kBudaOp or not current_epoch_ops.count(consumer))
            {
                continue;
            }

            ResourceUsage ru = get_edge_resource_usage(
                graph,
                balancer_cache_collection->pipe_to_resource_usage_cache,
                e,
                selected_op_models.at(node),
                selected_op_models.at(consumer));
            // From the perspective of the edge, we're interested in the producer phases, but from the perspective of
            // the current node, those are consumer-side phases
            total_consumer_phases += ru.producer_phases;
        }

        // We confirm that the total phases for both producers and consumers are within the limits. If this asserts, it
        // probably means we didn't do a good job in graph solver when calculating EdgeCosts
        //
        TT_ASSERT(total_producer_phases <= kPhases64kb, "Node {} exceeds 64kb phases with producers", node->name());
        TT_ASSERT(total_consumer_phases <= kPhases64kb, "Node {} exceeds 64kb phases with consumers", node->name());

        if (total_producer_phases > kPhases32kb || total_consumer_phases > kPhases32kb)
        {
            // In this case, we need to add extra overlay memory to the op model
            //
            selected_op_models.at(node).overlay_size = 128 * 1024;
        }
    }

    return &selected_op_models;
}

// Returns GraphSolverSolution.
// FINISH also performs graph modification in case buffer was used(there are NOPs inserted by this instance of GS)
// they will be no longer virtual and edges which were marked for removal will be removed.
// Therefore it is important that Balancer pass is completed by invoking finish on chosen GS instance.
//
GraphSolverSolution const GraphSolver::finish()
{
    // Assert in condition finish was already called(potentially from another instance of GS) as this could lead to
    // data inconsistency in graph structure due to virtual nodes being persisted from two different sources.
    //
    TT_ASSERT(!shared_data->graph_solving_finished, "Finish already called on another instance of GraphSolver!");

    // Validate that all nodes have been assigned an OpModel
    // with a consistent state in op_disabled_bitset_cache.
    //
    for (const auto& [node, op_model] : selected_op_models)
    {
        const auto it = op_disabled_bitset_cache.find(node->id());

        // Every selected node must be represented in op_disabled_bitset_cache.
        //
        TT_ASSERT(it != op_disabled_bitset_cache.end());

        // Every selected node must have exactly one bit unset in op_disabled_bitset_cache.
        //
        TT_ASSERT(it->second.count() == it->second.size() - 1);
    }

    // Prevent cleanup of nodes added by this GS.
    //
    for (std::shared_ptr<graphlib::NodeGraphContainer> node_container : virtual_nodes_management)
    {
        node_container->remove_from_graph = false;
    }

    // Persist changes made via this instance of GraphSolver.
    // Remove virtual queues.
    //
    for (const Node* node : virtual_nodes)
    {
        if (node->node_type() != graphlib::NodeType::kQueue)
        {
            graph->mark_node_persisted(node);
        }
        else
        {
            graph->remove_node(node);
        }
    }

    // Remove edges marked for removal.
    //
    for (Edge edge : edges_pending_removal)
    {
        graph->remove_edge(edge);
    }

    update_constraint_info();
    reportify::dump_constraints(graph->name(), this);
    shared_data->graph_solving_finished = true;

    return GraphSolverSolution(selected_op_models, cut_edges);
}

// Will return legal OpModels from shared data, but only for persisted common nodes without buffered nodes/NOPs.
//
LegalOpModels const& GraphSolver::legal_op_models_no_buffering() const { return shared_data->legal_op_models; }

// Will cut passed in edges in a graph and call resolve to recompute GraphSolver.
//
void GraphSolver::cut(std::vector<graphlib::Edge> const& edges, bool epoch_cut)
{
    TT_ASSERT(edges.size() > 0, "At least one edge needs to be passed in for cutting!");
    std::unordered_set<graphlib::Node*> nodes_to_legalize;
    bool partial_reset_allowed = env_as<bool>("PYBUDA_GRAPHSOLVER_FAST") and bitsets.size() > 0;

    for (Edge edge : edges)
    {
        Node* src = graph->node_by_id(edge.producer_node_id);
        Node* dest = graph->node_by_id(edge.consumer_node_id);

#ifdef DEBUG
        // Cutting between non-op nodes will make GraphSolver sad.
        //
        TT_ASSERT(dest->node_type() == graphlib::NodeType::kBudaOp, "Only cutting between BudaOps is supported!");
        TT_ASSERT(src->node_type() == graphlib::NodeType::kBudaOp, "Only cutting between BudaOps is supported!");
#endif
        TT_ASSERT(cut_edges.count(edge) == 0, "Same edge should not be cut twice!");
        TT_LOG_ASSERT(
            selected_op_models.count(src) == 0 or selected_op_models.count(dest) == 0,
            "At least one node affected by CUT must not be SET! {} -> {}",
            src->name(),
            dest->name());
        cut_edges.insert(std::make_pair(edge, false /* self cutting edge */));
        if (selected_op_models.count(src) == 0)
        {
            nodes_to_legalize.insert(src);

            if (partial_reset_allowed)
            {
                *get_bitset(src->id()) = kBitsetAll;
            }
        }

        if (selected_op_models.count(dest) == 0)
        {
            nodes_to_legalize.insert(dest);
            if (partial_reset_allowed)
            {
                *get_bitset(dest->id()) = kBitsetAll;
            }
        }

        // Insert virtual queue on cut edge.
        //
        insert_virtual_queue(edge, src, dest, epoch_cut);
    }

    // Recalculate OpModels for nodes affected by queue insertion
    // and resolve whole graph again.
    //
    recompute_legal_op_models_on_cut(nodes_to_legalize);
    resolve(partial_reset_allowed);
}

void GraphSolver::insert_virtual_queue(graphlib::Edge& edge, const Node* src, const Node* dest, bool is_e2e_queue)
{
    TT_ASSERT(edge.edge_type == graphlib::EdgeType::kData, "Cut only data edges!");
    graphlib::Node* queue_node = nullptr;
    uintptr_t gs_unique = (uintptr_t) static_cast<const void*>(this);
    std::string queue_name = "virtual_queue_" + src->name() + "_" + dest->name() + "_" +
                             std::to_string(edge.consumer_input_port_id) + "_" + std::to_string(gs_unique);

    // Insert virtual dummy queue on this edge.
    //
    if (!is_e2e_queue)
    {
        queue_node = graph->add_node(
            graphlib::create_node<graphlib::BufferingQueueNode>(queue_name, 1 /* num_entries */),
            graph->get_subgraph_id_for_node(src->id()));
    }
    else
    {
        // cross_epoch_type and cross_chip_type will be properly recalculated at a later phase
        // since these virtual dummy queues will be replaced with proper ones in post placer pass.
        //
        queue_node = graph->add_node(
            graphlib::create_node<graphlib::EpochToEpochQueueNode>(
                queue_name, false /*cross_epoch_type*/, false /*cross_chip_type*/),
            graph->get_subgraph_id_for_node(src->id()));
    }

    queue_node->set_shape(graph->node_by_id(edge.producer_node_id)->shape());
    queue_node->set_output_df(graph->node_by_id(edge.producer_node_id)->output_df());
    queue_node->set_epoch_type(dest->get_epoch_type());

    Edge node_to_q_edge(
        edge.producer_node_id, edge.producer_output_port_id, queue_node->id(), 0, graphlib::EdgeType::kData);
    graph->add_edge(node_to_q_edge);
    graph->get_edge_attributes(node_to_q_edge)->set_ublock_order(graph->get_edge_attributes(edge)->get_ublock_order());

    graphlib::Edge q_to_node_edge =
        Edge(queue_node->id(), 0, edge.consumer_node_id, edge.consumer_input_port_id, graphlib::EdgeType::kData);
    graph->add_edge(q_to_node_edge);
    graph->copy_edge_attributes(edge, q_to_node_edge);

    // Going forward this edge will be ignored within GraphTraversalContext.
    //
    edges_to_ignore.insert(edge);

    // Register inserted queue as virtual.
    //
    register_virtual_node(queue_node);
}

void GraphSolver::recompute_legal_op_models_on_cut(std::unordered_set<graphlib::Node*>& nodes_to_legalize)
{
    if (use_op_model_recalculation_on_cut)
    {
        TT_ASSERT(nodes_to_legalize.size() > 0, "At least one node must be specified for legal op models recompute!");
        graphlib::GraphTraversalContext graph_solver_graph_context(graph, &virtual_nodes, &edges_to_ignore);

        recompute_legal_op_models(nodes_to_legalize);
    }
}

void GraphSolver::recompute_legal_op_models(std::unordered_set<graphlib::Node*>& nodes_to_legalize)
{
    // Call Legalizer and calculate OpModels for inserted NOP nodes. At the same time recalculate OpModels for
    // affected non virtual nodes, ie nodes which are in direct connection with inserted NOP nodes/virtual queues.
    //
    LegalOpModels recomputed_legal_op_models =
        legalizer::get_legal_op_models(graph, balancer_config, balancer_cache_collection, &nodes_to_legalize);

    // Insert or update legal OpModels in recomputed_legal_op_models. Assert if SET was already invoked for them.
    //
    for (auto& it : recomputed_legal_op_models)
    {
        // Reset disabled op model cache as op models are being recomputed.
        //
        op_disabled_bitset_cache.erase(it.first->id());

        auto recomputed_it = this->shared_data->recomputed_legal_op_models.find(it.first);
        if (recomputed_it == this->shared_data->recomputed_legal_op_models.end())
        {
            std::vector<std::vector<OpModel>> versioned_recomputed_legal_op_models;
            versioned_recomputed_legal_op_models.push_back(std::move(it.second));
            this->shared_data->recomputed_legal_op_models.emplace(
                it.first, std::move(versioned_recomputed_legal_op_models));
            op_model_recompute_version.emplace(it.first, 0);
        }
        else
        {
            recomputed_it->second.push_back(std::move(it.second));
            op_model_recompute_version[it.first] = recomputed_it->second.size() - 1;
        }
    }
}

// BUFFER is used to directly insert NOPs in GraphSolving phase on vector of provided graph edges according to
// attributes specified in BufferInfo. NOPs are inserted only for this instance of GS and are not visible globally until
// FINISH is invoked for GS instance which is accepted as a solution. This is achieved via introduction of concept of
// virtual nodes and GraphTraversalContext allowing separate instances of GS to work independently with their own
// version of the graph.
//
// RETURNS vector of nodes which were inserted plus modified nodes in case SET needs to be called  again for them.
// Note that for every edge, producer-consumer node pair, selected op models will be reset for both producer and
// consumer as OpModels for them will be recalculated. If SET was already invoked for those nodes, selected op models
// will be reset, and they will be returned as part of the results along with NOP nodes so that SET can be invoked again
// for them.
//
std::vector<graphlib::Node*> GraphSolver::buffer(std::vector<BufferInfo>& buffer_edges)
{
    graphlib::GraphTraversalContext graph_solver_graph_context(graph, &virtual_nodes, &edges_to_ignore);
    bool partial_reset_allowed = env_as<bool>("PYBUDA_GRAPHSOLVER_FAST") and bitsets.size() > 0;
    std::vector<graphlib::Node*> inserted_nodes;
    std::unordered_set<graphlib::Node*> nodes_to_legalize;
    auto op_name = [](Node* src, Node* dest, std::uint32_t buffer_index)
    {
        return "buffer_" + std::to_string(buffer_index) + "_" + std::to_string(src->id()) + "_" +
               std::to_string(dest->id());
    };
    for (BufferInfo buff_info : buffer_edges)
    {
        TT_ASSERT(buff_info.nop_count > 0, "NOP insertion count must be higher than 0!");

        Node* dest = graph->node_by_id(buff_info.edge.consumer_node_id);
        Node* src = graph->node_by_id(buff_info.edge.producer_node_id);
        TT_ASSERT(
            graph->is_node_visible(dest),
            "Node invisible for this instance of GS, probably owned by different instance of GS!");
        TT_ASSERT(
            graph->is_node_visible(src),
            "Node invisible for this instance of GS, probably owned by different instance of GS!");
        Node* original_dest = dest;

        std::size_t buffer_index = 0;

        // Besides legalizing newly inserted NOP nodes, re-legalize nodes which will be connected to them
        // as having NOP in between can potentially produce more valid/different OpModels.
        // Do this only for nodes which are not SET already, this is up to Balancer policy to decide/control.
        //
        if (selected_op_models.count(src) == 0)
        {
            nodes_to_legalize.insert(src);
            if (partial_reset_allowed)
            {
                *get_bitset(src->id()) = kBitsetAll;
            }
        }

        if (selected_op_models.count(dest) == 0)
        {
            nodes_to_legalize.insert(dest);
            if (partial_reset_allowed)
            {
                *get_bitset(dest->id()) = kBitsetAll;
            }
        }

        for (int nop_count = 0; nop_count < buff_info.nop_count; nop_count++)
        {
            graphlib::BudaOpNode* buffer_nop = nullptr;
            while (graph->has_node_with_name(op_name(src, original_dest, buffer_index))) buffer_index++;

            for (graphlib::Edge e : graph->get_edges(src, dest))
            {
                if (e.edge_type != graphlib::EdgeType::kData)
                    continue;

                if (buffer_nop == nullptr)
                {
                    buffer_nop = graph->add_node(
                        graphlib::create_node<graphlib::BudaOpNode>(op_name(src, original_dest, buffer_index), "nop"),
                        graph->get_subgraph_id_for_node(src->id()));
                    buffer_nop->set_shape(src->shape());
                    buffer_nop->set_buffering_op(true);
                    buffer_nop->set_epoch_type(original_dest->get_epoch_type());
                    buffer_nop->set_output_df(src->output_df());
                    auto src_buda_op = dynamic_cast<graphlib::BudaOpNode*>(src);
                    if (src_buda_op != nullptr and src_buda_op->op_name() != "dequantization")
                    {
                        buffer_nop->set_intermediate_df(src_buda_op->intermediate_df());
                        buffer_nop->set_accumulate_df(src_buda_op->accumulate_df());
                        buffer_nop->set_math_fidelity(src_buda_op->math_fidelity());
                    }

                    register_virtual_node(buffer_nop);
                    nodes_to_legalize.insert(buffer_nop);
                    inserted_nodes.push_back(buffer_nop);
                }

                auto [edge0, edge1] = graphlib::insert_node_on_edge(
                    graph, e, buffer_nop, false /*inherit_consumer_attrs*/, false /*remove_edge*/);

                // Edge cannot be removed right away from the graph as we will affect global state
                // for all GS instances and end up modifing common graph by GS instance that may end up discarded.
                // Thats why we collect list of edges which are pending for removal and removing them in FINISH method
                // when balancing is complete.
                //
                edges_to_ignore.insert(e);
                edges_pending_removal.push_back(e);

                log_trace(
                    LogGraphCompiler,
                    "Inserted buffer nop node {} between {} and {}",
                    buffer_nop->name(),
                    src->name(),
                    dest->name());

                // Move TMs to edge1.
                //
                auto& tms = graph->get_edge_attributes(edge0)->get_tms();

                // TODO Should we do this by default, should hoist_tms remain?
                //
                if (not buff_info.hoist_tms)
                {
                    // Not hoisting tms, move them to edge1.
                    //
                    graph->get_edge_attributes(edge1)->set_tms(tms);
                    graph->get_edge_attributes(edge0)->set_tms(std::vector<graphlib::OpType>{});
                }

                dest = buffer_nop;
            }
        }
    }

    recompute_legal_op_models(nodes_to_legalize);
    // We need to resolve again with new nodes, edges and OpModels.
    //
    resolve(partial_reset_allowed);

    return inserted_nodes;
}

// REGISTER_VIRTUAL_NODE marks inserted node as virtual, and tracks it in virtual_nodes and
// virtual_nodes_management for auto removal from graph in case this GS gets discarded.
//
void GraphSolver::register_virtual_node(graphlib::Node* virtual_node)
{
    graph->mark_node_virtual(virtual_node);
    virtual_nodes.insert(virtual_node);
    virtual_nodes_management.emplace_back(std::make_shared<graphlib::NodeGraphContainer>(virtual_node, graph));
}

// Set GraphTraversalContext of this GS instance externally wherever it is needed for graph operations
// set in context of this GS instance.
//
std::unique_ptr<graphlib::GraphTraversalContext> GraphSolver::get_graph_traversal_context()
{
    return std::make_unique<graphlib::GraphTraversalContext>(graph, &virtual_nodes, &edges_to_ignore);
}

// Similar to above but for epoch traversal context based on passed in nodes.
//
std::unique_ptr<graphlib::GraphTraversalContext> GraphSolver::get_graph_epoch_traversal_context(
    const std::unordered_set<const graphlib::Node*>* epoch_nodes)
{
    return std::make_unique<graphlib::GraphTraversalContext>(graph, epoch_nodes, &virtual_nodes, &edges_to_ignore);
}

// Suboptimal op models invalidation according to provided invalidation
// strategy(GraphSolverOpModelInvalidationStrategy).
//
void GraphSolver::invalidate_suboptimal_op_models(int invalidation_strategy)
{
    if (!env_as<bool>("PYBUDA_BALANCER_PREPASS_DISABLED"))
    {
        this->suboptimal_opmodel_invalidation_strategy = invalidation_strategy;
        invalidate_suboptimal_op_models(graphlib::topological_sort(*graph));
    }
}

void GraphSolver::invalidate_suboptimal_op_models(const std::vector<graphlib::Node*>& nodes)
{
    for (GraphSolverOpModelInvalidationStrategyTier tier : {FirstTier, SecondTier})
    {
        for (const graphlib::Node* node : nodes)
        {
            if (node->node_type() == graphlib::NodeType::kBudaOp)
            {
                const graphlib::BudaOpNode* op_node = static_cast<const graphlib::BudaOpNode*>(node);
                invalidate_suboptimal_op_models_for_op(op_node, tier);
            }
        }
    }
}

void GraphSolver::invalidate_streaming_into_output(const std::vector<graphlib::Node*>& nodes)
{
    for (graphlib::Node* node : nodes)
    {
        // Try to eliminate streaming into output if possible.
        //
        if (node->node_type() == graphlib::NodeType::kOutput)
        {
            for (graphlib::Node* operand_node : graph->data_operands(node))
            {
                if (operand_node->node_type() == graphlib::NodeType::kBudaOp)
                {
                    bool no_stream_output_valid = false;
                    const graphlib::BudaOpNode* op_node = static_cast<const graphlib::BudaOpNode*>(operand_node);

                    // Op model already selected for this node, skip.
                    //
                    if (selected_op_models.count(op_node) > 0)
                    {
                        continue;
                    }

                    const std::vector<tt::balancer::OpModel>& op_models = get_legal_op_models(op_node);
                    if (op_models.size() == 1)
                    {
                        continue;
                    }

                    Bitset* node_bitset = get_bitset(op_node->id());
                    std::uint32_t op_model_count = std::min(kNumBitsetBits, std::max(1lu, op_models.size()));

                    for (size_t index = 0; index < op_model_count; index++)
                    {
                        if (!node_bitset->test(index))
                        {
                            continue;
                        }

                        if (op_models[index].t_stream_factor.none())
                        {
                            no_stream_output_valid = true;
                            break;
                        }
                    }

                    // At least one valid non stream option present. Eliminate streaming ones.
                    //
                    if (no_stream_output_valid)
                    {
                        Bitset discarded_op_models_bitset;
                        bool stream_option_eliminated = false;

                        for (std::size_t index = 0; index < op_model_count; index++)
                        {
                            if (node_bitset->test(index) and !op_models[index].t_stream_factor.none())
                            {
                                discarded_op_models_bitset.set(index);
                                stream_option_eliminated = true;
                            }
                        }

                        // We eliminated at least one op_model. Update bitset and solver.
                        // Also update op_disabled_bitset_cache for this op node, to speed up future resolves.
                        //
                        if (stream_option_eliminated)
                        {
                            *node_bitset &= ~discarded_op_models_bitset;
                            auto it = op_disabled_bitset_cache.find(op_node->id());

                            if (it == op_disabled_bitset_cache.end())
                            {
                                op_disabled_bitset_cache.emplace(op_node->id(), discarded_op_models_bitset);
                            }
                            else
                            {
                                it->second |= discarded_op_models_bitset;
                            }

                            update_solver(operand_node);
                        }
                    }
                }
            }
        }
    }
}

void GraphSolver::invalidate_suboptimal_op_models_for_op(
    const graphlib::BudaOpNode* op_node, GraphSolverOpModelInvalidationStrategyTier tier)
{
    // Op model already selected for this node, skip.
    //
    if (selected_op_models.count(op_node) > 0)
    {
        return;
    }

    const std::vector<tt::balancer::OpModel>& op_models = get_legal_op_models(op_node);

    if (op_models.size() == 1)
    {
        return;
    }

    Bitset* node_bitset = get_bitset(op_node->id());
    std::uint32_t op_model_count = std::min(kNumBitsetBits, std::max(1lu, op_models.size()));

    switch (tier)
    {
        case FirstTier:
        {
            if (op_node->is_matmul_not_sparse())
            {
                if (suboptimal_opmodel_invalidation_strategy & ((int)DenseMatmulPrologue | (int)DenseMatmulBetterUkt))
                {
                    uint32_t disabled_op_models = 0;
                    uint32_t discarded_op_models = 0;
                    Bitset discarded_op_models_bitset;

                    int max_ukt = 0;
                    bool has_one_valid_prologue = false;

                    // First screen what is available for pruning.
                    //
                    for (size_t i = 0; i < op_model_count; i++)
                    {
                        if (!node_bitset->test(i))
                        {
                            disabled_op_models++;
                            continue;
                        }

                        bool has_prologue = op_models[i].parameter_buffers[1];
                        has_one_valid_prologue |= has_prologue;

                        int u_kt = op_models[i].input_buffers[0].block_shape.ublock.ct;
                        int m_k = op_models[i].op_shape.inputs[0].ct / u_kt;

                        if ((m_k * u_kt >= 8) and u_kt > max_ukt)
                        {
                            max_ukt = u_kt;
                        }
                    }

                    if (has_one_valid_prologue and suboptimal_opmodel_invalidation_strategy & (int)DenseMatmulPrologue)
                    {
                        max_ukt = 0;
                        for (size_t i = 0; i < op_model_count; i++)
                        {
                            if (!node_bitset->test(i))
                            {
                                continue;
                            }

                            bool has_prologue = op_models[i].parameter_buffers[1];
                            if (!has_prologue)
                            {
                                discarded_op_models_bitset.set(i);
                                discarded_op_models++;
                                continue;
                            }

                            int u_kt = op_models[i].input_buffers[0].block_shape.ublock.ct;
                            int m_k = op_models[i].op_shape.inputs[0].ct / u_kt;

                            if ((m_k * u_kt >= 8) and u_kt > max_ukt)
                            {
                                max_ukt = u_kt;
                            }
                        }
                    }

                    int ukt_limit = std::min(max_ukt, 4);

                    if (ukt_limit > 1 and suboptimal_opmodel_invalidation_strategy & (int)DenseMatmulBetterUkt)
                    {
                        for (size_t i = 0; i < op_model_count; i++)
                        {
                            if (!node_bitset->test(i) or discarded_op_models_bitset.test(i))
                            {
                                continue;
                            }

                            int u_kt = op_models[i].input_buffers[0].block_shape.ublock.ct;
                            int m_k = op_models[i].op_shape.inputs[0].ct / u_kt;
                            if ((m_k * u_kt >= 8) && u_kt < ukt_limit)
                            {
                                discarded_op_models_bitset.set(i);
                                discarded_op_models++;
                                continue;
                            }
                        }
                    }

                    if (discarded_op_models > 0 and discarded_op_models + disabled_op_models < op_model_count)
                    {
                        *node_bitset &= ~discarded_op_models_bitset;
                        auto it = op_disabled_bitset_cache.find(op_node->id());

                        if (it == op_disabled_bitset_cache.end())
                        {
                            op_disabled_bitset_cache.emplace(op_node->id(), discarded_op_models_bitset);
                        }
                        else
                        {
                            it->second |= discarded_op_models_bitset;
                        }

                        update_solver(op_node);
                    }
                }
            }
        }
        break;

        case SecondTier:
        {
            if (op_node->is_sparse_matmul())
            {
                if (suboptimal_opmodel_invalidation_strategy & (int)MatmulSparseDenseGridPairing)
                {
                    uint32_t disabled_op_models = 0;
                    uint32_t discarded_op_models = 0;
                    Bitset discarded_op_models_bitset;

                    for (size_t i = 0; i < op_model_count; i++)
                    {
                        if (!node_bitset->test(i))
                        {
                            disabled_op_models++;
                            continue;
                        }

                        if (op_models[i].grid_shape.c != 1)
                        {
                            discarded_op_models_bitset.set(i);
                            discarded_op_models++;
                            continue;
                        }
                    }

                    if (discarded_op_models > 0 and discarded_op_models + disabled_op_models < op_model_count)
                    {
                        *node_bitset &= ~discarded_op_models_bitset;
                        auto it = op_disabled_bitset_cache.find(op_node->id());

                        if (it == op_disabled_bitset_cache.end())
                        {
                            op_disabled_bitset_cache.emplace(op_node->id(), discarded_op_models_bitset);
                        }
                        else
                        {
                            it->second |= discarded_op_models_bitset;
                        }

                        update_solver(op_node);
                    }

                    PathSet* sparse_to_dense_pathset = get_user_path_sets_pts(op_node)[0];
                    const graphlib::BudaOpNode* consumer =
                        dynamic_cast<const graphlib::BudaOpNode*>(sparse_to_dense_pathset->get_consumer_node());
                    if (!consumer or !consumer->should_pair_with_sparse(op_node, graph))
                    {
                        return;
                    }

                    bool can_prune_paths = false;
                    const std::vector<tt::balancer::OpModel>& dense_op_models = get_legal_op_models(consumer);
                    for (const auto& path : sparse_to_dense_pathset->get_paths())
                    {
                        if (op_models[path.producer_id].grid_shape.r == dense_op_models[path.consumer_id].grid_shape.r)
                        {
                            can_prune_paths = true;
                            break;
                        }
                    }

                    if (can_prune_paths)
                    {
                        PathSet::Paths* paths = sparse_to_dense_pathset->get_paths_pt();
                        for (size_t i = 0; i < paths->size(); i++)
                        {
                            if (op_models[(*paths)[i].producer_id].grid_shape.r !=
                                dense_op_models[(*paths)[i].consumer_id].grid_shape.r)
                            {
                                (*paths)[i] = paths->back();
                                paths->pop_back();
                                i--;
                            }
                        }

                        update_solver(consumer);
                    }
                }
            }
        }
        break;

        default: TT_ASSERT("Invalid/undefined tier!");
    }
}

void GraphSolver::set_filter_grid_size(graphlib::Node const* node, OpModel const& role_op_model)
{
    const std::vector<tt::balancer::OpModel>& op_models = get_legal_op_models(node);

    Bitset* node_bitset = get_bitset(node->id());
    Bitset temp_bitset = *node_bitset;
    std::uint32_t op_model_count = std::min(kNumBitsetBits, std::max(1lu, op_models.size()));
    Bitset discarded_op_models_bitset;
    for (size_t i = 0; i < op_model_count; i++)
    {
        if (!node_bitset->test(i))
        {
            continue;
        }

        if (op_models[i].grid_shape.c > role_op_model.grid_shape.c || op_models[i].grid_shape.r != role_op_model.grid_shape.r)
        {
            discarded_op_models_bitset.set(i);
        }
    }

    if (discarded_op_models_bitset.none())
    {
        return;
    }

    temp_bitset &= ~discarded_op_models_bitset;

    TT_ASSERT(temp_bitset.any());

    *node_bitset = temp_bitset;

    auto it = op_disabled_bitset_cache.find(node->id());

    if (it == op_disabled_bitset_cache.end())
    {
        op_disabled_bitset_cache.emplace(node->id(), discarded_op_models_bitset);
    }
    else
    {
        it->second |= discarded_op_models_bitset;
    }

    update_solver(node);
}

#ifdef DEBUG
// Computes and logs if there are valid connections for this edge among paths
// that were discarded by previously computed edges(edge eliminated by disabling some OpModels).
//
void GraphSolver::compute_edge_elimination_debug_info(
    Edge& edge,
    Bitset* producer_bitset,
    Bitset* consumer_bitset,
    Bitset& edge_producer_bitset,
    Bitset& edge_consumer_bitset,
    std::vector<OpModel>& producer_op_models_debug,
    std::vector<OpModel>& consumer_op_models_debug,
    std::uint64_t producer_count,
    std::uint64_t consumer_count,
    EdgeConstraintDebugInfo& edge_constraint_debug_info,
    EdgeConstraintDebugInfo& graph_constraint_debug_info)
{
    Constraint* constraint = shared_data->constraint.get();
    Bitset eliminatedProducers = *producer_bitset ^ edge_producer_bitset;
    Bitset eliminatedConsumers = *consumer_bitset ^ edge_consumer_bitset;

    // Propagate edge elimination for path eliminated but valid prod-consumer combinations.
    //
    for (std::uint64_t producer_id = 0; producer_id < producer_count; ++producer_id)
    {
        for (std::uint64_t consumer_id = 0; consumer_id < consumer_count; ++consumer_id)
        {
            bool producerNodeOpModelDisabled = (*producer_bitset)[producer_id] == 0;
            bool consumerNodeOpModelDisabled = (*consumer_bitset)[consumer_id] == 0;
            bool producerNodeOpModelEliminatedByCurrentEdge = eliminatedProducers[producer_id] != 0;
            bool consumerNodeOpModelEliminatedByCurrentEdge = eliminatedConsumers[consumer_id] != 0;

            // We already went through these non disabled paths.
            //
            if (!producerNodeOpModelDisabled && !consumerNodeOpModelDisabled)
            {
                continue;
            }

            auto [cost, constraint_failure_reason] = cost_fn(
                constraint, graph, edge, producer_op_models_debug, consumer_op_models_debug, producer_id, consumer_id);

            if (NoConstraintFailure == constraint_failure_reason)
            {
                if (not cost.exceeded())
                {
                    TT_ASSERT(producer_id <= std::numeric_limits<decltype(Path::producer_id)>::max());
                    TT_ASSERT(consumer_id <= std::numeric_limits<decltype(Path::consumer_id)>::max());

                    if (producerNodeOpModelDisabled && !producer_op_models_debug.empty())
                    {
                        if (!consumerNodeOpModelDisabled)
                        {
                            if (consumerNodeOpModelEliminatedByCurrentEdge && !consumer_op_models_debug.empty())
                            {
                                consumer_op_models_debug[consumer_id].eliminating_edge =
                                    producer_op_models_debug[producer_id].eliminating_edge;
                            }
                        }

                        edge_constraint_debug_info.addEliminatingEdge(
                            producer_op_models_debug[producer_id].eliminating_edge);
                    }

                    if (consumerNodeOpModelDisabled && !consumer_op_models_debug.empty())
                    {
                        if (!producerNodeOpModelDisabled)
                        {
                            if (producerNodeOpModelEliminatedByCurrentEdge && !producer_op_models_debug.empty())
                            {
                                producer_op_models_debug[producer_id].eliminating_edge =
                                    consumer_op_models_debug[consumer_id].eliminating_edge;
                            }
                        }

                        edge_constraint_debug_info.addEliminatingEdge(
                            consumer_op_models_debug[consumer_id].eliminating_edge);
                    }

                    edge_constraint_debug_info.recordEdgeConstraintFailure(EdgePathRemovedByPriorEdgeElimination);
                    graph_constraint_debug_info.recordEdgeConstraintFailure(EdgePathRemovedByPriorEdgeElimination);
                }
            }
        }
    }

    // Mark OpModels eliminated by current edge which are not already eliminated by ancestor egdes.
    //
    for (std::uint64_t producer_id = 0; producer_id < producer_count; ++producer_id)
    {
        if (eliminatedProducers[producer_id] != 0)
        {
            if (0 == std::get<0>(producer_op_models_debug[producer_id].eliminating_edge))
            {
                producer_op_models_debug[producer_id].eliminating_edge = edge.unique_id();
            }
        }
    }

    for (std::uint64_t consumer_id = 0; consumer_id < consumer_count; ++consumer_id)
    {
        if (eliminatedConsumers[consumer_id] != 0)
        {
            if (0 == std::get<0>(consumer_op_models_debug[consumer_id].eliminating_edge))
            {
                consumer_op_models_debug[consumer_id].eliminating_edge = edge.unique_id();
            }
        }
    }
}
#endif

void GraphSolver::update_constraint_info()
{
    PROFILE_SCOPE();

    if (!env_as<bool>("PYBUDA_COLLECT_CONSTRAINT_INFO"))
        return;

    auto create_edge_name = [](graphlib::Edge edge)
    {
        return fmt::format(
            "{}@{}:{}@{}",
            edge.producer_node_id,
            edge.producer_output_port_id,
            edge.consumer_node_id,
            edge.consumer_input_port_id);
    };

    int num_pages = (graph->num_nodes() + ConstraintInfo::kPageSize - 1) / ConstraintInfo::kPageSize;
    auto nodes = graphlib::topological_sort(*graph);
    bool recomputed = false;

    // Needs to change if the number of nodes changed or if GS owner instance changed.
    //
    if (constraint_info_ptr->node_id_to_name.size() != nodes.size() or
        constraint_info_ptr->needs_to_be_recomputed(this))
    {
        constraint_info_ptr->graph_name = graph->name();
        constraint_info_ptr->pages.clear();
        constraint_info_ptr->pages.resize(num_pages);
        constraint_info_ptr->node_name_to_page.clear();
        recomputed = true;

        int page_idx = 0;
        for (ConstraintInfo::Page& page : constraint_info_ptr->pages)
        {
            page.node_id_order.reserve(ConstraintInfo::kPageSize);
            for (int i = page_idx * ConstraintInfo::kPageSize;
                 i < std::min((page_idx + 1) * ConstraintInfo::kPageSize, (int)nodes.size());
                 ++i)
            {
                auto* node = nodes[i];

                page.node_id_order.push_back(node->id());
                constraint_info_ptr->node_id_to_name.insert({std::to_string(node->id()), node->name()});
                constraint_info_ptr->node_name_to_page.insert(
                    {node->name(), std::make_pair(page_idx, i % ConstraintInfo::kPageSize)});

                auto const& op_models = get_legal_op_models(node);
                if (!op_models.empty())
                {
                    auto& node_op_model_ids = page.node_id_to_op_model_ids[std::to_string(node->id())];
                    for (auto const& op_model : op_models)
                    {
                        page.id_to_op_models.insert({std::to_string(op_model.id.id), op_model});
                        node_op_model_ids.push_back(op_model.id.id);
                    }
                }
            }
            page_idx += 1;
        }

        constraint_info_ptr->gs_owner_cache = this;
    }

    // Clear existing info and overwrite with latest resolve
    int page_idx = 0;
    for (ConstraintInfo::Page& page : constraint_info_ptr->pages)
    {
        for (int i = page_idx * ConstraintInfo::kPageSize;
             i < std::min((page_idx + 1) * ConstraintInfo::kPageSize, (int)nodes.size());
             ++i)
        {
            auto* node = nodes[i];

            // Do not update nodes that have already been selected for
            if (not recomputed and selected_op_models.find(node) != selected_op_models.end())
                continue;

            auto const& consumer_op_models = get_legal_op_models(node);
            for (graphlib::Edge edge : graph->operand_data_edges(node))
            {
                auto* producer = graph->node_by_id(edge.producer_node_id);
                auto const& producer_op_models = get_legal_op_models(producer);
                auto match = path_set_ids.find(edge);
                if (match != path_set_ids.end())
                {
                    std::string edge_name = create_edge_name(edge);
                    auto& paths = page.edge_to_path_sets[edge_name];
                    PathSet const& path_set = get_path_set(edge);
                    paths.clear();
                    paths.reserve(path_set.get_paths().size());
                    for (Path const& path : path_set.get_paths())
                    {
                        paths.push_back(std::make_tuple(path.producer_id, path.consumer_id));
                    }
                }

                std::uint64_t producer_count = std::min(kNumBitsetBits, producer_op_models.size());
                std::uint64_t consumer_count = std::min(kNumBitsetBits, consumer_op_models.size());
                for (std::uint64_t producer_id = 0; producer_id < producer_count; ++producer_id)
                {
                    for (std::uint64_t consumer_id = 0; consumer_id < consumer_count; ++consumer_id)
                    {
                        std::string key = fmt::format(
                            "{}:{}", producer_op_models[producer_id].id.id, consumer_op_models[consumer_id].id.id);
                        auto match = failure_reasons.find(key);
                        if (match != failure_reasons.end())
                            page.failure_reason_ids[key] = match->second;
                    }
                }
            }
        }

        page_idx += 1;
    }

    constraint_info_ptr->op_model_selection.clear();
    constraint_info_ptr->op_model_selection.reserve(selected_op_models.size());
    for (auto const& [node, op_model] : selected_op_models)
    {
        constraint_info_ptr->op_model_selection.push_back(op_model.id.id);
    }
}
}  // namespace tt::balancer::legalizer
