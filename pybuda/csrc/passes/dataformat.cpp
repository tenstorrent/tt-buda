// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/dataformat.hpp"

#include "backend_api/device_config.hpp"
#include "buda_passes.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "lower_to_buda/common.hpp"
#include "passes/amp.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{

static std::vector<Node*> get_non_constants(const std::vector<Node*>& nodes)
{
    std::vector<Node*> non_constants;
    for (const auto node : nodes)
    {
        if (not graphlib::is_constant_input(node))
        {
            non_constants.push_back(node);
        }
    }
    return non_constants;
}

static std::vector<DataFormat> get_data_formats(const std::vector<Node*>& nodes)
{
    std::vector<DataFormat> data_formats;
    std::transform(nodes.cbegin(), nodes.cend(), std::back_inserter(data_formats),
                [](const Node* node) { return node->output_df(); });
    return data_formats;
}

static bool are_data_formats_same_exponent_widths(const std::vector<DataFormat> &data_formats)
{
    // For non-integer data formats, we'll check whether all a/b data formats are the same
    std::vector<DataFormat> non_integer_data_formats;
    std::copy_if(
        std::begin(data_formats),
        std::end(data_formats),
        std::back_inserter(non_integer_data_formats),
        [](DataFormat data_format)
        { return data_format != DataFormat::Float32 and not is_integer_data_format(data_format); });

    if (non_integer_data_formats.empty())
    {
        return true;
    }

    return std::all_of(
        non_integer_data_formats.begin(),
        non_integer_data_formats.end(),
        [&non_integer_data_formats](DataFormat data_format)
        { return is_b_data_format(data_format) == is_b_data_format(non_integer_data_formats.at(0)); });
}

static bool are_data_formats_all_integer(const std::vector<DataFormat> &data_formats)
{
    return std::all_of(
        data_formats.begin(),
        data_formats.end(),
        [](DataFormat data_format) { return is_integer_data_format(data_format); });
}

static bool are_data_formats_all_float(const std::vector<DataFormat> &data_formats)
{
    return std::all_of(
        data_formats.begin(),
        data_formats.end(),
        [](DataFormat data_format) { return !is_integer_data_format(data_format); });
}

static bool are_data_formats_same(const std::vector<DataFormat> &data_formats)
{
    if (data_formats.empty())
    {
        return true;
    }
    return std::all_of(
        data_formats.begin(),
        data_formats.end(),
        [&data_formats](DataFormat data_format) { return data_format == data_formats.at(0); });
}

static bool contains_data_format(const std::vector<DataFormat> &data_formats, DataFormat target_data_format)
{
    if (data_formats.empty())
    {
        return false;
    }
    return std::any_of(
        data_formats.begin(),
        data_formats.end(),
        [&target_data_format](DataFormat data_format) { return data_format == target_data_format; });
}

static bool is_configured_for_int8(const graphlib::Graph *graph, const Node* node)
{
    return (contains_data_format(get_data_formats(graph->data_operands(node)), DataFormat::Int8)
            or contains_data_format({node->output_df()}, DataFormat::Int8));
}

static bool is_configured_for_int32(const graphlib::Graph *graph, const Node* node)
{
    return contains_data_format(get_data_formats(graph->data_operands(node)), DataFormat::Int32);
}

static bool is_exponent_width_reconfigured(DataFormat from, DataFormat to)
{
    return is_b_data_format(from) != is_b_data_format(to);
}

static DataFormat get_highest_precision_data_format(const std::vector<DataFormat> &data_formats)
{
    TT_LOG_ASSERT(not data_formats.empty(), "data_formats supplied to get_highest_precision_data_format(..) is empty");
    DataFormat highest_precision_data_format = data_formats.at(0);
    for (auto data_format : data_formats)
    {
        if (get_precision_bits(data_format) > get_precision_bits(highest_precision_data_format))
        {
            highest_precision_data_format = data_format;
        }
    }
    return highest_precision_data_format;
}


DataFormat get_inferred_accumulate_df(
    const graphlib::Graph *graph, const graphlib::BudaOpNode *op, const bool fp32_acc_supported)
{
    // infer the accumulation data-format; opt to highest-precision setting
    // data formats out of the unpacker gasket
    //  - For grayskull: {fp16a/b}
    //  - For wormhole: {fp16a/b, tf32}
    auto operands = graph->data_operands(op);
    DataFormat input_data_format = operands.at(0)->output_df();
    DataFormat accumulate_data_format;

    if (is_configured_for_int8(graph, op))
    {
        accumulate_data_format = DataFormat::Int32;
    }
    else if (fp32_acc_supported and input_data_format == DataFormat::Float32)
    {
        accumulate_data_format = DataFormat::Float32;
    }
    else if (is_b_data_format(input_data_format))
    {
        accumulate_data_format = DataFormat::Float16_b;
    }
    else
    {
        accumulate_data_format = DataFormat::Float16;
    }
    return accumulate_data_format;
}

// Apply math fidelity settings
void apply_math_fidelity(
    graphlib::Graph *graph, const MathFidelity default_math_fidelity)
{
    for (Node *node : graph->nodes())
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
            continue;

        auto op = node->as<graphlib::BudaOpNode>();

        op->set_math_fidelity(default_math_fidelity);
        if (bool enable_integer_mode = is_configured_for_int8(graph, op) or is_configured_for_int32(graph, op); enable_integer_mode)
        {
            if (enable_integer_mode and op->math_fidelity() != MathFidelity::HiFi4)
            {
                log_warning("Node {} is configured for int8, but math fidelity is not HiFi4. "
                            "Setting math fidelity from {} to HiFi4.",
                            op->name(),
                            op->math_fidelity());
                op->set_math_fidelity(MathFidelity::HiFi4);
            }
        }
        else if (op and op->is_sparse_matmul())
        {
            std::vector<graphlib::Node *> data_operands = graph->data_operands(op);
            TT_LOG_ASSERT(data_operands.size() >=2, "Sparse Matmul: {} must have at least two operands.", op->name());
            graphlib::Node* activations = graph->data_operands(op)[1];

            // If activations have more than 4 bits on mantissa, bump up the math fidelity
            // ensure we run at least two phases of math fidelity and consume all the bits for bfp8a/b.
            if (get_precision_bits(activations->output_df()) > 4 and
                get_num_fidelity_phases(op->math_fidelity()) < 2)
            {
                // For sparse matmul, bump up the math fidelity 
                op->set_math_fidelity(MathFidelity::HiFi2);
            }
        }
    }
}

// Convert unsupported formats to provided fallbacks (or best available)
void lower_fallback_data_formats(graphlib::Graph *graph, DataFormat fp32_fallback, bool fp32_acc_supported)
{
    if (fp32_acc_supported)
        return;

    // Currently, this pass will lower all fp32 into fp32_fallback, and that's all...
    // TODO: handle integers and other such types
    for (Node *node : graph->nodes())
    {
        if (node->output_df() == DataFormat::Float32)
            node->set_output_df(fp32_fallback);

        if (node->node_type() != graphlib::NodeType::kBudaOp)
            continue;

        graphlib::BudaOpNode *op = node->as<graphlib::BudaOpNode>();
        if (!fp32_acc_supported &&
            (op->accumulate_df() == DataFormat::Float32 or op->intermediate_df() == DataFormat::Float32))
        {
            op->set_accumulate_df(fp32_fallback);
            op->set_intermediate_df(fp32_fallback);
        }
    }
}

// Apply user overrides
void configure_output_data_formats(
    graphlib::Graph *graph, std::optional<DataFormat> default_df_override)
{

    for (Node *node : graph->nodes())
    {
        bool disallow_default_override = is_integer_data_format(node->output_df());
        if (default_df_override and not disallow_default_override)
        {
            node->set_output_df(preserve_lower_precision_cast(node->output_df(), *default_df_override));
        }
    }
}

void configure_input_data_formats(graphlib::Graph *graph)
{
    // All operands must be aligned to the same exponent size (i.e. a vs. b). Math is then done at that exponent size.
    auto first_non_constant_input = [](std::vector<graphlib::Node *> const &operands)
    {
        TT_ASSERT(not operands.empty());
        for (auto *node : operands)
        {
            if (not is_constant_input(node))
                return node;
        }
        return operands[0];
    };

    for (Node *node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() == graphlib::NodeType::kBudaOp)
        {
            // Convert constant input data formats to match that of other inputs
            auto operands = graph->data_operands(node);
            DataFormat df = first_non_constant_input(operands)->output_df();
            bool is_b = is_b_data_format(df);
            if (not std::all_of(
                    operands.begin(),
                    operands.end(),
                    [df, is_b](auto const *n)
                    { return (df == DataFormat::Float32) or (is_b_data_format(n->output_df()) == is_b); }))
            {
                for (auto *operand : operands)
                {
                    bool input_is_b_data_format = is_b_data_format(operand->output_df());
                    if (is_constant_input(operand) and input_is_b_data_format != is_b)
                        operand->set_output_df(
                            input_is_b_data_format ? to_a_data_format(operand->output_df())
                                                   : to_b_data_format(operand->output_df()));
                }
            }
        }
    }
}

static std::vector<ExpPrecision> get_exponent_conversion_preference(
    const std::vector<DataFormat> &operand_dfs,
    const std::optional<DataFormat> default_df_override)
{
    int num_a_formats = std::count_if(operand_dfs.begin(), operand_dfs.end(), is_a_data_format);
    int num_b_formats = std::count_if(operand_dfs.begin(), operand_dfs.end(), is_b_data_format);

    // if equal, use the default_df to break the tie.
    if (num_a_formats == num_b_formats)
    {
        return (default_df_override and is_a_data_format(*default_df_override)) 
            ? std::vector<ExpPrecision>{ExpPrecision::A, ExpPrecision::B}
            : std::vector<ExpPrecision>{ExpPrecision::B, ExpPrecision::A};
    }
    else if (num_a_formats > num_b_formats)
    {
        return {ExpPrecision::A, ExpPrecision::B};
    }
    else
    {
        return {ExpPrecision::B, ExpPrecision::A};
    }
}

static bool is_match_precision_data_format(DataFormat df, ExpPrecision precision)
{
    return (precision == ExpPrecision::A and is_a_data_format(df)) or
           (precision == ExpPrecision::B and is_b_data_format(df));
}

void cast_input_data_formats(graphlib::Graph *graph, Node *node, ExpPrecision to_precision)
{
    const std::unordered_map<ExpPrecision, std::function<DataFormat(DataFormat)>> conversion_function = {
        {ExpPrecision::B, to_a_data_format},
        {ExpPrecision::A, to_b_data_format}
    };
    ExpPrecision from_precision = to_precision == ExpPrecision::A ? ExpPrecision::B : ExpPrecision::A;
    const std::function<DataFormat(DataFormat)> convert_df_function = conversion_function.at(from_precision);

    for (Node *operand : graph->data_operands(node))
    {
        if (is_match_precision_data_format(operand->output_df(), from_precision))
        {
            operand->set_output_df(convert_df_function(operand->output_df()));
        }
    }
}

void cast_and_resolve_input_data_formats(
    graphlib::Graph *graph,
    Node *node,
    const std::vector<DataFormat> &input_data_formats,
    const std::optional<DataFormat> default_df_override)
{
    std::vector<ExpPrecision> conversion_preference =
        get_exponent_conversion_preference(input_data_formats, default_df_override);
    
    // Try to cast to the preferred exponent size first. We can always fallback to
    // casting to FP32 if we can't cast to the preferred exponent size.
    // For now, let's keep it simple and select first conversion preference.
    log_debug(LogGraphCompiler, "{} contains inputs with mixed a/b data formats: {}", node->name(), input_data_formats);

    ExpPrecision preferred_precision = conversion_preference.at(0);
    cast_input_data_formats(graph, node, preferred_precision);

    std::vector<DataFormat> updated_input_data_formats;
    for (const auto &operand : graph->data_operands(node))
    {
        updated_input_data_formats.push_back(operand->output_df());
    }
    log_debug(LogGraphCompiler, "{} updated input_data_formats: {}", node->name(), updated_input_data_formats);
}

void configure_a_b_format_conversion(
    graphlib::Graph *graph, const DeviceConfig &device_config, const std::optional<DataFormat> default_df_override)
{
    // To ensure all input data formats are aligned to the same exponent width,
    // we can issue a/b format conversion on the packer of the input op
    // to make sure we conform to this constraint.
    for (Node *node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() == graphlib::NodeType::kBudaOp)
        {
            graphlib::BudaOpNode *op = dynamic_cast<graphlib::BudaOpNode *>(node);
            std::vector<DataFormat> all_data_formats = get_data_formats(graph->data_operands(op));
            if (not are_data_formats_same_exponent_widths(all_data_formats))
            {
                cast_and_resolve_input_data_formats(graph, op, all_data_formats, default_df_override);
            }
        }
    }

    for (Node *node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() == graphlib::NodeType::kBudaOp)
        {
            graphlib::BudaOpNode *op = dynamic_cast<graphlib::BudaOpNode *>(node);

            if (device_config.is_grayskull() and is_exponent_width_reconfigured(op->accumulate_df(), op->output_df()))
            {
                const std::vector<DataFormat> valid_target_conversions = {
                    DataFormat::Float32, DataFormat::Float16, DataFormat::Float16_b};

                if (std::find(valid_target_conversions.begin(), valid_target_conversions.end(), op->output_df()) ==
                    valid_target_conversions.end())
                {
                    // Format promotion to a valid target conversion
                    DataFormat target_df =
                        is_b_data_format(op->output_df()) ? DataFormat::Float16_b : DataFormat::Float16;
                    log_warning(
                        "Op {} is performing a/b format conversion but not supported for grayskull. Setting output_df "
                        "from {} to {}",
                        op->name(),
                        op->output_df(),
                        target_df);
                    op->set_output_df(target_df);
                }
            }
        }
    }
}

//
// Fix illegal situations
//
// Current rules:
// 1. On Grayskull, the output can convert from a to b exponent for FP16, or when writing out FP32. On Wormhole,
//    BFP* formats can also convert from a to b when packing.
// 2. Intermediate format is currently only used for matmul, but for non-matmul we should follow the rule that
//    intermed df == output df
// 3. Matmul is a special case where operand 1, intermed df, and output df must all match
// 4. Acc_df must match math format on Grayskull, but can be either math format or FP32 on Wormhole
// 5. Untilize op can't be in Bfp* formats
//
// When fixing, try not to change formats that were specifically overriden by the user
void fix_data_formats(graphlib::Graph *graph, bool fp32_acc_supported)
{
    for (Node *node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() == graphlib::NodeType::kBudaOp)
        {
            graphlib::BudaOpNode *op = dynamic_cast<graphlib::BudaOpNode *>(node);
            TT_ASSERT(op);
            if (op->op_name() == "nop" and not op->is_gradient_op())
            {
                op->set_accumulate_df(get_inferred_accumulate_df(graph, op, fp32_acc_supported));
                op->set_intermediate_df(op->accumulate_df());
                op->set_output_df(graph->data_operands(op)[0]->output_df());
            }

            // Both sparse and reduce z don't have support for reconfiguring unpacker from spill,
            // so they must have input_df == output_df == intermediate_df == accumulate_df
            bool is_reduce_z = op->op_name() == "reduce" and std::get<std::string>(op->buda_attrs().at("dim")) == "z";
            if (op->is_sparse_matmul() or is_reduce_z)
            {
                int input_idx = op->is_sparse_matmul() ? 1 : 0;
                std::vector<graphlib::Node *> data_operands = graph->data_operands(op);
                TT_ASSERT((int)data_operands.size() > input_idx);
                auto df = data_operands[input_idx]->output_df();
                op->set_output_df(df);
                op->set_intermediate_df(df);
                op->set_accumulate_df(get_inferred_accumulate_df(graph, op, fp32_acc_supported));
            }

            if (op->is_embedding() || op->is_tilize())
            {
                std::vector<graphlib::Node *> data_operands = graph->data_operands(op);
                TT_ASSERT(not data_operands.empty());
                graphlib::Node* table = data_operands[0];
                if (get_precision_bits(table->output_df()) < 16)
                {
                    table->set_output_df(
                        is_b_data_format(table->output_df()) ? DataFormat::Float16_b : DataFormat::Float16);
                }
            }

            if (op->is_gradient_op())
            {
                std::string reason;
                reason += op->is_gradient_op() ? "Gradient Op " : "";
                log_warning(
                    LogGraphCompiler,
                    "intermed_df must have same exponent size as output_df for packer programming, falling "
                    "back intermed_df: {} Reason: {} Op: {}",
                    op->output_df(),
                    reason,
                    op->name());
                auto gradient_accum_df = get_highest_precision_data_format({op->output_df(), op->accumulate_df()});
                op->set_intermediate_df(gradient_accum_df);
                op->set_output_df(gradient_accum_df);
            }
            if (op->op_type() == "splice")
            {
                std::vector<DataFormat> non_constants_data_formats = get_data_formats(get_non_constants(graph->data_operands(op)));
                std::vector<DataFormat> input_data_formats = get_data_formats(graph->data_operands(op));
                if (not are_data_formats_same(input_data_formats))
                {
                    // we require that all input data formats to a splice are the same.
                    // cast all inputs to the the highest precision input data format
                    DataFormat highest_precision_df = get_highest_precision_data_format(non_constants_data_formats);
                    log_debug(LogGraphCompiler,
                              "Splice op {} has inputs with different data formats, casting all inputs to {}",
                              op->name(),
                              highest_precision_df);
                    for (auto &operand : graph->data_operands(op))
                    {
                        log_debug(LogGraphCompiler, "Splice op {} casting input {} to {}", op->name(), operand->name(), highest_precision_df);
                        operand->set_output_df(highest_precision_df);
                    }
                }
                op->set_output_df_from_operands(graph);
            }
            if (is_configured_for_int8(graph, node))
            {
                if (op->intermediate_df() != DataFormat::Int32)
                {
                    log_warning(
                        "Op {} is configured for Int8, but intermediate_df != Int32. "
                        "Setting intermediate_df from {} to Int32.",
                        op->name(),
                        op->intermediate_df());
                    op->set_intermediate_df(DataFormat::Int32);
                }
                if (op->accumulate_df() != DataFormat::Int32)
                {
                    log_warning(
                        "Op {} is configured for Integer Data Formats, but accumulate_df != Int32. "
                        "Setting accumulate_df from {} to Int32.",
                        op->name(),
                        op->accumulate_df());
                    op->set_accumulate_df(DataFormat::Int32);
                }
                if (op->output_df() != DataFormat::Int8 
                    and op->op_name() != "dequantization"
                    and op->buda_attrs().find("has_dequant") == op->buda_attrs().end())
                {
                    if (op->buda_attrs().find("has_requant") != op->buda_attrs().end()) {
                        log_warning(
                            "Op {} is configured for Int8, but output_df != Int8. "
                            "Setting output_df from {} to Int8.",
                            op->name(),
                            op->output_df());
                        op->set_output_df(DataFormat::Int8);
                    } else if (op->is_matmul()){
                        log_warning(
                            "Op {} is configured for Int8, but output_df != Int8. "
                            "Setting output_df from {} to Int8.",
                            op->name(),
                            op->output_df());
                        op->set_output_df(DataFormat::Int32);
                    } else {
                        log_warning(
                            "Op {} is configured for Int8, but output_df != Int8. "
                            "Setting output_df from {} to Int8.",
                            op->name(),
                            op->output_df());
                        op->set_output_df(DataFormat::Int8);
                    }
                }
            }
            else if (is_configured_for_int32(graph, node))
            {
                if (op->intermediate_df() != DataFormat::Int32)
                {
                    log_warning(
                        "Op {} is configured for Int32, but intermediate_df != Int32. "
                        "Setting intermediate_df from {} to Int32.",
                        op->name(),
                        op->intermediate_df());
                    op->set_intermediate_df(DataFormat::Int32);
                }
                if (op->accumulate_df() != DataFormat::Int32)
                {
                    log_warning(
                        "Op {} is configured for Integer Data Formats, but accumulate_df != Int32. "
                        "Setting accumulate_df from {} to Int32.",
                        op->name(),
                        op->accumulate_df());
                    op->set_accumulate_df(DataFormat::Int32);
                }
                if (op->output_df() != DataFormat::Int32 and op->op_name() != "dequantization")
                {
                    // Requantization must be applied
                    if (not (op->buda_attrs().find("has_requant") != op->buda_attrs().end() and
                             std::get<bool>(op->buda_attrs().at("has_requant")))) {
                        log_warning(
                            "Op {} is configured for Int32, but output_df != Int32. "
                            "Setting output_df from {} to Int32.",
                            op->name(),
                            op->output_df());
                        op->set_output_df(DataFormat::Int32);
                    }
                }
            }
        }
        else if (node->node_type() == graphlib::NodeType::kQueue)
        {
            // The producer may have had its output_df modified. We need to update the output_df 
            // of user-defined queues so that queue->output_df() reflects the producer output_df.
            auto producer = graph->data_operands(node)[0];
            node->set_output_df(producer->output_df());
        }
        else if (node->node_type() == graphlib::NodeType::kOutput)
        {
            auto output_op = graph->data_operands(node)[0];
            if (node->as<graphlib::OutputNode>()->untilize())
            {
                if ((output_op->output_df() == DataFormat::Bfp8_b) || (output_op->output_df() == DataFormat::Bfp4_b) ||
                    (output_op->output_df() == DataFormat::Bfp2_b) || (output_op->output_df() == DataFormat::Float16_b))
                {
                    output_op->set_output_df(DataFormat::Float16_b);
                }
                else if (
                    (output_op->output_df() == DataFormat::Bfp8) || (output_op->output_df() == DataFormat::Bfp4) ||
                    (output_op->output_df() == DataFormat::Bfp2) || (output_op->output_df() == DataFormat::Float16))
                {
                    output_op->set_output_df(DataFormat::Float16);
                }
            }

            std::vector<graphlib::Edge> partial_datacopy_edges = graph->user_partial_datacopy_edges(node);
            if (not partial_datacopy_edges.empty())
            {
                // Current queue is aliased to an existing queue. Impose constraint on write-back producer
                auto consumer_node_id = partial_datacopy_edges.front().consumer_node_id;
                auto aliased_queue = graph->node_by_id(consumer_node_id);
                if (output_op->output_df() != aliased_queue->output_df())
                {
                    log_warning(
                        "Op ({}) writing to aliased queue ({}) must have matching data-formats."
                        "Overriding {} output_df to {}.",
                        output_op->name(),
                        aliased_queue->name(),
                        output_op->output_df(),
                        aliased_queue->output_df());
                    output_op->set_output_df(aliased_queue->output_df());
                }
            }

            node->set_output_df(output_op->output_df());
        }
    }
}

void configure_accumulation_data_formats(
    graphlib::Graph *graph,
    const std::optional<DataFormat> default_accumulate_df,
    bool fp32_acc_supported)
{
    // apply user-overrides if set, otherwise infer using the input data formats loaded into src registers
    for (Node *node : graph->nodes())
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
            continue;

        graphlib::BudaOpNode *op = node->as<graphlib::BudaOpNode>();
        std::optional<DataFormat> acc_df_override = std::nullopt;
        if (default_accumulate_df)
        {
            acc_df_override = default_accumulate_df;
        }

        // if override found, apply it else infer it
        if (acc_df_override)
        {
            if (not fp32_acc_supported and *acc_df_override == DataFormat::Float32)
            {
                *acc_df_override = get_inferred_accumulate_df(graph, op, fp32_acc_supported);
                log_warning(
                    LogGraphCompiler,
                    "Accumulation data-format override for op: {} is set to Float32, but FP32 accumulation is not "
                    "supported on this device. Fall back to highest inferred precision accumulation data-format: {}.",
                    op->name(),
                    *acc_df_override);
            }

            if (not is_valid_accumulate_df(*acc_df_override))
            {
                auto fallback = get_inferred_accumulate_df(graph, op, fp32_acc_supported);
                log_warning(
                    LogGraphCompiler,
                    "Invalid accumulation data-format override for op: {} {}, fallback to {}",
                    op->name(),
                    *acc_df_override,
                    fallback);
                *acc_df_override = fallback;
            }

            op->set_accumulate_df(*acc_df_override);
        }
        else
        {
            op->set_accumulate_df(get_inferred_accumulate_df(graph, op, fp32_acc_supported));
        }
    }
}

void configure_intermediate_data_formats(graphlib::Graph *graph)
{
    // apply user-overrides if set, otherwise infer using the input data formats loaded into src registers
    for (Node *node : graph->nodes())
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
            continue;

        graphlib::BudaOpNode *op = node->as<graphlib::BudaOpNode>();
        op->set_intermediate_df(op->accumulate_df());

        if (op->op_type() == "reduce" 
            and std::get<std::string>(op->buda_attrs().at("dim")) == "z"
            and op->output_df() == DataFormat::Int8)
        {
            op->set_intermediate_df(op->output_df());
        }
    }
}

void fix_math_fidelity(graphlib::Graph *graph)
{
    bool disable_cap_sparse_mm_fidelity = env_as<bool>("PYBUDA_DISABLE_CAP_SPARSE_MM_FIDELITY", false);
    
    for (Node *node : graph->nodes())
    {
        if (node->node_type() == graphlib::NodeType::kBudaOp)
        {
            graphlib::BudaOpNode *op = node->as<graphlib::BudaOpNode>();
            if (bool enable_int8_mode = is_configured_for_int8(graph, op);
                enable_int8_mode and op->math_fidelity() != MathFidelity::HiFi4)
            {
                log_warning(
                    "Node {} is configured for int8, but math fidelity is not HiFi4. "
                    "Setting math fidelity from {} to HiFi4.",
                    op->name(),
                    op->math_fidelity());
                op->set_math_fidelity(MathFidelity::HiFi4);
            }
            else if (op and op->is_sparse_matmul() and not disable_cap_sparse_mm_fidelity)
            {
                std::vector<graphlib::Node *> data_operands = graph->data_operands(op);
                TT_ASSERT(data_operands.size() >= 2);
                graphlib::Node* sparse_matrix = data_operands[0];

                // data-aware opt -- mantissas should be zeros; we'll additionally check for bfp2a/b 
                if (get_precision_bits(sparse_matrix->output_df()) <= 2)
                {
                    auto capped_math_fidelity = MathFidelity::HiFi2;
                    op->set_math_fidelity(capped_math_fidelity);
                }
            }
        }
    }
}

void validate_data_formats(const graphlib::Graph *graph, const DeviceConfig& device_config)
{
    for (Node *node : graph->nodes())
    {
        if (node->node_type() == graphlib::NodeType::kBudaOp)
        {
            graphlib::BudaOpNode *op = node->as<graphlib::BudaOpNode>();
            std::vector<DataFormat> all_data_formats = get_data_formats(graph->data_operands(op));

            if (op->is_gradient_op())
            {
                TT_LOG_ASSERT(
                    op->intermediate_df() == op->output_df(),
                    "For fused, gradient ops, we currently constrain intermediate_df: {} == output_df: {} \
                    because gradient results are not pushed into the output buffer/stream.",
                    op->intermediate_df(),
                    op->output_df());
            }
            if (op->is_fused_op())
            {
                all_data_formats.push_back(op->intermediate_df());

                TT_LOG_ASSERT(are_data_formats_same_exponent_widths(all_data_formats),
                    "For fused ops, we expect all data formats to be of the same type. (a or b type)\
                    Data formats for {}: {}",
                    op->name(),
                    all_data_formats);
            }
            if (op->is_sparse_matmul())
            {
                std::vector<graphlib::Node *> data_operands = graph->data_operands(op);
                TT_ASSERT(data_operands.size() >= 3);

                // assert the sparse matmul encodings are RawUInt* type
                TT_ASSERT(data_operands[2]->output_df() == DataFormat::RawUInt8 or
                          data_operands[2]->output_df() == DataFormat::RawUInt16 or
                          data_operands[2]->output_df() == DataFormat::RawUInt32);
            }
            if (op->is_embedding())
            {
                std::vector<graphlib::Node *> data_operands = graph->data_operands(op);
                TT_ASSERT(data_operands.size() == 2);
                graphlib::Node* table = data_operands[0];
                graphlib::Node *indices = data_operands[1];
                TT_LOG_ASSERT(
                    get_precision_bits(table->output_df()) > 8, "Embedding table must be f16 or f32 precision");
                TT_LOG_ASSERT(
                    is_integer_data_format(indices->output_df()), "Embedding indices must be of integer format");
            }
            if (op->op_type() == "splice")
            {
                std::vector<DataFormat> operand_data_formats = get_data_formats(graph->data_operands(op));
                TT_LOG_ASSERT(
                    are_data_formats_same(operand_data_formats),
                    "For splice op: {}, we expect all data formats to be the same: {}",
                    op->name(),
                    operand_data_formats);
            }
            if (op->op_type() == "reduce" and std::get<std::string>(op->buda_attrs().at("dim")) == "z")
            {
                all_data_formats.push_back(op->intermediate_df());
                all_data_formats.push_back(op->output_df());
                TT_LOG_ASSERT(
                    are_data_formats_same(all_data_formats),
                    "For reduce op: {}, we expect all data formats to be the same: {}",
                    op->name(),
                    all_data_formats);
            }
            if (is_configured_for_int8(graph, op) or is_configured_for_int32(graph, op))
            {
                // Operation is configured for Int8, there are several constraints we must follow
                TT_LOG_ASSERT(
                    op->math_fidelity() == MathFidelity::HiFi4,
                    "op: {}, math_fidelity: {}: If op is configured for Int8, math fidelity must be HiFi4.",
                    op->name(),
                    op->math_fidelity());
                TT_LOG_ASSERT(
                    op->accumulate_df() == DataFormat::Int32,
                    "op: {}, accumulate_df: {}: If op is configured for Int8/Int32, accumulate data format must be Int32.",
                    op->name(),
                    op->accumulate_df());
                TT_LOG_ASSERT(
                    device_config.is_wormhole_b0() && !device_config.is_blackhole(),
                    "op: {}, arch: {}: Int8/Int32 is only supported on Wormhole B0.",
                    op->name(),
                    device_config.arch_name);
            }
            if (graphlib::is_eltwise_binary(op) or op->is_splice()) {
                TT_LOG_ASSERT(
                    are_data_formats_all_float(all_data_formats) or are_data_formats_all_integer(all_data_formats),
                    "All input data formats should either be all float or all integer. Data formats for {}: {}",
                    op->name(),
                    all_data_formats);
            }
            if (device_config.is_grayskull() and is_exponent_width_reconfigured(op->accumulate_df(), op->output_df()))
            {
                // if a/b format is reconfigured, we want to capture grayskull-specific limitations
                // see BBE#1437. Format conversion using BFP8A/B disallowed.
                std::vector<DataFormat> valid_target_conversions = {DataFormat::Float32, DataFormat::Float16, DataFormat::Float16_b};
                TT_LOG_ASSERT(
                    std::find(valid_target_conversions.begin(), valid_target_conversions.end(), op->output_df()) != valid_target_conversions.end(),
                    "op: {}, accumulate_df: {}, output_df: {}. For grayskull, we can only do format conversions to {}.",
                    op->name(),
                    op->accumulate_df(),
                    op->output_df(),
                    valid_target_conversions
                );
            }

            all_data_formats.push_back(op->intermediate_df());
            TT_LOG_ASSERT(are_data_formats_same_exponent_widths(all_data_formats),
                "All input data formats to be of the same type. (a or b type) Data formats for {}: {}",
                op->name(),
                all_data_formats);

            TT_ASSERT(is_valid_accumulate_df(op->accumulate_df()));
        }
        else if (node->node_type() == graphlib::NodeType::kQueue)
        {
            auto producer = graph->data_operands(node).at(0);
            TT_LOG_ASSERT(
                producer->output_df() == node->output_df(),
                "Queue: {} is configured for data format: {}, but producer: {} is configured for data format: {}",
                node->name(),
                node->output_df(),
                producer->name(),
                producer->output_df());
        }
        else if (node->node_type() == graphlib::NodeType::kOutput)
        {
            auto producer = graph->data_operands(node).at(0);
            std::vector<graphlib::Edge> partial_datacopy_edges = graph->user_partial_datacopy_edges(node);
            if (not partial_datacopy_edges.empty())
            {
                // Current queue is aliased to an existing queue. Impose constraint on write-back producer
                auto consumer_node_id = partial_datacopy_edges.front().consumer_node_id;
                auto aliased_queue = graph->node_by_id(consumer_node_id);

                TT_LOG_ASSERT(
                    producer->output_df() == aliased_queue->output_df(),
                    "Producer Op ({}) output df ({}) must have matching data-formats as ({}) aliased queue ({}).",
                    producer->name(),
                    producer->output_df(),
                    aliased_queue->name(),
                    aliased_queue->output_df());
            }
        }
    }
}

void validate_post_placer_data_formats(const graphlib::Graph *graph, const DeviceConfig& device_config)
{
    if (not device_config.is_grayskull())
    {
        return;
    }

    for (Node *node : graph->nodes())
    {
        if (node->node_type() == graphlib::NodeType::kBudaOp)
        {
            if (graphlib::BudaOpNode *op = node->as<graphlib::BudaOpNode>(); op->is_matmul())
            {
                const auto& buda_attrs = op->buda_attrs();
                if (auto mk_it = buda_attrs.find("m_k"); mk_it != buda_attrs.end() and std::get<int>(mk_it->second) > 1)
                {
                    std::vector<DataFormat> all_data_formats = get_data_formats(graph->data_operands(op));
                    all_data_formats.push_back(op->intermediate_df());
                    all_data_formats.push_back(op->accumulate_df());
                    all_data_formats.push_back(op->output_df());

                    TT_LOG_ASSERT(are_data_formats_same_exponent_widths(all_data_formats),
                        "All input data formats to be of the same type. (a or b type) Data formats for {}: with m_k > 1. dfs={}",
                        op->name(),
                        all_data_formats);
                }
            }
        }
    }
}

void configure_stochastic_rounding(graphlib::Graph *graph, const bool is_stochastic_rounding_supported)
{
    bool enable_stochastic_rounding = env_as<bool>("PYBUDA_ENABLE_STOCHASTIC_ROUNDING", false);

    // For WH_B0, we support a few different flavours of stochastic rounding
    // 1. No stochastic rounding - fpu default: RN, sfpu default: RNE (supported)
    // 2. FPU/SFPU Stochastic rounding:
    //       - fpu: final result in dst (supported)
    //       - sfpu: final result in dst (unsupported: pending bbe#1297)
    // 3. Stochastic Rounding in the packer during format conversion (unsupported: pending bbe):
    // This flag toggles between 1 and 2 above where support is present
    if (not enable_stochastic_rounding)
    {
        return;
    }
    if (not is_stochastic_rounding_supported)
    {
        return;
    }

    for (Node *node : graphlib::topological_sort(*graph))
    {
        if (graphlib::BudaOpNode *op = dynamic_cast<graphlib::BudaOpNode *>(node); op != nullptr)
        {
            const bool use_stochastic_rounding = op->accumulate_df() != DataFormat::Float32;
            if ((op->is_matmul() or graphlib::is_eltwise_binary(op)) and use_stochastic_rounding)
            {
                if (op->accumulate_df() == DataFormat::Float32)
                {
                    log_fatal(LogGraphCompiler, "User has requested stochastic rounding but accumulate_df = Float32");
                }

                BudaOpAttrs buda_attrs = op->buda_attrs();
                buda_attrs["srnd_fpu_en"] = use_stochastic_rounding;
                op->overwrite_buda_attrs(buda_attrs);
            }
        }
    }
}

void satisfy_data_format_constraints(
    graphlib::Graph *graph,
    bool fp32_acc_supported)
{
    fix_data_formats(graph, fp32_acc_supported);
    fix_math_fidelity(graph);
}

void run_dataformat_passes(
    graphlib::Graph *graph,
    const DeviceConfig &device_config,
    const std::optional<DataFormat> default_df_override,
    const std::optional<DataFormat> default_accumulate_df,
    const DataFormat fp32_fallback,
    const MathFidelity default_math_fidelity,
    const int amp_level,
    const std::vector<AMPNodeProperties> &amp_properties)
{
    // Apply user overrides
    configure_output_data_formats(graph, default_df_override);

    // Convert unsupported formats to provided fallbacks (or best available)
    lower_fallback_data_formats(graph, fp32_fallback, device_config.supports_fp32_accumulation());

    configure_input_data_formats(graph);

    configure_a_b_format_conversion(graph, device_config, default_df_override);

    configure_accumulation_data_formats(
        graph, default_accumulate_df, device_config.supports_fp32_accumulation());

    configure_intermediate_data_formats(graph);

    // Apply math fidelity
    apply_math_fidelity(graph, default_math_fidelity);

    // Fix illegal situations
    satisfy_data_format_constraints(graph, device_config.supports_fp32_accumulation());

    // Apply automatic mixed precision based on user-defined levels
    run_automatic_mixed_precision(graph, device_config, default_df_override, amp_level, amp_properties);

    configure_stochastic_rounding(graph, device_config.supports_stochastic_rounding());

    validate_data_formats(graph, device_config);
}

}  // namespace tt::passes
