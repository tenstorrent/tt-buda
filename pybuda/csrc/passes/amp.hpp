// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <optional>
#include <variant>
#include <unordered_map>
#include <regex>

#include "graph_lib/graph.hpp"
#include "lower_to_buda/common.hpp"
#include "graph_lib/defines.hpp"
#include "third_party/json/json.hpp"
#include "shared_utils/json_extension.hpp"


namespace tt
{

struct DeviceConfig;

} // namespace tt

namespace tt::graphlib
{
class Graph;
class Node;
}

namespace tt::passes
{

using DataFormat = tt::DataFormat;
using MathFidelity = tt::MathFidelity;
using NodeEpochType = tt::graphlib::NodeEpochType;
using InputIndexToConfig = std::map<std::uint32_t, std::pair<DataFormat, bool>>;
using InputDfConfig = std::variant<InputIndexToConfig, DataFormat, std::monostate>;
struct AMPNodeProperties
{
    std::optional<std::string> op_type;
    std::optional<NodeEpochType> epoch_type;

    std::optional<DataFormat> output_df;

    // math-op specific
    std::optional<DataFormat> intermediate_df;
    std::optional<DataFormat> accumulate_df;
    std::optional<MathFidelity> math_fidelity;

    std::optional<std::string> name_regex_match;
    std::optional<InputDfConfig> input_df;
    std::optional<bool> is_gradient_op;
    std::optional<std::vector<std::pair<std::uint32_t, std::uint32_t>>> input_parameter_indices_to_optimize;

    AMPNodeProperties(
        std::optional<std::string> op_type = std::nullopt,
        std::optional<NodeEpochType> epoch_type = std::nullopt,
        std::optional<DataFormat> output_df = std::nullopt,
        std::optional<DataFormat> intermediate_df = std::nullopt,
        std::optional<DataFormat> accumulate_df = std::nullopt,
        std::optional<MathFidelity> math_fidelity = std::nullopt,
        std::optional<std::string> name_regex_match = std::nullopt,
        std::optional<InputDfConfig> input_df = std::nullopt,
        std::optional<bool> is_gradient_op = std::nullopt,
        std::optional<std::vector<std::pair<std::uint32_t, std::uint32_t>>> input_parameter_indices_to_optimize = std::nullopt) :
        op_type(op_type),
        epoch_type(epoch_type),
        output_df(output_df),
        intermediate_df(intermediate_df),
        accumulate_df(accumulate_df),
        math_fidelity(math_fidelity),
        name_regex_match(name_regex_match),
        input_df(input_df),
        is_gradient_op(is_gradient_op),
        input_parameter_indices_to_optimize(input_parameter_indices_to_optimize)
    {
    }
};

class RegexMatcher {
private:
    std::unordered_map<std::string, std::regex> regex_cache;

public:
    bool has_matching_string(const std::string& regex_string, const std::string& candidate_string);
};

bool is_matched_op(AMPNodeProperties &amp_properties, RegexMatcher &regex_matcher, const graphlib::Node* node);

void to_json(nlohmann::json& j, const AMPNodeProperties& p);
void from_json(const nlohmann::json& j, AMPNodeProperties& p);

void run_automatic_mixed_precision(
    graphlib::Graph *graph,
    const DeviceConfig &device_config,
    const std::optional<DataFormat> default_df_override,
    const int amp_level,
    const std::vector<AMPNodeProperties>& amp_properties);

nlohmann::json write_mixed_precision_json(graphlib::Graph *graph);
void dump_mixed_precision_json_to_file(graphlib::Graph *graph, std::optional<std::string> file_path = std::nullopt);

}
