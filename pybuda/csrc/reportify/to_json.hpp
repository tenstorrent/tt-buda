// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "json_fwd.hpp"

#include "lower_to_buda/common.hpp"
#include "graph_lib/node_types.hpp"
#include "placer/placer.hpp"
#include "balancer/types.hpp"
#include "balancer/legalizer/graph_solver.hpp"

namespace std
{
template <typename... Ts>
void to_json(json& j, variant<Ts...> const& v)
{
    visit([&j](auto&& elem) { j = elem; }, v);
}
}  // namespace std

namespace tt {
inline void to_json(json& j, DramLoc const& dram_loc) { j = std::make_pair(dram_loc.channel, dram_loc.address); }
}

namespace tt {
namespace graphlib
{
void to_json(json& j, UBlockOrder const& ublock_order);
void to_json(json& j, OpType const& op_type);
void to_json(json& j, EdgeAttributes const& attrs);
}  // namespace graphlib

namespace balancer
{
std::string to_string(TStreamDir const& dir);
void to_json(json& j, TStreamFactor const& tsr);
void to_json(json& j, BlockShape const& block_shape);
void to_json(json& j, BufferModel const& buffer_model);
void to_json(json& j, TensorShape const& shape);
void to_json(json& j, OpModel const& op_model);
void to_json(json& j, GridShape const& grid_shape);
namespace legalizer
{
void to_json(json& j, GraphSolver::ConstraintInfo::Page const& constraint_info_page);
void to_json(json& j, GraphSolver::ConstraintInfo const& constraint_info);
}  // namespace legalizer
}  // namespace balancer
}  // namespace tt
