// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "reportify/to_json.hpp"

#include "json.hpp"

namespace tt
{
namespace graphlib
{
void to_json(json& j, UBlockOrder const& ublock_order)
{
    switch (ublock_order)
    {
        case UBlockOrder::R: j = "R"; break;
        case UBlockOrder::C: j = "C"; break;
        default: j = "UBlockOrder::Unknown"; break;
    }
}

void to_json(json& j, OpType const& op_type)
{
    j["op_type"] = {};
    j["op_type"]["type"] = op_type.op;
    j["op_type"]["attrs"] = op_type.attr;
    j["op_type"]["buda_attrs"] = op_type.buda_attrs;
    j["op_type"]["named_attrs"] = op_type.named_attrs;
}

void to_json(json& j, EdgeAttributes const& attrs)
{
    j["ublock_order"] = attrs.get_ublock_order();
    j["tms"] = attrs.get_tms();
}
}  // namespace graphlib

namespace balancer
{
std::string to_string(TStreamDir const& dir)
{
    switch (dir.v)
    {
        case TStreamDir::R: return "R";
        case TStreamDir::C: return "C";
        case TStreamDir::RZ: return "RZ";
        case TStreamDir::CZ: return "CZ";
        default: return "Unknown";
    }
}

void to_json(json& j, TStreamFactor const& tsr)
{
    j["dir"] = tsr.none() ? "None" : to_string(tsr.dir);
    j["factor"] = {tsr.r, tsr.c};
}

void to_json(json& j, BlockShape const& block_shape)
{
    j["t"] = block_shape.t;
    j["tblock_m"] = block_shape.tblock_m;
    j["tblock_n"] = block_shape.tblock_n;
    j["mblock_m"] = block_shape.mblock_m;
    j["mblock_n"] = block_shape.mblock_n;
    j["ublock_rt"] = block_shape.ublock.rt;
    j["ublock_ct"] = block_shape.ublock.ct;
}

void to_json(json& j, BufferModel const& buffer_model)
{
    j["block_shape"] = buffer_model.block_shape;
    j["buffer_factor"] = buffer_model.buffer_factor;
    j["l1_size_tiles"] = buffer_model.l1_size_tiles;
    std::stringstream ss;
    ss << buffer_model.data_format;
    j["data_format"] = ss.str();
    j["kernel_broadcast_tiles"] = buffer_model.kernel_broadcast_tiles;
}

void to_json(json& j, TensorShape const& shape) { j = {shape.w, shape.z, shape.rt, shape.ct}; }

void to_json(json& j, OpModel const& op_model)
{
    j["op_model_id"] = op_model.id.id;
    if (op_model.buda_op_node)
        j["op_type"] = op_model.op_type();
    j["grid_shape"] = op_model.grid_shape;
    j["t_stream_factor"] = op_model.t_stream_factor;
    j["inputs"] = op_model.input_buffers;
    j["outputs"] = op_model.output_buffers;
    j["input_shapes"] = op_model.op_shape.inputs;
    j["execution_cycles"] = op_model.cached_execution_cycles;
}

void to_json(json& j, GridShape const& grid_shape) { j = {grid_shape.r, grid_shape.c}; }
namespace legalizer
{
void to_json(json& j, GraphSolver::ConstraintInfo::Page const& info)
{
    j["node_id_order"] = info.node_id_order;
    j["op_models"] = info.id_to_op_models;
    j["node_op_models"] = info.node_id_to_op_model_ids;
    j["edge_path_sets"] = info.edge_to_path_sets;
    j["failure_reasons"] = info.failure_reason_ids;
}

void to_json(json& j, GraphSolver::ConstraintInfo const& info)
{
    j["graph_name"] = info.graph_name;
    j["paged"] = true;
    j["num_pages"] = info.pages.size();
    j["page_size"] = GraphSolver::ConstraintInfo::kPageSize;
    j["op_model_selection"] = info.op_model_selection;
    j["node_names"] = info.node_id_to_name;
    j["node_pages"] = info.node_name_to_page;
    j["error_codes"] = ConstraintFailureReasonDesc;
}

}  // namespace legalizer
}  // namespace balancer
}  // namespace tt
