// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "autograd/binding.hpp"
#include "passes/fuse_ops.hpp"

#include <vector>

std::tuple<Shape, std::vector<DimBroadcast>> get_op_shape(OpType type, std::vector<Shape> &operands, bool is_buda, TileDim tile_dim)
{
    int tile_height = tt::graphlib::get_row_size_from_tile_size(tile_dim);
    int tile_width = tt::graphlib::get_col_size_from_tile_size(tile_dim);
    auto eval_module = is_buda ? py::module_::import("pybuda.op.eval.buda") : py::module_::import("pybuda.op.eval.pybuda");
    py::function pybuda_shape = is_buda ? eval_module.attr("get_f_pybuda_shape")(type, tile_height, tile_width)
                                        : eval_module.attr("get_f_pybuda_shape")(type);

    std::vector<std::vector<std::uint32_t>> operand_tuples;
    for(Shape &shape : operands)
        operand_tuples.push_back(shape.as_vector());

    py::tuple ret = pybuda_shape(operand_tuples);
    Shape s = is_buda ? Shape::create_buda(ret[0].cast<std::vector<std::uint32_t>>(), tile_height, tile_width) : 
                        Shape::create(ret[0].cast<std::vector<std::uint32_t>>());

    return std::make_tuple(s, ret[1].cast<std::vector<DimBroadcast>>());
}

std::tuple<Shape, std::vector<DimBroadcast>> get_fused_op_shape(tt::graphlib::BudaOpNode *op, std::vector<Shape> &operands)
{
    std::unordered_map<std::uint32_t, Shape> buffers;
    std::vector<DimBroadcast> dim_broadcasts;
    std::optional<Shape> dest;
    for (auto schedule : op->get_fused_op()->get_schedules())
    {
        for (auto sub_op : schedule.ops)
        {
            std::vector<Shape> sub_op_inputs;
            for (tt::FusedSubOpInput i : sub_op.inputs)
            {
                if (i.type == tt::FusedSubOpInput::InputType::INPUT)  {
                    TT_ASSERT(i.index < operands.size(), "Refering to input that doesn't exist for fused op");
                    sub_op_inputs.push_back(operands.at(i.index));
                }
                else if (i.type == tt::FusedSubOpInput::InputType::DEST) {
                    TT_ASSERT(dest.has_value(), "Reading from dest that has not value");
                    sub_op_inputs.push_back(dest.value());
                    dest = std::nullopt;
                }
                else {
                    auto it = buffers.find(i.index);
                    TT_ASSERT(it != buffers.end(), "Referring to intermediate buffer that doesn't exist");
                    sub_op_inputs.push_back(it->second);
                }

                // All inputs to the fused op are already properly broadcasted.
                // But for the sub-op inputs which are outputs of previously executed sub-ops,
                // we need to apply broadcast.
                // NOTE: We don't need to apply tile broadcasts for shape calculation, since each
                // input is at least the size of a tile.
                if (i.type != tt::FusedSubOpInput::InputType::INPUT
                    && i.has_broadcast())
                {
                    Shape operand_shape = sub_op_inputs.back();

                    int broadcast_dim = i.broadcast.first;
                    int broadcast_factor = i.broadcast.second;

                    OpType broadcast_op = OpType("broadcast", {broadcast_dim, broadcast_factor}, {});
                    std::vector<Shape> shapes = {operand_shape};
                    std::tuple<Shape, std::vector<DimBroadcast>> shape_data = get_op_shape(broadcast_op, shapes, true);

                    operand_shape = std::get<0>(shape_data);

                    sub_op_inputs.pop_back();
                    sub_op_inputs.emplace_back(operand_shape);
                }
            }

            Shape result;
            std::vector<DimBroadcast> broadcast;
            tie(result, broadcast) = get_op_shape(sub_op.op_type, sub_op_inputs, true);

            if (sub_op.output_type == tt::FusedSubOp::OutputType::OUTPUT)
                return std::make_pair(result, dim_broadcasts);

            else if (sub_op.output_type == tt::FusedSubOp::OutputType::DEST)
                dest = result;
            
            else {
                // intermed
                if (buffers.count((std::uint32_t)sub_op.output_buffer) == 0)
                    buffers.insert(std::make_pair((std::uint32_t)sub_op.output_buffer, result));
                else
                    buffers[(std::uint32_t)sub_op.output_buffer] = result;
            }
        }
    }
    TT_THROW("Evaluated the full fused op, but haven't reached the output shape.");
    return std::make_pair(Shape(), std::vector<DimBroadcast>{});
}

NodeContext insert_backward(
    autograd_context context,
    OpType type,
    int operand, 
    const std::vector<NodeContext> &inputs,
    NodeContext output,
    NodeContext gradient)
{
    auto eval_module = py::module_::import("pybuda.op.eval.pybuda");
    py::function pybuda_backward = eval_module.attr("get_f_pybuda_backward")(type);

    return pybuda_backward(context, operand, inputs, output, gradient).cast<NodeContext>();
}

