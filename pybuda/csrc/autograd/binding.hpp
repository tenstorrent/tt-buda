// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// Interface to python definitions of op shapes & backward generators

#pragma once

#include "utils/assert.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"

#include "autograd/autograd.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace tt::autograd2;
using Shape = tt::graphlib::Shape;
using OpType = tt::graphlib::OpType;
using DimBroadcast = tt::graphlib::DimBroadcast;
using TileDim = tt::TileDim;

std::tuple<Shape, std::vector<DimBroadcast>> get_op_shape(
    OpType type, std::vector<Shape> &operands, bool is_buda, TileDim tile_dim = TileDim::Dim32x32);
std::tuple<Shape, std::vector<DimBroadcast>> get_fused_op_shape(tt::graphlib::BudaOpNode *op, std::vector<Shape> &operands);
inline Shape get_tm_shape(OpType type, Shape operand, bool is_buda)
{
    Shape shape;
    std::vector<Shape> operands = {operand};
    std::vector<DimBroadcast> bcast;
    std::tie(shape, bcast) = ::get_op_shape(type, operands, is_buda, operand.get_tile_dim());
    return shape;
}
NodeContext insert_backward(
    autograd_context context,
    OpType type,
    int operand, 
    const std::vector<NodeContext> &inputs,
    NodeContext output,
    NodeContext gradient);

