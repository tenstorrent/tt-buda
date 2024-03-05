# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torch.fx import subgraph_rewriter
from typing import Any, Tuple

# Decompose
def decompose_split(self: torch.Tensor, split_size: int, dim: int = 0) -> Tuple[torch.Tensor, ...]:
    starts = list(range(0, self.size(dim), split_size))
    stops = starts[1:] + [self.size(dim)]
    slices = []
    for start, stop in zip(starts, stops):
        slices.append(self.narrow(dim, start, stop - start))
    return slices

def decompose_matmul(bias, input, weight) -> torch.Tensor:
    res = torch.matmul(input, weight)
    res = torch.add(res, bias)
    return res

pybuda_decompositions = {
   torch.ops.aten.split.Tensor: decompose_split,
   torch.ops.aten.addmm.default: decompose_matmul,
}

def get_pybuda_decompositions():
    return pybuda_decompositions

# Reconstruct
class ReconstructBilinearResize2d():
    @staticmethod
    def pattern(x, scale, output_size, input_size, device):
        arange = torch.ops.aten.arange.start_step(0, output_size, dtype = torch.float32, layout = torch.strided, device = device, pin_memory = False)
        arange_1 = torch.ops.aten.arange.start_step(0, output_size, dtype = torch.float32, layout = torch.strided, device = device, pin_memory = False)
        mul = torch.ops.aten.mul.Tensor(arange, scale)
        mul_1 = torch.ops.aten.mul.Tensor(arange_1, scale)
        _to_copy = torch.ops.aten._to_copy.default(mul, dtype = torch.int64)
        ceil = torch.ops.aten.ceil.default(mul)
        clamp = torch.ops.aten.clamp.default(ceil, None, input_size)
        _to_copy_1 = torch.ops.aten._to_copy.default(clamp, dtype = torch.int64)
        _to_copy_2 = torch.ops.aten._to_copy.default(mul_1, dtype = torch.int64)
        ceil_1 = torch.ops.aten.ceil.default(mul_1)
        clamp_1 = torch.ops.aten.clamp.default(ceil_1, None, input_size)
        _to_copy_3 = torch.ops.aten._to_copy.default(clamp_1, dtype = torch.int64)
        unsqueeze = torch.ops.aten.unsqueeze.default(mul, 1)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(_to_copy, 1)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(_to_copy_1, 1)
        index = torch.ops.aten.index.Tensor(x, [None, None, unsqueeze_1, _to_copy_2])
        index_1 = torch.ops.aten.index.Tensor(x, [None, None, unsqueeze_2, _to_copy_2])
        index_2 = torch.ops.aten.index.Tensor(x, [None, None, unsqueeze_1, _to_copy_3])
        index_3 = torch.ops.aten.index.Tensor(x, [None, None, unsqueeze_2, _to_copy_3])
        sub = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1)
        sub_1 = torch.ops.aten.sub.Tensor(1.0, sub)
        sub_2 = torch.ops.aten.sub.Tensor(mul_1, _to_copy_2)
        sub_3 = torch.ops.aten.sub.Tensor(1.0, sub_2)
        mul_2 = torch.ops.aten.mul.Tensor(index, sub_1)
        mul_3 = torch.ops.aten.mul.Tensor(index_1, sub)
        add = torch.ops.aten.add.Tensor(mul_2, mul_3)
        mul_4 = torch.ops.aten.mul.Tensor(index_2, sub_1)
        mul_5 = torch.ops.aten.mul.Tensor(index_3, sub)
        add_1 = torch.ops.aten.add.Tensor(mul_4, mul_5)
        mul_6 = torch.ops.aten.mul.Tensor(add, sub_3)
        mul_7 = torch.ops.aten.mul.Tensor(add_1, sub_2)
        add_2 = torch.ops.aten.add.Tensor(mul_6, mul_7)
        return add_2

    @staticmethod
    def replacement(x, scale, output_size, input_size, device):
        return torch.nn.functional.interpolate(x, size=output_size, mode='bilinear', align_corners=True)

def apply_torch_reconstruct_patterns(aten):
    patterns = [
        ReconstructBilinearResize2d(),
    ]

    for p in patterns:
        subgraph_rewriter.replace_pattern_with_filters(aten, p.pattern, p.replacement)

