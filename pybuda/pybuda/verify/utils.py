# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch

class CPUCombiner(torch.nn.Module):
    """ 
    Generic CPU module that takes any number of inputs and combines them into one
    """
    def forward(self, *inputs):
        if len(inputs) == 1:
            return inputs[0] + 0
        
        # Expand all inputs to 4 dims
        inputs = list(inputs)
        for i in range(len(inputs)):
            inputs[i] = inputs[i].reshape((*[1 for _ in range(4 - len(inputs[i].shape))], *inputs[i].shape))

        def max_shape(shapes):
                mshp = [1]*4
                for i in range(-1, -4, -1):
                    mx = -1
                    for shape in shapes:
                        if len(shape) < -i:
                            continue
                        if shape[i] > mx:
                            mx = shape[i]
                    mshp[i] = mx
                return tuple(mshp)

        def get_pad(shape, new_shape):
            pad = []
            for j in range(-1, -len(shape), -1):
                pad.append(new_shape[j] - shape[j])
                pad.append(0)
            return pad

        new_shape = max_shape([input.shape for input in inputs])
        pad = get_pad(inputs[0].shape, new_shape)
        sum = torch.nn.functional.pad(inputs[0], tuple(pad))
        for i in range(1, len(inputs)):
            pad = get_pad(inputs[i].shape, new_shape)
            sum = sum + torch.nn.functional.pad(inputs[i], tuple(pad))

        return sum

class LossMultiplier(torch.nn.Module):
    """
    L1Loss module that allows providing an arbitrary multiplicand
    """
    def __init__(self, multiplicand):
        super().__init__()
        self.multiplicand = multiplicand
        self.l1loss = torch.nn.L1Loss()

    def forward(self, output, target):
        loss = self.l1loss.forward(output, target)
        high_loss = loss * self.multiplicand

        return high_loss

