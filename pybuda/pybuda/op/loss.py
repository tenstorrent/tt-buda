# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from ..module import PyBudaModule
from .reduce import ReduceSum, ReduceAvg
from .eltwise_unary import Exp, Log, Abs
from .eltwise_binary import Subtract, Multiply
from .constant import Constant

class CrossEntropyLoss(PyBudaModule):
    """
    Cross-Entropy Loss
    """
    def forward(self, predictions, labels):
        exponential_predicted = Exp("exp", predictions)
        reduced_exponential_predicted = ReduceSum(
            "reduce_sum", exponential_predicted, dim=-1
        )
        ln_reduced_exponential_predicted = Log("log", reduced_exponential_predicted)
        output_minus_ln_reduced_exponential_predicted = Subtract(
            "subtract", predictions, ln_reduced_exponential_predicted
        )
        negative_one_constant = Constant("negative_one_const", constant=-1.0)
        negative_one_times_output_minus_ln_reduced_exponential_predicted = Multiply(
            "multiply",
            output_minus_ln_reduced_exponential_predicted,
            negative_one_constant,
        )
        word_losses = Multiply(
            "word_losses",
            negative_one_times_output_minus_ln_reduced_exponential_predicted,
            labels,
        )
        token_losses = ReduceSum("token_losses", word_losses, dim=-1)
        avg_sequence_loss = ReduceAvg("avg_sequence_loss", token_losses, dim=-2)

        out = avg_sequence_loss

        return out


class L1Loss(PyBudaModule):
    """
    L1Loss

    L1Loss is abs(labels - prediction), optionally reduced using ReduceAvg(default) or ReduceSum.
    """

    def __init__(self, name: str, reduction:str  = "avg"):
        super().__init__(name)
        self.reduction = reduction

    def forward(self, prediction, labels):
        diff = Abs("abs", Subtract("sub", prediction, labels))

        if self.reduction == "none":
            return diff

        # TODO: z reduce?
        if self.reduction == "avg":
            r_reduce = ReduceAvg("r_avg", diff, -2)
            c_reduce = ReduceAvg("c_avg", r_reduce, -1)
            return c_reduce

        if self.reduction == "sum":
            r_reduce = ReduceSum("r_avg", diff, -2)
            c_reduce = ReduceSum("c_avg", r_reduce, -1)
            return c_reduce

        raise RuntimeError("Unsupported reduce type: " + self.reduction)

