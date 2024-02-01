# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from enum import Enum

import pybuda._C.graph.query as query


class NodePredicateType(Enum):
    Negate = 1
    And = 2
    Or = 3
    NameRegex = 4
    LayerRegex = 5
    OpType = 6

    def eval_fn(self):
        fns = {
            NodePredicateType.NameRegex: query.name_regex,
            NodePredicateType.LayerRegex: query.layer_regex,
            NodePredicateType.OpType: query.op_type,
        }

        if self not in fns:
            raise NotImplemented()
        return fns[self]


class NodePredicateBuilder:
    def __init__(self, ty: NodePredicateType, value):
        self.ty = ty
        self.value = value

    def negate(self):
        return NodePredicateBuilder(NodePredicateType.Negate, self)

    def __and__(self, other):
        return NodePredicateBuilder(NodePredicateType.And, (self, other))

    def __or__(self, other):
        return NodePredicateBuilder(NodePredicateType.Or, (self, other))

    def eval(self):
        if self.ty == NodePredicateType.Negate:
            assert type(self.value) is NodePredicateBuilder
            return self.value.eval().negate()
        elif self.ty == NodePredicateType.And or self.ty == NodePredicateType.Or:
            assert type(self.value) is tuple
            assert len(self.value) == 2
            assert type(self.value[0]) is NodePredicateBuilder
            assert type(self.value[1]) is NodePredicateBuilder
            a = self.value[0].eval()
            b = self.value[1].eval()
            if self.ty == NodePredicateType.And:
                return a & b
            else:
                assert self.ty == NodePredicateType.Or
                return a | b
        else:
            assert type(self.value) is str
            return self.ty.eval_fn()(self.value)

    def _value_to_json(self):
        if self.ty == NodePredicateType.Negate:
            assert type(self.value) is NodePredicateBuilder
            return self.value.to_json()
        elif self.ty == NodePredicateType.And or self.ty == NodePredicateType.Or:
            assert type(self.value) is tuple
            assert len(self.value) == 2
            assert type(self.value[0]) is NodePredicateBuilder
            assert type(self.value[1]) is NodePredicateBuilder
            return [self.value[0].to_json(), self.value[1].to_json()]
        else:
            assert type(self.value) is str
            return self.value

    def to_json(self):
        return {
            "type": self.ty.name,
            "value": self._value_to_json(),
        }

    @staticmethod
    def _value_from_json(ty, value):
        if ty == NodePredicateType.Negate:
            assert type(value) is dict
            return NodePredicateBuilder.from_json(value)
        elif ty == NodePredicateType.And or ty == NodePredicateType.Or:
            assert type(value) is list
            assert len(value) == 2
            return (
                NodePredicateBuilder.from_json(value[0]),
                NodePredicateBuilder.from_json(value[1]),
            )
        else:
            assert type(value) is str
            return value

    @staticmethod
    def from_json(j):
        assert "type" in j
        assert "value" in j
        assert type(j["type"]) is str
        ty = NodePredicateType[j["type"]]
        value = NodePredicateBuilder._value_from_json(ty, j["value"])
        return NodePredicateBuilder(ty, value)


def name_regex(regex: str):
    return NodePredicateBuilder(NodePredicateType.NameRegex, regex)


def layer_regex(regex: str):
    return NodePredicateBuilder(NodePredicateType.LayerRegex, regex)


def op_type(name: str):
    return NodePredicateBuilder(NodePredicateType.OpType, name)


def _simple_test():
    a = name_regex("a")
    b = layer_regex("b")
    c = op_type("c")
    d = a & b
    e = c | d
    f = e.negate()
    j = f.to_json()
    g = NodePredicateBuilder.from_json(j)
    print(j)
    print(g.to_json())
    assert j == g.to_json()
