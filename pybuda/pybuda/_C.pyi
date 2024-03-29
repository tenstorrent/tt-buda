from typing import ClassVar, Dict, List

Backward: NodeEpochType
Bfp2: DataFormat
Bfp2_b: DataFormat
Bfp4: DataFormat
Bfp4_b: DataFormat
Bfp8: DataFormat
Bfp8_b: DataFormat
Float16: DataFormat
Float16_b: DataFormat
Float32: DataFormat
Forward: NodeEpochType
HiFi2: MathFidelity
HiFi3: MathFidelity
HiFi4: MathFidelity
Int8: DataFormat
Invalid: DataFormat
Lf8: DataFormat
LoFi: MathFidelity
Optimizer: NodeEpochType
UInt16: DataFormat
VERSION: int

class BudaNetlist:
    def __init__(self) -> None: ...
    def append_comment(self, arg0: str) -> None: ...
    def dump_to_yaml(self) -> str: ...

class BudaNetlistConfig:
    def __init__(self) -> None: ...

class DataFormat:
    __doc__: ClassVar[str] = ...  # read-only
    __members__: ClassVar[dict] = ...  # read-only
    Bfp2: ClassVar[DataFormat] = ...
    Bfp2_b: ClassVar[DataFormat] = ...
    Bfp4: ClassVar[DataFormat] = ...
    Bfp4_b: ClassVar[DataFormat] = ...
    Bfp8: ClassVar[DataFormat] = ...
    Bfp8_b: ClassVar[DataFormat] = ...
    Float16: ClassVar[DataFormat] = ...
    Float16_b: ClassVar[DataFormat] = ...
    Float32: ClassVar[DataFormat] = ...
    Int8: ClassVar[DataFormat] = ...
    Invalid: ClassVar[DataFormat] = ...
    Lf8: ClassVar[DataFormat] = ...
    UInt16: ClassVar[DataFormat] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class MathFidelity:
    __doc__: ClassVar[str] = ...  # read-only
    __members__: ClassVar[dict] = ...  # read-only
    HiFi2: ClassVar[MathFidelity] = ...
    HiFi3: ClassVar[MathFidelity] = ...
    HiFi4: ClassVar[MathFidelity] = ...
    Invalid: ClassVar[MathFidelity] = ...
    LoFi: ClassVar[MathFidelity] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class NodeEpochType:
    __doc__: ClassVar[str] = ...  # read-only
    __members__: ClassVar[dict] = ...  # read-only
    Backward: ClassVar[NodeEpochType] = ...
    Forward: ClassVar[NodeEpochType] = ...
    Optimizer: ClassVar[NodeEpochType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class PostPlacerConfig:
    def __init__(self) -> None: ...

class PytorchTensorDesc:
    def __init__(self, arg0: int, arg1: int, arg2: DataFormat, arg3: List[int[4]], arg4: List[int[4]]) -> None: ...
    def format(self) -> DataFormat: ...
    def ptr(self) -> int: ...
    def shape(self) -> List[int[4]]: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...

def dump_epoch_id_graphs(graph: graph.Graph, test_name: str, graph_name: str, placer_solution: placer.PlacerSolution) -> None: ...
def dump_epoch_type_graphs(graph: graph.Graph, test_name: str, graph_name: str, placer_solution: placer.PlacerSolution = ...) -> None: ...
def dump_graph(graph: graph.Graph, test_name: str, graph_name: str, placer_solution: placer.PlacerSolution = ..., balancer_solution: balancer.BalancerSolution ...) -> None: ...
def lower_to_buda_netlist(arg0: graph.Graph, arg1: str, arg2: int, arg3: placer.PlacerSolution, arg4: balancer.BalancerSolution) -> BudaNetlist: ...
def run_post_placer_buda_passes(arg0: graph.Graph, arg1: placer.PlacerSolution, arg2: PostPlacerConfig, arg2: balancer.BalancerSolution) -> None: ...
def run_pre_placer_buda_passes(arg0: graph.Graph) -> graph.Graph: ...
