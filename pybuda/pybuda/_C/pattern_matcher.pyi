import pybuda._C.graph
from typing import Dict, List, Tuple

class MatchResult:
    is_subgraph_loopable: bool
    is_subgraph_pattern_found: bool
    subgraph_matches: List[Dict[int, int]]
    def __init__(self, *args, **kwargs) -> None: ...

def lower_json_to_pattern_matcher(arg0: json, arg1: int) -> MatchResult: ...
def lower_pybuda_to_pattern_matcher(arg0: pybuda._C.graph.Graph, arg1: int) -> Tuple[pybuda._C.graph.Graph, MatchResult]: ...
