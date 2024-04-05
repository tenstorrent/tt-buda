# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

from enum import Enum
import pkg_resources
from typing import Tuple, Dict, List, Optional, Union, Set
from collections.abc import Iterable
from dataclasses import dataclass, field
from pybuda._C import DataFormat, MathFidelity, NopInsertionInstruction, AMPNodeProperties, DramQueueConfigOverride
import pybuda._C.balancer as pybalancer
import pybuda._C.placer as pyplacer
from pybuda._C.backend_api import DeviceMode, load_cached_sys_param
import pybuda.query as query
from dataclasses_json import dataclass_json, config

from pybuda.utils import as_json, dict_as_json, list_as_json, optional_as_json, resolve_output_build_directory, resolve_device_descriptor_path
from loguru import logger


class CompileDepth(Enum):
    INIT_COMPILE = 0
    GENERATE_INITIAL_GRAPH = 1
    POST_INITIAL_GRAPH_PASS = 2
    CONSTEVAL_GRAPH = 3
    POST_PATTERN_MATCHER = 4
    OPTIMIZED_GRAPH = 5
    AUTOGRAD = 6
    POST_AUTOGRAD_PASS = 7
    PRE_LOWERING_PASS = 8
    BUDA_GRAPH_PRE_PLACER = 9
    BALANCER_PASS = 10
    PRE_NETLIST_PASS = 11
    GENERATE_NETLIST = 12
    BACKEND_GOLDEN_VERIFY = 13
    FULL = 14

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_ 

    @classmethod
    def to_json(cls, value):
        return value.name

    @classmethod
    def from_json(cls, value):
        return cls[value.upper()]

class PerfTraceLevel(Enum):
    NONE = 0                    # No backend performance trace data will be captured
    LIGHT = 1                   # Basic op start/end times will be captured
    VERBOSE = 2                 # Detailed set of events will be captured. This could have negative impact on performance.

    @classmethod
    def to_json(cls, value):
        return value.name

    @classmethod
    def from_json(cls, value):
        return cls[value.upper()]

    def get_backend_cfg_string(self) -> str:
        if self.value == PerfTraceLevel.NONE.value:
            return ""

        if self.value == PerfTraceLevel.LIGHT.value:
            return "--dump-perf-events --perf-level 0 --perf-target-inputs 0,1,2,-1,-2 --perf-suppress-warnings"

        if self.value == PerfTraceLevel.VERBOSE.value:
            #return "--dump-perf-events-intermediate --perf-level 1 --perf-target-inputs 0,1,2,3,-1,-2,-3 --perf-suppress-warnings"
            return "--dump-perf-events-intermediate --perf-level 1 --perf-target-inputs 0,1,2,35,36,37,38,39,40,41,42,43,44,45,46,47,56,57,58,59,60,61,62,-1 --perf-suppress-warnings"
            # The below command (with concurrent instead of intermediate) is useful when some traces result in out-of-memory errors
            # However, it hasn't been extensively tested that I know of, so won't turn it into the default setting just yet...
            # return "--dump-perf-events-concurrent --perf-level 1 --perf-target-inputs 0,1,2,35,36,37,38,39,40,41,42,43,44,45,46,47,56,57,58,59,60,61,62,-1 --perf-suppress-warnings"

        raise RuntimeError("Unsupported level")

class PlacerOpOverridesAsJson:
    @classmethod
    def to_json(cls, value):
        assert type(value) is tuple
        if type(value[0]) is str:
            lhs = value[0]
        else:
            lhs = value[0].to_json()
        return [lhs, value[1].to_json()]

    @classmethod
    def from_json(cls, value):
        assert type(value) is list
        if type(value[0]) is str:
            lhs = value[0]
        else:
            lhs = value[0].to_json()
        return [lhs, pyplacer.OpOverride.from_json(value[1])]

class PlacerBreaksAsJson:
    @classmethod
    def to_json(cls, value):
        if type(value) is query.NodePredicateBuilder:
            return value.to_json()
        elif type(value) is list:
            return [PlacerBreaksAsJson.to_json(v) for v in value]
        else:
            assert type(value) is str
            return value

    @classmethod
    def from_json(cls, value):
        if type(value) is dict:
            return query.NodePredicateBuilder.from_json(value)
        elif type(value) is list:
            return [PlacerBreaksAsJson.from_json(v) for v in value]
        else:
            assert type(value) is str
            return value


class TTIDumpFormat(Enum):
    """
    Enumerates the supported formats for dumping a TTDeviceImage to disk.
    This specifies the serialization format for tensor data.
    """
    DEFAULT = "DEFAULT"
    BACKEND = "BACKEND"
    BACKEND_TILIZED = "BACKEND_TILIZED"

    def extension(self) -> str:
        if self == TTIDumpFormat.DEFAULT:
            return "pkl"
        elif self == TTIDumpFormat.BACKEND:
            return "bin"
        elif self == TTIDumpFormat.BACKEND_TILIZED:
            return "tbin"

    @classmethod
    def to_json(cls, value):
        return value.name

    @classmethod
    def from_json(cls, value):
        return cls[value.upper()]


@dataclass_json
@dataclass
class CompilerConfig:
    enable_training: bool = False           # enable training; run autograd & training passes
    enable_recompute: bool = False          # enable training recompute during autograd
    match_subgraph_patterns: Optional[int] = None   # invokes pattern_matcher to compact isomorphic subgraphs

    balancer_policy: str = "default"        # balancer policy to determine decision making on grid-shapes, blocks etc
    scheduler_policy: str = "ModuleInputsBFS" # scheduler policy to determine ordering of the ops submitted to be placed by placer
    enable_t_streaming: bool = True         # enable flattening r/c dims into t for minimal buffering
    manual_t_streaming: bool = False        # only respect overrides, by default no streaming
    enable_consteval: bool = True           # enable promotion of nodes to be constant evaluated where possible
    enable_auto_fusing: bool = True         # enable automatic fusing of ops
    compile_subgraphs: bool = False         # Compile each disjoint graph separately into its own program
    graph_solver_self_cut_type: str = "FastCut" # which type of self-cut to use for graphsolver
    use_interactive_placer: bool = True     # use interactive placer if chosen policy supports it
    enable_enumerate_u_kt: bool = True      # Enable searching all possible matmul u_kts
    enable_link_past_cache_ios: bool = False # Enable auto detection and linking of past key-value pairs
    enable_pt2_fx_graph_link: bool = False  # Enable linking of past key-value pairs in the graph

    compile_depth: int = field(default=CompileDepth.FULL, metadata=as_json(CompileDepth))  # Defines compilation depth. Used to limit scope of some unit tests

    enable_tvm_cpu_fallback: bool = True    # Create cpu device for unsupported pybuda ops
    cpu_fallback_ops: Set[str] = field(default_factory=lambda: set(["embedding"])) # Types of ops to fall back to CPU for
    enable_tm_cpu_fallback: bool = False    # Extend CPU fallback for TM ops
    tm_cpu_fallback_max_depth: int = 10     # Max search depth for extended CPU fallback

    enable_tvm_dropout: bool = False        # (Temporary): Remove when buda supports dropout
    enable_tvm_unsupported_ops: bool = False# Create "unsupported" pybuda ops in python file, allowing user to modify later
    enable_op_level_comparision: bool = False # Should we need to compare every op with framework output at each compilation stage.
    enable_tvm_constant_prop: bool = False  # Should we constant prop in tvm
    convert_framework_params_to_tvm: bool = True # Convert framework params to relay params
    enable_xla_jax_convert: bool = False    # Convert JAX model to TF through XLA
    enable_tvm_jax_freeze_large_model: bool = True # When model param is larger than 2GB, Protobuf will error out. This flag will enable large model tracing
    framework_model_output_names: List[str] = field(default_factory=lambda: list())   # List of output names specified by framework
    tvm_constnat_prop_mask: Set[str] = field(default_factory=lambda: set())  # Which parameters should be constant propped by tvm
    compile_tvm_to_python: bool = True      # instead of generating a direct pybuda graph from TVM, generate a pybuda python class
    retain_tvm_python_files: bool = False   # Whether to keep generated python code, or load and delete
    tvm_graph_store_path: str = ""          # Defines store path of serilized TVM graphs. 
    tvm_graph_load_path: str = ""           # Defines load path of serilized TVM graphs.
    tvm_module_to_num_patterns: Dict[str, int] = field(default_factory=lambda: dict()) # Number of patterns to match for each module

    enable_conv_prestride: bool = True      # Enables a transform for conv that directly reads input, such that it goes from stride > 1 to stride = 1
                                            # This usually translates to lower DRAM BW and less math as the input better populates tiles
                                            # More can be found here: tenstorrent/budabackend#957
    max_pool_add_sub_surround: bool = False         # Add add op before, and subtract op after max_pool during the decomposition. The reason for 
                                                    # adding it is to tangle with negative values for max_pool, as current decomposition uses sparse
                                                    # matmul which is padded with 0. Therefore, 0 will be maximum value when max_pool is run - which 
                                                    # represents invalid results. 
    max_pool_add_sub_surround_value: float = 1.0    # Value which will be added and subtracted before and after (respectively) around max_pool op. This
                                                    # way, we're able to support negative values even with sparse matmul implementation.

    loopback_outputs: Dict[str, int] = field(default_factory=lambda: dict())  # Outputs will be kept on device and looped back to inputs of subsequent runs
    enable_device_tilize: bool = False # If true, enables Tilize op for embedded platform
    enable_forked_dram_inputs = False # If true, enables forked_dram_inputs optimization

    chip_placement_policy: str = "MMIO_LAST"       # how to order the given chip ids for placement
    op_names_to_epoch_break: List[Union[query.NodePredicateBuilder, List[Union[str, query.NodePredicateBuilder]]]] = field(default_factory=list, metadata=list_as_json(PlacerBreaksAsJson))   # Each op in the list will be placed on a new epoch
    op_names_to_chip_break: List[Union[query.NodePredicateBuilder, List[Union[str, query.NodePredicateBuilder]]]] = field(default_factory=list, metadata=list_as_json(PlacerBreaksAsJson)) # Each op in the list will be placed on a new chip
    op_names_dont_fuse: List[str] = field(default_factory=lambda: list())           # A list of ops to disable being fused
    op_names_manual_fuse: List[str] = field(default_factory=lambda: list())          # A list of ops to allow being fused, non specified ops will no longer participate in fusion
    balancer_op_overrides: Dict[str, pybalancer.OpOverride] = field(default_factory=lambda: dict(), metadata=dict_as_json(pybalancer.OpOverride))  # User override of op balancer attributes (i.e. grid sizes)
    default_dram_parameters: Optional[bool] = None # If set to true/false, place parameters in dram by default i.e. prologue=False/True, if it's None we refer to microbatch-size to set prologue config
    placer_op_overrides: List[Tuple[Union[str, query.NodePredicateBuilder], pyplacer.OpOverride]] = field(default_factory=list, metadata=list_as_json(PlacerOpOverridesAsJson))
    default_df_override: Optional[DataFormat] = field(default=None, metadata=optional_as_json(DataFormat)) # Default override for all node data formats, None means automatically inferred
    default_accumulate_df: Optional[DataFormat] = field(default=None, metadata=optional_as_json(DataFormat)) # Accumulation format, for chips that support it
    enable_broadcast_splitting: bool = False  # if true, large broadcasts will be split into multiple edges with nops between them
    default_math_fidelity: MathFidelity = field(default=MathFidelity.HiFi3, metadata=as_json(MathFidelity)) # default math fidelity for all ops
    performance_trace: PerfTraceLevel = field(default=PerfTraceLevel.NONE, metadata=as_json(PerfTraceLevel)) # backend performance trace level
    amp_level: Optional[int] = None     # Configure Automatic Mixed Precision (AMP) level. By default it's set to 'None' (0), which means no AMP is applied. However, there
                                        # are few levels of AMP that can be applied: 
                                        # 1: Matmuls inputs/outputs are set to BFP8_b;  Fused ops, Softmax, LayerNorm ops are set to FP16_b;
                                        # 2: Matmuls inputs/outputs are set to BFP8;    Fused ops, Softmax, LayerNorm ops are set to FP16;  GELU is BFP8;
                                        # 
                                        # Have in mind that in each AMP level, non-mentioned op types are left with default data format (usually set by user; i.e. FP32).
    harvesting_mask: int = 0 # List of harvested rows (same across all chips)
    enable_auto_transposing_placement: bool = ("PYBUDA_ENABLE_AUTO_TRANSPOSE" in os.environ)  # compiler automatically detects ops to transpose on placement when the flag is set
    fracture_groups: List[Tuple[List[Tuple[str, int, int]], List[str], List[int]]] = field(default_factory=lambda: list()) # see insert_fracture_group
    conv_multi_op_fracture_factor_override: Dict[str, int] = field(default_factory=lambda: dict())  # override multi op fracture factor for conv
    enable_stable_softmax: bool = True
    enable_single_buffer_fallback: bool = False

    device_mode: DeviceMode = field(default=DeviceMode.CompileAndRun, metadata=as_json(DeviceMode))
    backend_opt_level: int = 4 # backend optimization level
    backend_output_dir: str = field(default_factory=lambda: resolve_output_build_directory()) # backend compile and perf trace output directory
    backend_device_descriptor_path: str = ""
    backend_cluster_descriptor_path: str = ""
    backend_runtime_params_path: str = ""
    backend_runtime_args: str = ""
    store_backend_db_to_yaml: bool = False # whether to dump the backend DB to a yaml file or not
    input_queues_on_host: bool = True # whether to place input queues on device
    output_queues_on_host: bool = True # whether to place output queues on device
    manual_dram_queue_placement: Dict[str, DramQueueConfigOverride] =  field(default_factory=lambda: dict(), metadata=dict_as_json(DramQueueConfigOverride)) # manual dram queue placements to target specific chip/dram chan
    buffering_nops_to_insert: Dict[Tuple[str,str,int,int],NopInsertionInstruction] = field(default_factory=lambda: dict(), metadata=dict_as_json(NopInsertionInstruction))
    insert_queues: List[Tuple[str, str, int]] = field(default_factory=lambda: list(), metadata=list_as_json(tuple)) # Insert queues between (producer_op_name, consumer_op_name, input_port_id)
    amp_properties: List[AMPNodeProperties] = field(default_factory=lambda: list(), metadata=list_as_json(AMPNodeProperties))
    scheduler_constraints: List[List[str]] = field(default_factory=lambda: list())
    paddings: Dict[str, bool] = field(default_factory=lambda: dict())
    op_intermediates_to_save: List[str] = field(default_factory=lambda: list()) # list of tagged ops that will spill its output to queue
    tti_dump_format: TTIDumpFormat = field(default=TTIDumpFormat.DEFAULT, metadata=as_json(TTIDumpFormat))
    dram_placement_algorithm: pyplacer.DRAMPlacementAlgorithm = field(default=pyplacer.DRAMPlacementAlgorithm.ROUND_ROBIN, metadata=as_json(pyplacer.DRAMPlacementAlgorithm))

    # TODO: add reportify dir

    def apply_env_config_overrides(self):
        if "PYBUDA_OVERRIDE_NUM_CHIPS" in os.environ:
            self.chip_ids = list(range(int(os.environ.get('PYBUDA_OVERRIDE_NUM_CHIPS'))))

        if "PYBUDA_DISABLE_OP_FUSING" in os.environ:
            self.enable_auto_fusing = False

        if "PYBUDA_PERFORMANCE_TRACE" in os.environ:
            self.performance_trace = {
                    "none": PerfTraceLevel.NONE,
                    "light": PerfTraceLevel.LIGHT,
                    "verbose": PerfTraceLevel.VERBOSE,
                }[os.environ["PYBUDA_PERFORMANCE_TRACE"].lower()]

        if "PYBUDA_COMPILE_DEPTH" in os.environ:
            self.compile_depth = {
                    "full": CompileDepth.FULL,
                    "init_compile": CompileDepth.INIT_COMPILE,
                    "generate_initial_graph": CompileDepth.GENERATE_INITIAL_GRAPH,
                    "post_initial_graph_pass": CompileDepth.POST_INITIAL_GRAPH_PASS,
                    "pre_lowering_pass": CompileDepth.PRE_LOWERING_PASS,
                    "buda_graph_pre_placer": CompileDepth.BUDA_GRAPH_PRE_PLACER,
                    "balancer_pass": CompileDepth.BALANCER_PASS,
                    "generate_netlist": CompileDepth.GENERATE_NETLIST,
                    "post_pattern_matcher": CompileDepth.POST_PATTERN_MATCHER,
                    "backend_golden_verify": CompileDepth.BACKEND_GOLDEN_VERIFY,
                }[os.environ["PYBUDA_COMPILE_DEPTH"].lower()]

        if "PYBUDA_ENABLE_INPUT_QUEUES_ON_HOST" in os.environ:
            self.input_queues_on_host = bool(int(os.environ["PYBUDA_ENABLE_INPUT_QUEUES_ON_HOST"]))

        if "PYBUDA_ENABLE_OUTPUT_QUEUES_ON_HOST" in os.environ:
            self.output_queues_on_host = bool(int(os.environ["PYBUDA_ENABLE_OUTPUT_QUEUES_ON_HOST"]))

        if "PYBUDA_DEFAULT_DRAM_PARAMETERS" in os.environ:
            self.default_dram_parameters = bool(int(os.environ["PYBUDA_DEFAULT_DRAM_PARAMETERS"]))

        if "PYBUDA_PRESTRIDE_DISABLE" in os.environ:
            self.enable_conv_prestride = not bool(int(os.environ["PYBUDA_PRESTRIDE_DISABLE"]))

        if "PYBUDA_DISABLE_STABLE_SOFTMAX" in os.environ:
            self.enable_stable_softmax = not bool(int(os.environ["PYBUDA_DISABLE_STABLE_SOFTMAX"]))

        if "PYBUDA_CONVERT_PARAMS_TO_TVM" in os.environ:
            self.convert_framework_params_to_tvm = bool(int(os.environ["PYBUDA_CONVERT_PARAMS_TO_TVM"]))

        if "PYBUDA_DEFAULT_DF" in os.environ:
            self.default_df_override = DataFormat.from_json(os.environ["PYBUDA_DEFAULT_DF"])

        if "PYBUDA_DISABLE_ENUMERATE_U_KT" in os.environ:
            self.enable_enumerate_u_kt = not bool(int(os.environ["PYBUDA_DISABLE_ENUMERATE_U_KT"]))

        if "PYBUDA_ENABLE_SINGLE_BUFFER_FALLBACK" in os.environ:
            self.enable_single_buffer_fallback = bool(int(os.environ["PYBUDA_ENABLE_SINGLE_BUFFER_FALLBACK"]))

        if "PYBUDA_TTI_BACKEND_FORMAT" in os.environ:
            self.tti_dump_format = TTIDumpFormat.BACKEND

        elif "PYBUDA_TTI_BACKEND_TILIZED_FORMAT" in os.environ:
            self.tti_dump_format = TTIDumpFormat.BACKEND_TILIZED

        if "PYBUDA_AMP_LIGHT" in os.environ:
            self.enable_amp_light(level=int(os.environ["PYBUDA_AMP_LIGHT"]))

        if "PYBUDA_ENABLE_DEVICE_TILIZE" in os.environ:
            self.enable_device_tilize = bool(int(os.environ["PYBUDA_ENABLE_DEVICE_TILIZE"]))
        if "PYBUDA_ENABLE_FORKED_DRAM_INPUTS" in os.environ:
            self.enable_forked_dram_inputs = bool(int(os.environ["PYBUDA_ENABLE_FORKED_DRAM_INPUTS"]))

        if "PYBUDA_SCHEDULER_POLICY" in os.environ:
            self.scheduler_policy = os.environ["PYBUDA_SCHEDULER_POLICY"]

        if "PYBUDA_OVERRIDE_DEVICE_YAML" in os.environ and os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] != "":
            self.backend_device_descriptor_path = resolve_device_descriptor_path(os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"])

    def __post_init__(self):
        self.apply_env_config_overrides()

    def enable_amp_light(self, level: int = 1):
        if level == 0:
            return

        level_to_config = {
            1: (8, MathFidelity.HiFi2),
            2: (4, MathFidelity.HiFi2),
            3: (4, MathFidelity.LoFi),
        }
        mantissa_bits, math_fidelity = level_to_config[level]
        target_mm_weights, target_mm_bias = True, True
        input_parameter_indices = [
            (operand_index, mantissa_bits)
            for use_lower_precision, operand_index in zip((target_mm_weights, target_mm_bias), range(1,3))
            if use_lower_precision
        ]
        self.amp_properties.append(
            AMPNodeProperties(op_type="matmul", math_fidelity=math_fidelity, input_parameter_indices_to_optimize=input_parameter_indices)
        )
    def place_on_new_epoch(self, op_names: Union[str, query.NodePredicateBuilder, List[Union[str, query.NodePredicateBuilder]]]) -> None:
        """
        Given a list of ops, `op_names`, select the op appearing first in the placement schedule and issue an epoch break.
        """
        if isinstance(op_names, str):
            self.op_names_to_epoch_break.append([op_names])
        elif isinstance(op_names, query.NodePredicateBuilder):
            self.op_names_to_epoch_break.append(op_names)
        else:
            assert isinstance(op_names, list)
            self.op_names_to_epoch_break.append(op_names)


    def place_on_new_chip(self, op_names: Union[str, query.NodePredicateBuilder, List[Union[str, query.NodePredicateBuilder]]]) -> None:
        """
        Given a list of ops, `op_names`, select the op appearing first in the placement schedule and issue a chip break.
        """
        if isinstance(op_names, str):
            self.op_names_to_chip_break.append([op_names])
        elif isinstance(op_names, query.NodePredicateBuilder):
            self.op_names_to_chip_break.append(op_names)
        else:
            assert isinstance(op_names, list)
            self.op_names_to_chip_break.append(op_names)

    def place_queue_to_chip_dram(self, dram_queue: str, *, chip_id: Optional[int], channel: Optional[int]) -> None:
        """
        Given a dict of dram queue names to (chip_id, dram_chan), force the placement of these queues
        """
        self.manual_dram_queue_placement[dram_queue] = DramQueueConfigOverride(chip_id, channel)

    def dont_fuse(self, op_names: Union[str, List[str]]) -> None:
        """
        `op_names`: An op name or list of op names to mark as not participating in the fusion graph pass
        """
        if isinstance(op_names, str):
            self.op_names_dont_fuse.append(op_names)
        else:
            assert isinstance(op_names, list)
            assert isinstance(op_names[0], str)
            self.op_names_dont_fuse.extend(op_names)

    def manual_fuse(self, op_names: Union[str, List[str]]) -> None:
        """
        `op_names`: A list of ops to allow being fused, non specified ops will no longer participate in fusion
        """
        if isinstance(op_names, str):
            self.op_names_manual_fuse.append(op_names)
        else:
            assert isinstance(op_names, list)
            assert isinstance(op_names[0], str)
            self.op_names_manual_fuse.extend(op_names)

    def balancer_op_override(self, op_name: str, attribute: str, value):
        op_override = self.balancer_op_overrides.get(op_name, pybalancer.OpOverride())
        if isinstance(value, dict):
            current_value = getattr(op_override, attribute) or dict()
            value = {**current_value, **value}
        setattr(op_override, attribute, value)
        self.balancer_op_overrides[op_name] = op_override

    def save_intermediates(self) -> bool:
        return len(self.op_intermediates_to_save) > 0


def get_harvesting_mask(row_indices: List[int]):
    harvested_rows_mask = 0 
    for r in row_indices:
        harvested_rows_mask += (1 << r)
    return harvested_rows_mask

# Backend runtime yaml path for supported B0 boards
supported_backend_configurations = {
    "wh_n150"      : "tti/runtime_param_yamls/wh_n150_syslevel.yaml",
    "wh_n300"      : "tti/runtime_param_yamls/wh_n300_syslevel.yaml",
    "galaxy"       : "tti/runtime_param_yamls/galaxy_syslevel.yaml",
    "gs_e150"      : "tti/runtime_param_yamls/gs_e150_syslevel.yaml",
    "gs_e300"      : "tti/runtime_param_yamls/gs_e300_syslevel.yaml",
}

# Global compiler configuration
g_compiler_config : CompilerConfig = CompilerConfig()


# 
# User-level API for setting compiler configuration options
#
def set_configuration_options(
        enable_recompute: Optional[bool] = None,
        balancer_policy: Optional[str] = None,
        place_on_one_row: Optional[bool] = None,
        enable_t_streaming: Optional[bool] = None,
        manual_t_streaming: Optional[bool] = None,
        enable_consteval: Optional[bool] = None,
        default_df_override: Optional[DataFormat] = None,
        accumulate_df: Optional[DataFormat] = None,
        math_fidelity: Optional[MathFidelity] = None,
        performance_trace: Optional[PerfTraceLevel] = None,
        backend_opt_level: Optional[int] = None,
        backend_output_dir: Optional[str] = None,
        backend_device_descriptor_path: Optional[str] = None,
        backend_cluster_descriptor_path: Optional[str] = None,
        backend_runtime_params_path: Optional[str] = None,
        backend_runtime_args: Optional[str] = None,
        enable_auto_fusing: Optional[bool] = None,
        enable_conv_prestride: Optional[bool] = None,
        enable_stable_softmax: Optional[bool] = None,
        amp_level: Optional[int] = None,
        harvested_rows: Optional[List[List[int]]] = None,
        store_backend_db_to_yaml: Optional[bool] = None,
        input_queues_on_host: Optional[bool] = None,
        output_queues_on_host: Optional[bool] = None,
        enable_auto_transposing_placement: Optional[bool] = None,
        use_interactive_placer: Optional[bool] = None,
        op_intermediates_to_save: Optional[List[str]] = None,
        enable_enumerate_u_kt: Optional[bool] = None,
        enable_device_tilize: Optional[bool] = None,
        dram_placement_algorithm: Optional[pyplacer.DRAMPlacementAlgorithm] = None,
        chip_placement_policy: Optional[str] = None,
        enable_forked_dram_inputs: Optional[bool] = None,
        device_config: Optional[str] = None):
    """
    Set global compile configuration options.

    Parameters
    ----------
    enable_recompute: Optional[bool] 
        For training only. Enable 'recompute' feature which significantly reduces memory requirements at a cost of
        some performance.

    balancer_policy: Optional[str] 
        Override default place & route policy. Valid values are:

        "NLP": Custom policy with reasonable defaults for NLP-like models
        "Ribbon": Custom policy with reasonable defaults for CNN-like models

        [DEBUG ONLY]
        "MaximizeTMinimizeGrid": Maximize t-streaming. Verification only.
        "MinimizeGrid": Super simple policy that always chooses smallest grid. Verification only.
        "Random": Pick random valid grids for each op. Verification only.

        [DEPRECATED]
        "CNN"

    place_on_one_row: Optional[bool] 
        For place & route to place every op on one row of cores only. 

    enable_t_streaming: Optional[bool] 
        Enable buffering optimization which reduces memory usage and latency.

    manual_t_streaming: Optional[bool] 
        Only respect override_t_stream_dir op overrides, otherwise no streaming.
        enable_t_streaming must also be true to take effect.

    enable_consteval: Optional[bool] 
        Use constant propagation to simplify the model.

    default_df_override: Optional[DataFormat], None default
        Set the default override for all node data formats, None means automatically inferred

    accumulate_df: Optional[DataFormat], Float16_b default
        Set default accumulation format for all operations, if supported by the device. 

    math_fidelity: Optional[MathFidelity], MathFidelity.HiFi3 default
        Set default math fidelity for all operations

    performance_trace: Optional[PerfTraceLevel]
        Set to value other than None to enable performance tracing. Note that the Verbose level could have impact on the performance due
        to the amount of data being captured and stored.

    backend_opt_level: Optional[int]
        The level of performance optimization in backend runtime (0-3)

    backend_output_dir: Optional[str]
        Set location for backend compile temporary files and binaries

    backend_device_descriptor_path: Optional[str]
        Set location for YAML file to load device descriptor

    backend_cluster_descriptor_path: Optional[str]
        Set location for YAML file to load multi-device cluster descriptor
    
    backend_runtime_params_path: Optional[str]
        Set location for YAML file to dump/load backend database configurations

    enable_auto_fusing: Optional[bool]
        Enabling automatic fusing of small operations into complex ops

    enable_conv_prestride: Optional[bool]
        Enabling host-side convolution prestiding (occurs during host-tilizer) for more efficient first convolution layer.

    amp_level: Optional[int]
        Configures the optimization setting for Automatic Mixed Precision (AMP).
        0: No Optimization (default)
        1: Optimizer ops are set with { OutputDataFormat.Float32, MathFidelity.HiFi4 }

    harvested_rows: Optional[List[int]]
        Configures manually induced harvested rows. Only row-indices within 1-5 or 7-11 are harvestable.

    store_backend_db_to_yaml: Optional[bool]
        Enabling automatic backend database configuration dump to the YAML file specified with backend_runtime_param_path.
        Note that all backend configurations are loaded from the YAML file if existing YAML file is specified and this flag is set to False.

    use_interactive_placer: Optional[bool]
        Enable or disable usage of interactive placer within balancer policies which support it. Enabled by default.

    enable_device_tilize: Optional[bool]
        Enable or Disable Tilize Op on the embedded platform

    chip_placement_policy: Optional[str]
        Determine the order of the chip ids used in placement

    dram_placement_algorithm: Optional[pyplacer.DRAMPlacementAlgorithm]
        Set the algorithm to use for DRAM placement. Valid values are: ROUND_ROBIN, ROUND_ROBIN_FLIP_FLOP, GREATEST_CAPACITY, CLOSEST
    enable_forked_dram_inputs: Optional[bool]
        Enable or Disable Forked Dram Optimization        

    device_config: Optional[str]
        Configure and Set runtime_param.yaml for offline WH compile based on the value.
        YAML files for supported configurations are mapped at 'supported_backend_configurations'
    """

    global g_compiler_config
    if enable_recompute is not None:
        g_compiler_config.enable_recompute = enable_recompute
    if balancer_policy is not None:
        g_compiler_config.balancer_policy = balancer_policy
    if enable_t_streaming is not None:
        g_compiler_config.enable_t_streaming = enable_t_streaming
    if manual_t_streaming is not None:
        g_compiler_config.manual_t_streaming = manual_t_streaming
    if enable_consteval is not None:
        g_compiler_config.enable_consteval = enable_consteval
    if default_df_override is not None:
        g_compiler_config.default_df_override = default_df_override
    if accumulate_df is not None:
        g_compiler_config.default_accumulate_df = accumulate_df
    if math_fidelity is not None:
        g_compiler_config.default_math_fidelity = math_fidelity
    if performance_trace is not None:
        g_compiler_config.performance_trace = performance_trace
    if backend_opt_level is not None:
        assert backend_opt_level >= 0 and backend_opt_level <= 4, "Backend opt level must be 0-4"
        g_compiler_config.backend_opt_level = backend_opt_level
    if backend_output_dir is not None:
        g_compiler_config.backend_output_dir = backend_output_dir
    if backend_device_descriptor_path is not None:
        g_compiler_config.backend_device_descriptor_path = resolve_device_descriptor_path(backend_device_descriptor_path)
    if backend_cluster_descriptor_path is not None:
        g_compiler_config.backend_cluster_descriptor_path = backend_cluster_descriptor_path
    if backend_runtime_params_path is not None:
        g_compiler_config.backend_runtime_params_path = backend_runtime_params_path
    if backend_runtime_args is not None:
        g_compiler_config.backend_runtime_args = backend_runtime_args
    if enable_auto_fusing is not None:
        g_compiler_config.enable_auto_fusing = enable_auto_fusing
    if enable_conv_prestride is not None:
        g_compiler_config.enable_conv_prestride = enable_conv_prestride
    if enable_stable_softmax is not None:
        g_compiler_config.enable_stable_softmax = enable_stable_softmax
    if amp_level is not None:
        g_compiler_config.amp_level = amp_level
    if harvested_rows is not None:
        g_compiler_config.harvesting_mask = get_harvesting_mask(harvested_rows)
    if store_backend_db_to_yaml is not None:
        g_compiler_config.store_backend_db_to_yaml = store_backend_db_to_yaml
    if input_queues_on_host is not None:
        g_compiler_config.input_queues_on_host = input_queues_on_host
    if output_queues_on_host is not None:
        g_compiler_config.output_queues_on_host = output_queues_on_host
    if enable_auto_transposing_placement is not None:
        g_compiler_config.enable_auto_transposing_placement = g_compiler_config.enable_auto_transposing_placement or enable_auto_transposing_placement
    if use_interactive_placer is not None:
        g_compiler_config.use_interactive_placer = use_interactive_placer
    if enable_enumerate_u_kt is not None:
        g_compiler_config.enable_enumerate_u_kt = enable_enumerate_u_kt
    if op_intermediates_to_save is not None:
        g_compiler_config.op_intermediates_to_save = op_intermediates_to_save 
    if enable_device_tilize is not None:
        g_compiler_config.enable_device_tilize = enable_device_tilize
    if chip_placement_policy is not None:
        g_compiler_config.chip_placement_policy = chip_placement_policy
    if dram_placement_algorithm is not None:
        g_compiler_config.dram_placement_algorithm = dram_placement_algorithm
    if enable_forked_dram_inputs is not None:
        g_compiler_config.enable_forked_dram_inputs = enable_forked_dram_inputs 
    if device_config is not None:
        if device_config in supported_backend_configurations and pkg_resources.resource_exists("pybuda", supported_backend_configurations[device_config]):
            g_compiler_config.backend_runtime_params_path = pkg_resources.resource_filename("pybuda", supported_backend_configurations[device_config])
            cached_syslevel_runtime_param = load_cached_sys_param(g_compiler_config.backend_runtime_params_path)
            g_compiler_config.harvesting_mask = int(cached_syslevel_runtime_param["system-device0-harvesting_mask"])
        else:
            raise RuntimeError(f"Unsupported backend device configuration: {device_config}. Valid options are: [{', '.join(supported_backend_configurations.keys())}]")

def set_epoch_break(op_names: Union[str, query.NodePredicateBuilder, List[Union[str, query.NodePredicateBuilder]]]):
    """
    Instruct place & route to start a new placement epoch on the given op(s)

    Parameters
    ----------
    op_names: Union[str, query.NodePredicateBuilder, List[Union[str, query.NodePredicateBuilder]]]
        Op or ops or predicate matches to start a new placement epoch
    """
    global g_compiler_config
    g_compiler_config.place_on_new_epoch(op_names)

def set_chip_break(op_names: Union[str, query.NodePredicateBuilder, List[Union[str, query.NodePredicateBuilder]]]):
    """
    Instruct place & route to start placing ops on the next chip in the pipeline.

    Parameters
    ----------
    op_names: Union[str, query.NodePredicateBuilder, List[Union[str, query.NodePredicateBuilder]]]
        Op or ops or predicate matches to start a new chip
    """
    global g_compiler_config
    g_compiler_config.place_on_new_chip(op_names)

def override_dram_queue_placement(dram_queue: str, *, chip_id: Optional[int] = None, channel: Optional[int] = None) -> None:
    """
    Override automatic dram placement with manual specification of chip and dram channel location

    Parameters
    ----------
    dram_queue: str
        Name of dram queue to override
    chip_id: int
        chip_id to place dram queue on
    dram_channel: int
        dram_channel to place dram queue on
    """
    global g_compiler_config
    g_compiler_config.place_queue_to_chip_dram(dram_queue, chip_id=chip_id, channel=channel)

def _balancer_op_override(op_name: str, attribute: str, value):
    global g_compiler_config
    g_compiler_config.balancer_op_override(op_name, attribute, value)

def override_dram_parameters(op_name: str, force_dram_parameters: bool):
    """
    Force parameters for specified op to reside in dram

    Parameters
    ----------
    op_name: str
        Name of the op to override

    force_dram_parameters: bool
        True: force parameters to reside in dram
        False: Allow parameters to be promoted to L1 (default)

    """
    _balancer_op_override(op_name, "force_dram_parameters", force_dram_parameters)

def override_op_size(op_name: str, grid_size: Tuple[int, int]):
    """
    Override automatic op sizing with given grid size.

    Parameters
    ----------
    op_name: str
        Name of the op to override

    grid_size: Tuple[int, int]
        Rectangular shape (row, column) of the placed op

    """
    _balancer_op_override(op_name, "grid_shape", grid_size)

def override_t_stream_dir(op_name: str, direction: str):
    """
    Override t stream direction

    Parameters
    ----------
    op_name: str
        Name of the op to override

    direction: str
        'n' or None: Don't stream
        'r': Stream in row major direction (lhs rows for matmul)
        'c': Stream in column major direction (rhs cols for matmul)
        'rz': Stream in z major direction, r major within each z slice
        'cz': Stream in z major direction, c major within each z slice

    """
    if direction is None:
        direction = 'n'
    _balancer_op_override(op_name, "t_stream_dir", direction.lower())

def override_t_stream_shape(op_name: str, shape: Tuple[int, int]):
    """
    Override t stream shape

    Parameters
    ----------
    op_name: str
        Name of the op to override

    shape: Tuple[int, int]
        stream shape to set

    """
    _balancer_op_override(op_name, "t_stream_shape", shape)

def override_fracture_factor(op_name: str, fracture_factor: int):
    """
    Override fracture factor

    Parameters
    ----------
    op_name: str
        Name of the op to override

    fracture_factor: int
        fracture_factor to set

    """
    _balancer_op_override(op_name, "fracture_factor", fracture_factor)

def override_u_kt(op_name: str, u_kt: int):
    """
    Override u_kt

    Parameters
    ----------
    op_name: str
        Name of the op to override

    u_kt: int
        u_kt value to set

    """
    _balancer_op_override(op_name, "u_kt", u_kt)

def override_input_buffer_multiplier(op_name: str, operand_index: int, *, multiplier: int):
    """
    Override input_buffer_multiplier factor

    Parameters
    ----------
    op_name: str
        Name of the op to override

    operand_index: int
        operand index value to set

    multiplier: int
        buffer multiplier value to set

    """
    _balancer_op_override(op_name, "input_buffer_multiplier", {operand_index: multiplier})

def internal_override_output_buffer_multiplier(op_name: str, *, multiplier: int):
    """
    Override u_kt

    Parameters
    ----------
    op_name: str
        Name of the op to override

    multiplier: int
        buffer multiplier value to set

    """
    logger.warning("internal_override_output_buffer_multiplier is an internal API and may result in hangs. Use at your own risk.")
    _balancer_op_override(op_name, "output_buffer_multiplier", multiplier)


def override_multi_op_fracture_factor(op_name: str, multi_op_fracture_factor: int):
    """
    Override fracture factor at op level (will fracture into multiple ops)

    Parameters
    ----------
    op_name: str
        Name of the op to override

    multi_op_fracture_factor: int
        multi_op_fracture_factor to set

    """

    global g_compiler_config
    g_compiler_config.conv_multi_op_fracture_factor_override[op_name] = multi_op_fracture_factor

def add_cpu_fallback_ops(op_types: Union[str, List[str]]):
    """
    Add one or more op types to CPU fallback list. These operation will be executed on the host. 
    """
    global g_compiler_config
    if isinstance(op_types, str):
        g_compiler_config.cpu_fallback_ops.add(op_types)
    else:
        g_compiler_config.cpu_fallback_ops.update(op_types)

def remove_cpu_fallback_ops(op_types: Union[str, List[str]]):
    """
    Remove one or more op types from the CPU fallback list.
    """
    global g_compiler_config
    if isinstance(op_types, str):
        g_compiler_config.cpu_fallback_ops.discard(op_types)
    else:
        for op_type in op_types:
            g_compiler_config.cpu_fallback_ops.discard(op_type)

def insert_fracture_group(nodes: List[Union[str, Tuple[str, Union[int, List[int]], Union[int, List[int]]]]], chip_ids: Union[List[int], Dict[str, List[int]]] = []):
    """
    Insert a fracture group, where a fracture group describes pybuda a subgraph
    to be fractured and along which dimension(s).

    Parameters
    ----------
    nodes: Union[str, List[Tuple[str, int, int]]]
        List of tuples (parameter_name, dim, factor) where dim is the dimension
        to fracture and factor is the amount fracture by.
        Or
        List of names, names to be made part of the same fracture group, but whose
        fracture factors should be inferred.

    chip_ids: Union[List[int], Dict[str, List[int]]]
        List of chip ids to fracture over.  The elements, if any, will be assigned
        in round robin order based on the corresponding fracture factors.
        Or a dictionary of op names to chip_ids, round robin assignment on a per
        op level.

    """
    global g_compiler_config
    for i in range(len(nodes)):
        if type(nodes[i]) is str:
            nodes[i] = (nodes[i], [], [])
        assert len(nodes[i]) == 3
        if type(nodes[i][1]) is int:
            assert type(nodes[i][2]) is int
            nodes[i] = list(nodes[i])
            nodes[i][1] = [nodes[i][1]]
            nodes[i][2] = [nodes[i][2]]
            nodes[i] = tuple(nodes[i])
        assert type(nodes[i][1]) is list
        assert type(nodes[i][2]) is list
        assert len(nodes[i][1]) == len(nodes[i][2])
        for j in range(len(nodes[i][1])):
            assert type(nodes[i][1][j]) is int
            assert type(nodes[i][2][j]) is int
    if type(chip_ids) is list:
        dict_chip_ids = {}
        for (name, dims, factors) in nodes:
            dict_chip_ids[name] = chip_ids
        chip_ids = dict_chip_ids
    g_compiler_config.fracture_groups.append((nodes, chip_ids))


def override_op_placement(
        op_name: Union[str, query.NodePredicateBuilder],
        *,
        start: Tuple[int, int] = None,
        transpose_op = False,
        chip_id: Optional[int] = None,
        spatial_epoch_break: bool = False,
        temporal_epoch_break: bool = False,
    ):
    """
    Override op_placement to provide to the placer.  Node that successive calls with the same node name or overlapping predicate match will throw an error.

    Parameters
    ----------
    op_name: str
        op name

    start: Tuple[int, int]
        Override the start grid-location for an op

    transpose_op: bool
        whether to manually transpose this op

    chip_id: int
        pin this op to a specific chip id. Only Wormhole_B0 is supported for now.

    spatial_epoch_break: bool
        Create a new spatial epoch and place `op_name` as the first op in the new epoch.

    temporal_epoch_break: bool
        Create a new spatial epoch and place `op_name` as the first op in the new epoch.
        This new spatial epoch will be placed on a new temporal epoch.

    """
    assert isinstance(op_name, str) or isinstance(op_name, query.NodePredicateBuilder), f"parameter `op_name` should be a string or NodePredicateBuilder. User provided `op_name`: {op_name}"
    assert isinstance(transpose_op, bool), f"parameter `tranpose_op` should be a bool. User provided `transpose_op`: {transpose_op}"
    if start is not None:
        assert isinstance(start, Iterable), f"parameter `start` should be an iterable. User provided `start`: {start}"
        assert len(start) == 2, f"parameter `start` should have two elements. User provided `start` with {len(start)} elements"

    global g_compiler_config
    g_compiler_config.placer_op_overrides.append((op_name, pyplacer.OpOverride(start, transpose_op, chip_id, temporal_epoch_break)))

    if temporal_epoch_break:
        set_epoch_break(op_name)
    if spatial_epoch_break:
        if g_compiler_config.use_interactive_placer == False or "PYBUDA_DISABLE_INTERACTIVE_PLACER" in os.environ:
            set_epoch_break(op_name)
        else:
            set_chip_break(op_name)


def __insert_nop_impl(
        src_op: str,
        dest_ops: Union[str, List[str]],
        *,
        hoist_tms: bool = True,
        nop_count: int = 1,
        daisy_chain: bool = False,
        is_fj_buffering = False,
    ):

    assert isinstance(src_op, str)
    if isinstance(dest_ops, str):
        dest_ops = [dest_ops]
    assert isinstance(hoist_tms, bool)

    global g_compiler_config
    merge_nops = bool(len(dest_ops) > 1)
    buff_ind = 0
    for dest_idx, dest_op in enumerate(dest_ops):
        buff_ind += 1
        request_merge = (dest_idx == len(dest_ops) -1)
        nop_instr = NopInsertionInstruction(
            src=src_op,
            dest=dest_op,
            hoist_tms=hoist_tms,
            nop_count=nop_count,
            input_id=None,
            fork_id=None,
            user_defined=True,
            mergeable=merge_nops,
            daisy_chain=daisy_chain,
            request_merge=request_merge,
            is_fj_buffering=is_fj_buffering
        )
        g_compiler_config.buffering_nops_to_insert[nop_instr.unique_id()] = nop_instr


def insert_nop(src_op: str, dest_ops: Union[str, List[str]], *, hoist_tms: bool = True, nop_count: int = 1, daisy_chain: bool = False):
    """
    Instruct pybuda compiler to insert a NOP instruction on the edge identified by the named src/dest pair.

    Parameters
    ----------
    src_op: str
        Name of the src op

    dest_op: str
        Name of the dest op

    hoist_tms: bool
        Configure whether the TMs on the original edge should be transfered to
        (src -> NOP edge) or to the (NOP -> dest edge).

    daisy_chain: bool
        Sets the merge-strategy for NOPs to `daisy_chain` when there are multiple dest-ops.
        By default, the merge-strategy will create a single buffer-nop forking to `dest_ops`.
        When `daisy_chain` is enabled, we will create a daisy-chain of nop operations to dest_ops.

    """

    __insert_nop_impl(
        src_op=src_op,
        dest_ops=dest_ops,
        hoist_tms=hoist_tms,
        nop_count=nop_count,
        daisy_chain=daisy_chain,
        is_fj_buffering=False,
    )


def _internal_insert_fj_buffering_nop(src_op: str, dest_ops: Union[str, List[str]], *, hoist_tms: bool = True, nop_count: int = 1, daisy_chain: bool = False):
    """
    Instruct pybuda compiler to insert a fork-join buffering NOP instruction on the edge identified by the named src/dest pair.
    Note: Adding a fork-join buffering NOP instructions may lead to exceptions!

    Parameters
    ----------
    src_op: str
        Name of the src op

    dest_op: str
        Name of the dest op

    hoist_tms: bool
        Configure whether the TMs on the original edge should be transfered to
        (src -> NOP edge) or to the (NOP -> dest edge).

    daisy_chain: bool
        Sets the merge-strategy for NOPs to `daisy_chain` when there are multiple dest-ops.
        By default, the merge-strategy will create a single buffer-nop forking to `dest_ops`.
        When `daisy_chain` is enabled, we will create a daisy-chain of nop operations to dest_ops.

    """
    __insert_nop_impl(
        src_op=src_op,
        dest_ops=dest_ops,
        hoist_tms=hoist_tms,
        nop_count=nop_count,
        daisy_chain=daisy_chain,
        is_fj_buffering=True,
    )


def insert_buffering_nop(src_op: str, dest_ops: Union[str, List[str]], *, hoist_tms: bool = True, nop_count: int = 1, daisy_chain: bool = False):
    """
    "DEPRECATION WARNING! Please use `insert_nop` instead of `insert_buffering_nop`. To add a buffering nop, use the \
    internal API `_internal_insert_fj_buffering_nop`."

    Instruct pybuda compiler to insert a buffering NOP instruction on the edge identified by the named src/dest pair.
    Note: Adding buffering NOP instructions may lead to exceptions!

    Parameters
    ----------
    src_op: str
        Name of the src op

    dest_op: str
        Name of the dest op

    hoist_tms: bool
        Configure whether the TMs on the original edge should be transfered to
        (src -> NOP edge) or to the (NOP -> dest edge).

    daisy_chain: bool
        Sets the merge-strategy for NOPs to `daisy_chain` when there are multiple dest-ops.
        By default, the merge-strategy will create a single buffer-nop forking to `dest_ops`.
        When `daisy_chain` is enabled, we will create a daisy-chain of nop operations to dest_ops.

    """

    logger.warning("DEPRECATION WARNING! Please use `insert_nop` instead of `insert_buffering_nop`. To add fork-join \
                   buffering nop, use the internal API `_internal_insert_fj_buffering_nop`.")

    _internal_insert_fj_buffering_nop(
        src_op=src_op,
        dest_ops=dest_ops,
        hoist_tms=hoist_tms,
        nop_count=nop_count,
        daisy_chain=daisy_chain
    )


def add_schedule_constraint(partial_ordering: List[str]):
    """
    Instruct pybuda compiler to schedule ops in a way that respects the given partial ordering.
    The compiler will ensure to schedule op_order[i] before op_order[i+1] in the final schedule.

    Parameters
    ----------
    partial_ordering: List[str]
        List of op names in the order they should be scheduled
    """

    global g_compiler_config
    g_compiler_config.scheduler_constraints.append(partial_ordering)


def set_num_repeated_patterns(module_name: str, num_patterns: int):
    """
    Override default MathFidelity used by an op

    Parameters
    ----------
    module_name: str
        Name of the module 

    num_patterns: int
        Number of repeated patterns in the module to scan through

    """
    global g_compiler_config
    g_compiler_config.tvm_module_to_num_patterns[module_name] = num_patterns


def set_auto_transposing_placement(is_enabled: bool):
    """
    Override default transpose placement setting

    Parameters
    ----------
    is_enabled: bool
        Set to true to enable auto transpose placement setting

    """
    global g_compiler_config
    g_compiler_config.enable_auto_transposing_placement = is_enabled

def configure_mixed_precision(
    *,
	op_type: Optional[str] = None,
	epoch_type: Optional[str] = None,
	output_df: Optional[DataFormat] = None,
	intermediate_df: Optional[DataFormat] = None,
	accumulate_df: Optional[DataFormat] = None,
	math_fidelity: Optional[MathFidelity] = None,
	name_regex: Optional[str] = None,
	input_df: Optional[Union[Dict[int, Tuple[DataFormat, bool]], DataFormat]] = None,
	is_gradient_op: Optional[bool] = None,
    input_parameter_indices_to_optimize: Optional[List[int]] = None):
    """

    Configure mixed precision settings

    Parameters
    ----------
    op_type: Optional[str]
        String defining the op-type

    epoch_type: Optional[str]
        epoch type: {fwd, bwd, opt}

    output_df: Optional[DataFormat]
        Data format from packer to L1

    intermediate_df: Optional[DataFormat]
        define data-format used for intermediate spills to L1

    accumulate_df: Optional[DataFormat]
        data format used for accumulation

    math_fidelity: Optional[MathFidelity]
        configure number of fidelity phases

    name_regex: Optional[str]
        The regular expression pattern to be used to match against the name of an
        operation in the graph or the name of the input/parameter/constant. The
        pattern should conform to the ECMAScript regular expression grammar,
        because it will be used in a C++ std::regex context.

    input_df: Optional[Dict[int, Tuple[DataFormat, bool]]]
        map containing keys of operand_index to Tuple[DataFormat, target_activations]
        where `target_activations` is a bool determining whether to include activations
        (i.e. non-inputs) as part of the mixed-precision query/configuration.

        e.g. input_df = {
            0: [DataFormat.Float16, True]
        }

    is_gradient_op: Optional[bool]
        define whether the op is a gradient op

    input_parameter_indices_to_optimize: Optional[List[int]]
        use lower precision data-formats for input parameter indices

    """
    assert op_type is None or isinstance(op_type, str)
    assert output_df is None or isinstance(output_df, DataFormat)
    assert intermediate_df is None or isinstance(intermediate_df, DataFormat)
    assert accumulate_df is None or isinstance(accumulate_df, DataFormat)
    assert math_fidelity is None or isinstance(math_fidelity, MathFidelity)
    assert name_regex is None or isinstance(name_regex, str)
    assert op_type or epoch_type or name_regex or is_gradient_op
    if input_df is None:
        input_df = {}
    if isinstance(input_df, dict):
        for operand_index, config in input_df.items():
            assert len(config) == 2, f"For operand index {operand_index}, invalid config"

    global g_compiler_config
    g_compiler_config.amp_properties.append(
        AMPNodeProperties(
            op_type,
            epoch_type,
            output_df,
            intermediate_df,
            accumulate_df,
            math_fidelity,
            name_regex,
            input_df,
            is_gradient_op,
            input_parameter_indices_to_optimize
        )
    )


def _get_global_compiler_config() -> CompilerConfig:
    return g_compiler_config

def _clear_global_compiler_config():
    global g_compiler_config
    g_compiler_config = CompilerConfig()

def _set_global_compiler_config(config: CompilerConfig):
    global g_compiler_config
    g_compiler_config = config


def _set_pybuda_override_veto(general_config_dict, environ_config_dict):
    import json
    
    env_dict = {key: value for key, value in os.environ.items() if key.startswith("PYBUDA_") and key != "PYBUDA_OVERRIDES_VETO"}
    env_dict = {**env_dict, **environ_config_dict}

    os.environ["PYBUDA_OVERRIDES_VETO"] = json.dumps({
        "general_conf": general_config_dict,
        "environ_conf": env_dict,
    })
