# GENERAL OP SUPPORT TEST PLAN:
# 1. Operand type - any supported type
# 2. Operand source(s):
# (+)  2.1 From another op
#       - Operator -> input
# (+)  2.2 From tm edge
#       - Combination: operator -> tm -> input
#       - tm -> input
# (+)  2.3 From DRAM queue
#       - input_queue flag = false
#       - Special case of From host? May it be triggered if the operator is not the first node of the network?
#       - Can this be triggered from pybuda.Parameter?
#       - Can this be triggered from big pybuda.Constant?
# (+)  2.4 From DRAM, but prologued (constant)
#       - Constants must be small enough to fit into L1
#       - Verification via netlists that scenario is triggered
#       - Input are not prologued for microbatch size = 1
# (+)  2.5 Const Inputs (const eval pass)
#       - Operator where all inputs are constants. Does it make difference if tensor is big > L1
#       - Verification via netlists that scenario is triggered???
# (+)  2.6 From host
#       - Input tensor as input of network -> Operator is first node in network and input_queue flag = true
#       - Can this scenario be triggered from pybuda.Parameter?
#       - Can this be triggered from big pybuda.Constant?
# 3 Operand shapes type(s):
# (+)  3.1 Full tensor (i.e. full expected shape)
#       - Is 3 dims max for all ops? Ex. Conv is 3d max
# (+)  3.2 Tensor reduce on one or more dims to 1
#       - Vector
#       - Only one dim is not equal to 1
# (/)  3.3 Scalar
#       - Create tensor of dimension equal to 0 (tensor from scalar) or just to use scalar as simple value
# 4. Operand / output size of dimensions (few examples of each, 10 values total)
# (+)  4.1 Divisible by 32
# (+)  4.2 Prime numbers
# (+)  4.3 Very large (thousands, 10s of thousands)
#       - 100x100, 100x1000
#       - maybe nightly only
# (+)  4.4 Extreme ratios between height/width
#      4.5 ...probably many more interesting combinations here
# 5. Data format - all supported formats
# (/)  5.1 Output DF
# (/)  5.2 Intermediate DF
# (/)  5.3 Accumulation DF
# (+)  5.4 Operand DFs
# (+) 6. Math fidelity - LoFi, HiFi2a, Hifi2b, Hifi3, Hifi4
# (/) 7. Special attributes - if applicable.. like approx_mode for Exp, for example

import pytest

from pybuda.config import _get_global_compiler_config
from pybuda import Tensor
import torch

from pybuda import pybuda

from pybuda.verify.config import TestKind, VerifyConfig

from test.common import run

from pybuda.module import PyBudaModule

from pybuda.verify.backend import verify_module

from test.operators.utils import NetlistValidation
from test.operators.utils import FailingReasons



def get_input_shapes(micro_batch_size=1):
                                              # Here we cover interesting combinations of input shapes:
    return [
            ((micro_batch_size, 64, 3, 4),         (4, 3)),         #1        # 3.1 Full tensor (i.e. full expected shape)
            ((micro_batch_size, 64, 45, 17),       (17, 45)),       #2        # 3.1 Full tensor (i.e. full expected shape)
            ((micro_batch_size, 64, 1, 23),        (23, 1)),        #3        # 3.2 Tensor reduce on one or more dims to 1
            ((micro_batch_size, 64, 64, 1),        (1, 64)),        #4        # 3.2 Tensor reduce on one or more dims to 1
            ((micro_batch_size, 64, 100, 100),     (100, 100)),     #5        # 4.3 Very large (thousands, 10s of thousands)
            ((micro_batch_size, 64, 1000, 100),    (100, 1000)),    #6        # 4.3 Very large (thousands, 10s of thousands)
            ((micro_batch_size, 64, 10, 1000),     (1000, 10)),     #7        # 4.4 Extreme ratios between height/width          
            ((micro_batch_size, 64, 9920, 1),      (1, 9920)),      #8        # 4.4 Extreme ratios between height/width
            ((micro_batch_size, 64, 10000, 1),     (1, 10000)),     #9        # 4.4 Extreme ratios between height/width
            ((micro_batch_size, 64, 32, 64),       (64, 32)),       #10       # 4.1 Divisible by 32
            ((micro_batch_size, 64, 160, 96),      (96, 160)),      #11       # 4.1 Divisible by 32
            ((micro_batch_size, 64, 17, 41),       (41, 17)),       #12       # 4.2 Prime numbers
            ((micro_batch_size, 64, 89, 3),        (3, 89)),        #13       # 4.2 Prime numbers
            ]


def get_sparse_tensor(shape, const_input = True):
    row_cnt = shape[0]
    col_cnt = shape[1]
    rows = torch.arange(row_cnt).tolist()
    cols = torch.arange(col_cnt).tolist()
    min = 0
    if row_cnt < col_cnt:
         min = rows
    else:
        min = cols
    sparse = torch.sparse_coo_tensor([min, min], torch.ones(len(min)), shape, dtype=torch.float32)
    sparse = torch.stack([sparse]*64, -3) 
    sparse = torch.unsqueeze(sparse, 0) 
    sparse = pybuda.Tensor.create_from_torch(sparse, constant=const_input)
    return sparse

@pytest.mark.parametrize("input_shape_dense, input_shape_sparse", get_input_shapes())
def test_smm_operand_src_from_host(
    input_shape_dense,
    input_shape_sparse,
    test_device,
    input_params=[], 
    math_fidelity=None
):
    class Model(PyBudaModule):
            def __init__(self, name, sparse_shape):
                super().__init__(name)
                self.add_constant("sparse")
                self.set_constant("sparse", get_sparse_tensor(sparse_shape))
    
            def forward(self, dense):
                result = pybuda.op.SparseMatmul("smm1", self.get_constant("sparse"), dense)
                return result
            
    mod = Model("test_sparse_matmul_operand_src_from_host", input_shape_sparse)
    
    input_shapes = tuple([input_shape_dense])
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.input_queues_on_host = True
    if (math_fidelity is not None):
        compiler_cfg.default_math_fidelity = math_fidelity
    verify_module(
        mod,
        input_shapes=input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
        input_params=[input_params],
    )

@pytest.mark.parametrize("input_shape_dense, input_shape_sparse", get_input_shapes())
def test_smm_operand_src_from_dram(
    input_shape_dense,
    input_shape_sparse,
    test_device,
    input_params=[], 
    math_fidelity=None
):
    class Model(PyBudaModule):
            def __init__(self, name, sparse_shape):
                super().__init__(name)
                self.add_constant("sparse")
                self.set_constant("sparse", get_sparse_tensor(sparse_shape))

            def forward(self, dense):
                result = pybuda.op.SparseMatmul("smm1", self.get_constant("sparse"), dense)
                return result
            
    mod = Model("test_sparse_matmul_operand_src_from_dram", input_shape_sparse)

    input_shapes = tuple([input_shape_dense])
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.input_queues_on_host = False
    if (math_fidelity is not None):
        compiler_cfg.default_math_fidelity = math_fidelity
    verify_module(
        mod,
        input_shapes=input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
        input_params=[input_params],
    )
    netlist = NetlistValidation()
    assert netlist.get_value("/queues/dense/loc") == 'dram'

@pytest.mark.parametrize("input_shape_dense, input_shape_sparse", get_input_shapes())
def test_smm_operand_src_from_const_inputs_const_eval(
    input_shape_dense,
    input_shape_sparse,
    test_device,
    input_params=[], 
    math_fidelity=None
):
    class Model(PyBudaModule):
            def __init__(self, name, sparse_shape, dense_shape):
                super().__init__(name)
                self.add_constant("sparse")
                self.set_constant("sparse", get_sparse_tensor(sparse_shape))

                self.add_constant("dense")
                self.set_constant("dense", pybuda.Tensor.create_from_torch(torch.rand(*dense_shape, requires_grad=False), constant=True))

            def forward(self, x1, x2):
                smm1 = pybuda.op.SparseMatmul("smm1", self.get_constant("sparse"), self.get_constant("dense"))
                mm1 = pybuda.op.Matmul("mm1", x2, x1)   
                add1 = pybuda.op.Add("add1", smm1, mm1)
                return add1
            
    mod = Model("test_sparse_matmul_operand_src_from_const_inputs_const_eval", input_shape_sparse, input_shape_dense)

    input_shape_dense_tr = (input_shape_dense[0],input_shape_dense[1],input_shape_dense[3],input_shape_dense[2])
    input_shapes = tuple([input_shape_dense, input_shape_dense_tr])
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.input_queues_on_host = True
    if (math_fidelity is not None):
        compiler_cfg.default_math_fidelity = math_fidelity
    verify_module(
        mod,
        input_shapes=input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
        input_params=[input_params],
    )
    netlist = NetlistValidation()
    d = netlist.get_value("/graphs/fwd_0_0_temporal_epoch_0")
    for key in d.keys():
        assert "Matmul" not in key

@pytest.mark.parametrize("input_shape_dense, input_shape_sparse", get_input_shapes())
def test_smm_operand_src_from_another_op(
    input_shape_dense,
    input_shape_sparse,
    test_device,
    input_params=[], 
    math_fidelity=None
):
    class Model(PyBudaModule):
            def __init__(self, name, sparse_shape):
                super().__init__(name)
                self.add_constant("sparse")
                self.set_constant("sparse", get_sparse_tensor(sparse_shape))
                
            def forward(self, x):
                add1 = pybuda.op.Add("add1", x, x)
                result = pybuda.op.SparseMatmul("smm1", self.get_constant("sparse"), add1)
                return result
            
    mod = Model("test_sparse_matmul_operand_src_from_another_op", input_shape_sparse)

    input_shapes = tuple([input_shape_dense])
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.input_queues_on_host = True
    if (math_fidelity is not None):
        compiler_cfg.default_math_fidelity = math_fidelity
    verify_module(
        mod,
        input_shapes=input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
        input_params=[input_params],
    )

@pytest.mark.parametrize("input_shape_dense, input_shape_sparse", get_input_shapes())
def test_smm_operand_src_from_tm_edge1(
    input_shape_dense,
    input_shape_sparse,
    test_device,
    input_params=[], 
    math_fidelity=None
):
    class Model(PyBudaModule):
            def __init__(self, name, sparse_shape):
                super().__init__(name)
                self.add_constant("sparse")
                self.set_constant("sparse", get_sparse_tensor(sparse_shape))
                
            def forward(self, x):
                tr1 = pybuda.op.Transpose("tr1", x, -1, -2)
                tr2 = pybuda.op.Transpose("tr2", tr1, -1, -2)
                result = pybuda.op.SparseMatmul("smm1", self.get_constant("sparse"), tr2)
                return result
            
    mod = Model("test_sparse_matmul_operand_src_from_tm_edge1", input_shape_sparse)

    input_shapes = tuple([input_shape_dense])
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.input_queues_on_host = True
    if (math_fidelity is not None):
        compiler_cfg.default_math_fidelity = math_fidelity
    verify_module(
        mod,
        input_shapes=input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
        input_params=[input_params],
    )

@pytest.mark.parametrize("input_shape_dense, input_shape_sparse", get_input_shapes())
def test_smm_operand_src_from_tm_edge2(
    input_shape_dense,
    input_shape_sparse,
    test_device,
    input_params=[], 
    math_fidelity=None
):
    class Model(PyBudaModule):
            def __init__(self, name, sparse_shape):
                super().__init__(name)
                self.add_constant("sparse")
                self.set_constant("sparse", get_sparse_tensor(sparse_shape))
                
            def forward(self, x):
                add1 = pybuda.op.Add("add1", x, x)
                tr1 = pybuda.op.Transpose("tr1", add1, -1, -2)
                tr2 = pybuda.op.Transpose("tr2", tr1, -1, -2)
                result = pybuda.op.SparseMatmul("smm1",  self.get_constant("sparse"), tr2)
                return result
            
    mod = Model("test_sparse_matmul_operand_src_from_tm_edge2", input_shape_sparse)

    input_shapes = tuple([input_shape_dense])
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.input_queues_on_host = True
    if (math_fidelity is not None):
        compiler_cfg.default_math_fidelity = math_fidelity
    verify_module(
        mod,
        input_shapes=input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
        input_params=[input_params],
    )

@pytest.mark.parametrize("input_shape_dense, input_shape_sparse", [
                    pytest.param((1, 64, 3, 4),         (4, 3)),                                                                #1    # 3.1 Full tensor (i.e. full expected shape)),
                    pytest.param((1, 64, 1, 23),        (23, 1)),                                                               #3        # 3.2 Tensor reduce on one or more dims to 1
                    pytest.param((1, 64, 100, 100),     (100, 100)),                                                            #5        # 4.3 Very large (thousands, 10s of thousands)

                    # Error message: E           AssertionError: Error during inference
                    pytest.param((1, 64, 45, 17),       (17, 45),    marks=pytest.mark.xfail(reason=FailingReasons.INFERENCE_FAILED)),        #2        # 3.1 Full tensor (i.e. full expected shape)    
                    pytest.param((1, 64, 64, 1),        (1, 64),     marks=pytest.mark.xfail(reason=FailingReasons.INFERENCE_FAILED)),        #4        # 3.2 Tensor reduce on one or more dims to 1    
                    pytest.param((1, 64, 1000, 100),    (100, 1000), marks=pytest.mark.xfail(reason=FailingReasons.INFERENCE_FAILED)),        #6        # 4.3 Very large (thousands, 10s of thousands)  
                    pytest.param((1, 64, 160, 96),      (96, 160),   marks=pytest.mark.xfail(reason=FailingReasons.INFERENCE_FAILED)),        #11       # 4.1 Divisible by 32                           
                    pytest.param((1, 64, 89, 3),        (3, 89),     marks=pytest.mark.xfail(reason=FailingReasons.INFERENCE_FAILED)),        #13       # 4.2 Prime numbers                             
            
                    # "Error message: E           AssertionError: Data mismatch detected"
                    pytest.param((1, 64, 10, 1000),     (1000, 10),  marks=pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)),           #7        # 4.4 Extreme ratios between height/width       
                    pytest.param((1, 64, 32, 64),       (64, 32),    marks=pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)),           #10       # 4.1 Divisible by 32                           
                    pytest.param((1, 64, 17, 41),       (41, 17),    marks=pytest.mark.xfail(reason=FailingReasons.DATA_MISMATCH)),           #12       # 4.2 Prime numbers                             
            
                    # "Fatal python error - xfail does not work; UserWarning: resource_tracker: There appear to be 26 leaked semaphore objects to clean up at shutdown"
                    pytest.param((1, 64, 9920, 1),      (1, 9920),   marks=pytest.mark.skip(reason=FailingReasons.SEMAPHORE_LEAK)),           #8        # 4.4 Extreme ratios between height/width     
                    pytest.param((1, 64, 10000, 1),     (1, 10000),  marks=pytest.mark.skip(reason=FailingReasons.SEMAPHORE_LEAK)),           #9        # 4.4 Extreme ratios between height/width     
        ])
def test_smm_operand_src_from_tm_edge3(
    input_shape_dense,
    input_shape_sparse,
    test_device,
    input_params=[], 
    math_fidelity=None
):
    class Model(PyBudaModule):
            def __init__(self, name, sparse_shape):
                super().__init__(name)
                self.add_constant("sparse")
                self.set_constant("sparse", get_sparse_tensor(sparse_shape))
                
            def forward(self, x):
                tr1 = pybuda.op.Transpose("tr1", self.get_constant("sparse"), -1, -2)
                tr2 = pybuda.op.Transpose("tr2", x, -1, -2)
                result = pybuda.op.SparseMatmul("smm1", tr1, tr2)
                return result
            
    mod = Model("test_sparse_matmul_operand_src_from_tm_edge3", input_shape_sparse)

    input_shapes = tuple([input_shape_dense])
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.input_queues_on_host = True
    if (math_fidelity is not None):
        compiler_cfg.default_math_fidelity = math_fidelity
    verify_module(
        mod,
        input_shapes=input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
        input_params=[input_params],
    )


def get_input_shapes_prologued():
                                              # Here we cover interesting combinations of input shapes:
    return [
            ((2, 64, 3, 4),      (4, 3),        True,   False),  #18       # 3.1 Full tensor (i.e. full expected shape)
            ((2, 64, 3, 4),      (4, 3),        False,  True),  #19       # 3.1 Full tensor (i.e. full expected shape)
            ((2, 64, 3, 4),      (4, 3),        None,   True) ,  #20       # 3.1 Full tensor (i.e. full expected shape)
            ((1, 64, 3, 4),      (4, 3),        True,   False),  #21       # 3.1 Full tensor (i.e. full expected shape)
            ((1, 64, 3, 4),      (4, 3),        False,  True),  #22       # 3.1 Full tensor (i.e. full expected shape)
            ((1, 64, 3, 4),      (4, 3),        None,   True),   #23       # 3.1 Full tensor (i.e. full expected shape) ! not working as described in docs
            ((2, 64, 45, 17),    (17, 45),      None,   True) ,  #24       # 3.1 Full tensor (i.e. full expected shape)
            ((2, 64, 1, 23),     (23, 1),       None,   True) ,  #25       # 3.2 Tensor reduce on one or more dims to 1
            ((2, 64, 64, 1),     (1, 64),       None,   True) ,  #26       # 3.2 Tensor reduce on one or more dims to 1
            ((2, 64, 100, 100),  (100, 100),    None,   True) ,  #27       # 4.3 Very large (thousands, 10s of thousands)
            # "Fatal python error - xfail does not work. Error message: Fatal Python error: Segmentation fault; UserWarning: resource_tracker: There appear to be 26 leaked semaphore objects to clean up at shutdown"
            pytest.param((2, 64, 1000, 100), (100, 1000),   None,   True, marks=pytest.mark.skip(reason=FailingReasons.SEMAPHORE_LEAK)),  # 4.3 Very large (thousands, 10s of thousands)         
            ((2, 64, 10, 1000),  (1000, 10),    None,   True) ,  #29       # 4.4 Extreme ratios between height/width        
            ((2, 64, 9920, 1),   (1, 9920),     None,   True) ,  #30       # 4.4 Extreme ratios between height/width 
            ((2, 64, 10000, 1),  (1, 10000),    None,   True) ,  #31       # 4.4 Extreme ratios between height/width   
            ((2, 64, 32, 64),    (64, 32),      None,   True) ,  #32       # 4.1 Divisible by 32
            ((2, 64, 160, 96),   (96, 160),     None,   True) ,  #33       # 4.1 Divisible by 32
            ((2, 64, 17, 41),    (41, 17),      None,   True) ,  #34       # 4.2 Prime numbers
            ((2, 64, 89, 3),     (3, 89),       None,   True) ,  #35       # 4.2 Prime numbers
            ]
@pytest.mark.parametrize("input_shape_dense, input_shape_sparse, default_dram_params, prologue", get_input_shapes_prologued())
def test_smm_operand_src_from_const_inputs_prologue(
    input_shape_dense,
    input_shape_sparse,
    default_dram_params,
    prologue,
    test_device,
    input_params=[], 
    math_fidelity=None
):
    class Model(PyBudaModule):
            def __init__(self, name, sparse_shape):
                super().__init__(name)
                self.add_constant("sparse")
                self.set_constant("sparse", get_sparse_tensor(sparse_shape))

            def forward(self, x):
                smm1 = pybuda.op.SparseMatmul("smm1", self.get_constant("sparse"), x)
                return smm1
            
    mod = Model("test_sparse_matmul_operand_src_from_const_inputs_prologue", input_shape_sparse)

    input_shapes = tuple([input_shape_dense])
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.input_queues_on_host = False
    compiler_cfg.default_dram_parameters = default_dram_params
    if (math_fidelity is not None):
        compiler_cfg.default_math_fidelity = math_fidelity
    verify_module(
        mod,
        input_shapes=input_shapes,
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
        ),
        input_params=[input_params],
    )
    netlist = NetlistValidation()
    d = netlist.get_value("/programs/0/run_fwd_0/4/execute/queue_settings/lc.input_tensor.smm1.0")
    if prologue:
        assert d['prologue']
    else:
        assert not d['prologue']
    

# We will not test all combinations of Data Format and Math Fidelity because it would be too much tests. 
#   1. First we will choose Data Format to be Float16_b and test all Math Fidelity values
#   2. Then we will set Math Fidelity to HiFi4 and test all Data Formats. 


## 1.

def get_input_shape_sparse(micro_batch_size=1):
    return (4, 3)

def get_input_shape_dense(micro_batch_size=1):
    return (micro_batch_size, 64, 3, 4)

verify_input_params=[ 
                        {"dev_data_format": pybuda.DataFormat.Float16_b},
                    ]
compiler_math_fidelity = [
                            pybuda.MathFidelity.LoFi,
                            pybuda.MathFidelity.HiFi2,
                            pybuda.MathFidelity.HiFi3,
                            pybuda.MathFidelity.HiFi4,
                         ]

@pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
def test_smm_mf_inputs_from_host(test_device, math_fidelity):
    test_smm_operand_src_from_host(get_input_shape_dense(), get_input_shape_sparse(), test_device, verify_input_params, math_fidelity)

# @pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
# def test_smm_mf_inputs_from_dram(test_device, math_fidelity):
#     test_smm_operand_src_from_dram(get_input_shape_dense(), get_input_shape_sparse(), test_device, verify_input_params, math_fidelity)

# @pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
# def test_smm_mf_inputs_from_const_inputs_const_eval(test_device, math_fidelity):
#     test_smm_operand_src_from_const_inputs_const_eval(get_input_shape_dense(), get_input_shape_sparse(), test_device, verify_input_params, math_fidelity)

# @pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
# def test_smm_mf_inputs_from_another_op(test_device, math_fidelity):
#     test_smm_operand_src_from_another_op(get_input_shape_dense(), get_input_shape_sparse(), test_device, verify_input_params, math_fidelity)

# @pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
# def test_smm_mf_inputs_from_tm_edge1(test_device, math_fidelity):
#     test_smm_operand_src_from_tm_edge1(get_input_shape_dense(), get_input_shape_sparse(), test_device, verify_input_params, math_fidelity)

# @pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
# def test_smm_mf_inputs_from_tm_edge2(test_device, math_fidelity):
#     test_smm_operand_src_from_tm_edge2(get_input_shape_dense(), get_input_shape_sparse(), test_device, verify_input_params, math_fidelity)


## 2.

verify_input_params=[
                        {"dev_data_format": pybuda.DataFormat.Bfp2},
                        {"dev_data_format": pybuda.DataFormat.Bfp2_b},
                        {"dev_data_format": pybuda.DataFormat.Bfp4},
                        {"dev_data_format": pybuda.DataFormat.Bfp4_b},
                        {"dev_data_format": pybuda.DataFormat.Bfp8},
                        {"dev_data_format": pybuda.DataFormat.Bfp8_b},
                        {"dev_data_format": pybuda.DataFormat.Float16},  
                        {"dev_data_format": pybuda.DataFormat.Float16_b},
                        {"dev_data_format": pybuda.DataFormat.Float32},
                        {"dev_data_format": pybuda.DataFormat.Int8},
                        {"dev_data_format": pybuda.DataFormat.Lf8},
                        {"dev_data_format": pybuda.DataFormat.RawUInt16},
                        {"dev_data_format": pybuda.DataFormat.RawUInt32},
                        {"dev_data_format": pybuda.DataFormat.RawUInt8},
                        {"dev_data_format": pybuda.DataFormat.UInt16},
                    ]

compiler_math_fidelity = pybuda.MathFidelity.HiFi4

@pytest.mark.parametrize("input_params", verify_input_params)
def test_smm_df_inputs_from_host(test_device, input_params):
    test_smm_operand_src_from_host(get_input_shape_dense(), get_input_shape_sparse(), test_device, input_params, compiler_math_fidelity)

# @pytest.mark.parametrize("input_params", verify_input_params)
# def test_smm_df_inputs_from_dram(test_device, input_params):
#     test_smm_operand_src_from_dram(get_input_shape_dense(), get_input_shape_sparse(), test_device, input_params, compiler_math_fidelity)

# @pytest.mark.parametrize("input_params", verify_input_params)
# def test_smm_df_inputs_from_const_inputs_const_eval(test_device, input_params):
#     test_smm_operand_src_from_const_inputs_const_eval(get_input_shape_dense(), get_input_shape_sparse(), test_device, input_params, compiler_math_fidelity)

# @pytest.mark.parametrize("input_params", verify_input_params)
# def test_smm_df_inputs_from_another_op(test_device, input_params):
#     test_smm_operand_src_from_another_op(get_input_shape_dense(), get_input_shape_sparse(), test_device, input_params, compiler_math_fidelity)

# @pytest.mark.parametrize("input_params", verify_input_params)
# def test_smm_df_inputs_from_tm_edge1(test_device, input_params):
#     test_smm_operand_src_from_tm_edge1(get_input_shape_dense(), get_input_shape_sparse(), test_device, input_params, compiler_math_fidelity)

# @pytest.mark.parametrize("input_params", verify_input_params)
# def test_smm_df_inputs_from_tm_edge2(test_device, input_params):
#     test_smm_operand_src_from_tm_edge2(get_input_shape_dense(), get_input_shape_sparse(), test_device, input_params, compiler_math_fidelity)







# Tests from sanity moved here:

@pytest.mark.parametrize("config", ["3x3conv", "data_mismatch", "c_stream", "in_out_stream"])
def test_sparse_matmul(test_device, config):
    from pybuda.op.eval.sparse_utils import create_conv2d_sparse_picker_matrix

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "MaximizeTMinimizeGrid"

    if config == "3x3conv":
        iH, iW = (64, 64)
        inC = 32
        kH, kW = (3, 3)
        stride = (2, 2)
        padding = (kW // 2, kW // 2, kH // 2, kH // 2)
        dilation = 1

        t = torch.arange(iH*iW*inC, dtype=torch.float32).reshape((1, 1, iH * iW, inC))
        act = Tensor.create_from_torch(t)

        pickers = []
        for kY in range(kH):
            for kX in range(kW):
                y_shift = ((kH - 1) // 2) - kY
                x_shift = ((kW - 1) // 2) - kX
                picker = create_conv2d_sparse_picker_matrix(iH, iW, y_shift, x_shift, kH, kW, stride, padding, dilation, tile_align=True)
                pickers.append(picker)
        sparse = Tensor.create_from_torch(torch.stack(pickers).unsqueeze(0), constant=True)
    elif config == "data_mismatch":
        minimal_tiles = 2
        act = torch.randn(32*minimal_tiles,32).unsqueeze(0).unsqueeze(0)
        out_tiles = minimal_tiles // 2
        eye = torch.eye(32*minimal_tiles, 32*minimal_tiles)
        pickers = [
            eye[:(out_tiles*32), :].to_sparse(),
            eye[(out_tiles*32-16):-16, :].to_sparse(),
        ]
        sparse = Tensor.create_from_torch(torch.stack(pickers).unsqueeze(0), constant=True)
    elif config == "c_stream":

        pytest.skip() # tenstorrent/budabackend#1543
        pybuda.config.override_t_stream_dir("sparse0.lc2", "C")
        pybuda.config.override_t_stream_shape("sparse0.lc2", (1, 32))
        iH, iW = (64, 64)
        inC = 1024
        kH, kW = (1, 1)
        stride = (2, 2)
        padding = (kW // 2, kW // 2, kH // 2, kH // 2)
        dilation = 1

        t = torch.arange(iH*iW*inC, dtype=torch.float32).reshape((1, 1, iH * iW, inC))
        act = Tensor.create_from_torch(t)

        pickers = []
        for kY in range(kH):
            for kX in range(kW):
                y_shift = ((kH - 1) // 2) - kY
                x_shift = ((kW - 1) // 2) - kX
                picker = create_conv2d_sparse_picker_matrix(iH, iW, y_shift, x_shift, kH, kW, stride, padding, dilation, tile_align=True)
                pickers.append(picker)
        sparse = Tensor.create_from_torch(torch.stack(pickers).unsqueeze(0), constant=True)
    elif config == "in_out_stream":
        pybuda.config.override_t_stream_dir("buf0", "R")
        pybuda.config.override_t_stream_shape("buf0", (2, 1))
        pybuda.config.override_t_stream_dir("sparse0.lc2", "R")
        pybuda.config.override_t_stream_shape("sparse0.lc2", (3, 1))

        iH, iW = (32, 32)
        inC = 32
        kH, kW = (3, 3)
        stride = (2, 2)
        padding = (kW // 2, kW // 2, kH // 2, kH // 2)
        dilation = 1

        t = torch.arange(iH*iW*inC, dtype=torch.float32).reshape((1, 1, iH * iW, inC))
        act = Tensor.create_from_torch(t)

        pickers = []
        for kY in range(kH):
            for kX in range(kW):
                y_shift = ((kH - 1) // 2) - kY
                x_shift = ((kW - 1) // 2) - kX
                picker = create_conv2d_sparse_picker_matrix(iH, iW, y_shift, x_shift, kH, kW, stride, padding, dilation, tile_align=True)
                pickers.append(picker)
        sparse = Tensor.create_from_torch(torch.stack(pickers).unsqueeze(0), constant=True)
    else:
        raise RuntimeError("Unknown config")

    @run(
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch),
    )
    def simple_sparse_matmul(act, sparse=None):
        if config == "in_out_stream":
            act = pybuda.op.Buffer("buf0", act)
        return pybuda.op.SparseMatmul("sparse0", sparse, act)

    simple_sparse_matmul(act, sparse=sparse)

    
def test_z_sparse_matmul(test_device):
    input_shape = (1, 64, 128, 128)

    class Model(PyBudaModule):
        def __init__(self):
            super().__init__(name="sparsematmul_test")
            rows = torch.arange(0, 128).tolist()
            cols = rows
            sparse = torch.sparse_coo_tensor([rows, cols],torch.ones(len(cols)), (128, 128), dtype=torch.float32)
            sparse = torch.stack([sparse]*64, -3)
            sparse = torch.unsqueeze(sparse, 0) 
            self.add_constant("sparse")
            self.set_constant("sparse", pybuda.Tensor.create_from_torch(sparse, constant=True))

        def forward(self, x):
            sparse = self.get_constant("sparse")
            out = pybuda.op.SparseMatmul("", self.get_constant("sparse"), x)
            return out

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "MaximizeTMinimizeGrid"

    pybuda.verify.verify_module(
        Model(),
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        ),
    )