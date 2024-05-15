# 
# Commands for running matmul tests
# 

# Run single test
# 
# To run using default parameters
# model,     --mm_model     --> model_1, Note: for matmul ops we have 7 models, model_[1-3, 6-8, 10]
# training,  --mm_train     --> True
# recompute, --mm_recompute --> True
# shape,     --mm_shape     --> [1, 16, 32, 64]
pytest -svv test_matmul_single.py

# Few examples with passed arguments
pytest -svv test_matmul_single.py --mm_model model_3 --mm_train True --mm_recompute True --mm_shape '[1, 32, 96, 128]'
pytest -svv test_matmul_single.py --mm_model model_1 --mm_train False --mm_recompute True --mm_shape '[1, 32, 256, 128]'
pytest -svv test_matmul_single.py --mm_model model_2 --mm_train True --mm_recompute False
pytest -svv test_matmul_single.py --mm_model model_7 --mm_train False
pytest -svv test_matmul_single.py --mm_model model_6 --mm_shape '[1, 32, 256, 2048]'

# Issues
pytest -svv test_matmul_single.py --mm_model model_2 --mm_train False --mm_recompute True --mm_shape '[38, 236, 141, 73]'
pytest -svv test_matmul_single.py --mm_model model_2 --mm_train False --mm_recompute True --mm_shape '[138, 204, 134, 134]'

pytest -svv test_matmul_single.py --mm_model model_2 --mm_train False --mm_recompute True --mm_shape '[1, 1, 8192, 8192]'

#Issues - per PyTorchs docs
GOLDEN_WORMHOLE_B0=1 pytest -svv pybuda/test/operators/matmul/test_matmul_single.py::test_matmul_according_to_pytorch_docs_single --mm_model model_11 --runxfail --no-skips
GOLDEN_WORMHOLE_B0=1 pytest -svv pybuda/test/operators/matmul/test_matmul_single.py::test_matmul_according_to_pytorch_docs_single --mm_model model_12 --runxfail --no-skips
GOLDEN_WORMHOLE_B0=1 pytest -svv pybuda/test/operators/matmul/test_matmul_single.py::test_matmul_according_to_pytorch_docs_single --mm_model model_13 --runxfail --no-skips
GOLDEN_WORMHOLE_B0=1 pytest -svv pybuda/test/operators/matmul/test_matmul_single.py::test_matmul_according_to_pytorch_docs_single --mm_model model_14 --runxfail --no-skips
GOLDEN_WORMHOLE_B0=1 pytest -svv pybuda/test/operators/matmul/test_matmul_single.py::test_matmul_according_to_pytorch_docs_single --mm_model model_15 --runxfail --no-skips

#Issues - when input shape is (1, 1, 10000, 1) - extreme ratios between height/width
GOLDEN_WORMHOLE_B0=1 pytest -svv pybuda/test/operators/matmul/test_matmul_single.py::test_matmul_according_to_test_plan_single --mm_model model_op_src_from_another_op  --mm_shape '[1, 1, 10000, 1]' --runxfail --no-skips
GOLDEN_WORMHOLE_B0=1 pytest -svv pybuda/test/operators/matmul/test_matmul_single.py::test_matmul_according_to_test_plan_single --mm_model model_op_src_from_dram2       --mm_shape '[1, 1, 10000, 1]' --runxfail --no-skips
GOLDEN_WORMHOLE_B0=1 pytest -svv pybuda/test/operators/matmul/test_matmul_single.py::test_matmul_according_to_test_plan_single --mm_model model_op_src_const_inputs1    --mm_shape '[1, 1, 10000, 1]' --runxfail --no-skips
GOLDEN_WORMHOLE_B0=1 pytest -svv pybuda/test/operators/matmul/test_matmul_single.py::test_matmul_according_to_test_plan_single --mm_model model_op_src_const_inputs2    --mm_shape '[1, 1, 10000, 1]' --runxfail --no-skips
GOLDEN_WORMHOLE_B0=1 pytest -svv pybuda/test/operators/matmul/test_matmul_single.py::test_matmul_according_to_test_plan_single --mm_model model_op_src_from_host        --mm_shape '[1, 1, 10000, 1]' --runxfail --no-skips