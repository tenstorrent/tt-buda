# 
# Commands for running element-wise binary tests
# 

# Run single test
# 
# To run using default parameters
# model,     --bin_model     --> model_1, Note: for binary ops we have 11 models, model_[1-11]
# training,  --bin_train     --> True
# recompute, --bin_recompute --> True
# shape,     --bin_shape     --> [1, 16, 32, 64]
# operation, --bin_op        --> Add
pytest -svv test_eltwise_binary_single.py

# Few examples with passed arguments
pytest -svv test_eltwise_binary_single.py --bin_model model_3 --bin_train True --bin_recompute True --bin_shape '[1, 32, 96, 128]' --bin_op 'Add'
pytest -svv test_eltwise_binary_single.py --bin_model model_1 --bin_train False --bin_recompute True --bin_shape '[1, 32, 256, 128]'
pytest -svv test_eltwise_binary_single.py --bin_model model_2 --bin_train True --bin_recompute False
pytest -svv test_eltwise_binary_single.py --bin_model model_5 --bin_train False
pytest -svv test_eltwise_binary_single.py --bin_model model_4 --bin_shape '[1, 32, 256, 2048]'

# Issue commands 
pytest -svv test_eltwise_binary_single.py --bin_model model_2 --bin_train True --bin_recompute True --bin_op 'Subtract' --bin_shape '[21, 127, 102, 19]'
pytest -svv test_eltwise_binary_single.py --bin_model model_2 --bin_train True --bin_recompute True --bin_op 'Subtract' --bin_shape '[29, 30, 15, 51]'

pytest -svv test_eltwise_binary_single.py --bin_model model_2 --bin_train True --bin_recompute True --bin_op 'Heaviside' --bin_shape '[29, 30, 15, 51]'

pytest -svv test_eltwise_binary_single.py --bin_model model_2 --bin_train True --bin_recompute True --bin_op 'Max' --bin_shape '[29, 30, 15, 51]'
pytest -svv test_eltwise_binary_single.py --bin_model model_2 --bin_train True --bin_recompute True --bin_op 'Max' --bin_shape '[114, 120, 95]'

pytest -svv test_eltwise_binary_single.py --bin_model model_4 --bin_train True --bin_recompute False --bin_op 'Add' --bin_shape '[29, 30, 15, 51]'
pytest -svv test_eltwise_binary_single.py --bin_model model_4 --bin_train True --bin_recompute False --bin_op 'Add' --bin_shape '[76, 6, 80]'
pytest -svv test_eltwise_binary_single.py --bin_model model_4 --bin_train True --bin_recompute False --bin_op 'Add' --bin_shape '[108, 13, 73]'

pytest -svv test_eltwise_binary_single.py --bin_model model_1 --bin_train True --bin_recompute False --bin_op 'Add' --bin_shape '[1, 1, 10000, 10000]'

# Run single test according to the new test plan 
pytest -svv test_eltwise_binary.py::test_eltwise_binary_ops_per_test_plan_single --bin_model 'ModelOpSrcFromTmEdge1' --bin_shape '(1, 1000, 100)' --bin_op "Heaviside" --runxfail --no-skips
pytest -svv test_eltwise_binary.py::test_eltwise_binary_ops_per_test_plan_single --bin_model 'ModelFromAnotherOp' --bin_shape '(1, 1, 9920, 1)' --bin_op "Equal" --runxfail --no-skips
pytest -svv test_eltwise_binary.py::test_eltwise_binary_ops_per_test_plan_single --bin_model 'ModelFromAnotherOp' --bin_shape '(1, 1, 9920, 1)' --bin_op "NotEqual" --runxfail --no-skips
pytest -svv test_eltwise_binary.py::test_eltwise_binary_ops_per_test_plan_single --bin_model 'ModelFromAnotherOp' --bin_shape '(1, 3, 3)' --bin_op "BinaryStack" --runxfail --no-skips
pytest -svv test_eltwise_binary.py::test_eltwise_binary_ops_per_test_plan_single --bin_model 'ModelConstEvalPass' --bin_shape '(1, 3, 3, 3)' --bin_op "BinaryStack" --runxfail --no-skips
pytest -svv test_eltwise_binary.py::test_eltwise_binary_ops_per_test_plan_single --bin_model 'ModelFromAnotherOp' --bin_shape '(1, 1, 10, 1000)' --bin_op "BinaryStack" --runxfail --no-skips

pytest -svv test_eltwise_binary.py::test_eltwise_binary_ops_per_test_plan_single_prologued --bin_shape_prologued '((2, 3, 3), InputSourceFlags.FROM_DRAM_NOT_PROLOGUED, False)' --bin_op "BinaryStack" --runxfail --no-skips
