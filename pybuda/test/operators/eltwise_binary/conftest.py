# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Conftest for parameters setup for element-wise binary operators
#

def pytest_addoption(parser):
    # model
    parser.addoption(
        "--bin_model", 
        action="store", 
        default="model_1", 
        help="The name of the file in which the model is located."
    )
    # training
    parser.addoption(
        "--bin_train", 
        action="store", 
        default=True, 
        help="Training or Inference."
    )
    # recompute
    parser.addoption(
        "--bin_recompute", 
        action="store", 
        default=True, 
        help="Recompute or not."
    )
    # shape
    parser.addoption(
        "--bin_shape", 
        action="store", 
        default=[1, 16, 32, 64], 
        help="Shape of the tensor."
    )
    # shape prologued
    parser.addoption(
        "--bin_shape_prologued", 
        action="store", 
        default='((2, 3, 3), InputSourceFlags.FROM_DRAM_NOT_PROLOGUED, False)', 
        help="Shape of the tensor, input source plag, should prolog flag."
    )
    # operation
    parser.addoption(
        "--bin_op",
        action="store",
        default='Add',
        help="Binary element-wise operation which we want to perform."
    )

def pytest_generate_tests(metafunc):

	option_model = metafunc.config.option.bin_model
	if 'bin_model' in metafunc.fixturenames and option_model is not None:
		metafunc.parametrize("bin_model", [option_model])

	option_train = metafunc.config.option.bin_train
	if 'bin_train' in metafunc.fixturenames and option_train is not None:
		metafunc.parametrize("bin_train", [option_train])

	option_recompute = metafunc.config.option.bin_recompute
	if 'bin_recompute' in metafunc.fixturenames and option_recompute is not None:
		metafunc.parametrize("bin_recompute", [option_recompute])

	option_shape = metafunc.config.option.bin_shape
	if 'bin_shape' in metafunc.fixturenames and option_shape is not None:
		metafunc.parametrize("bin_shape", [option_shape])
		
	option_shape_prologued = metafunc.config.option.bin_shape_prologued
	if 'bin_shape_prologued' in metafunc.fixturenames and option_shape_prologued is not None:
		metafunc.parametrize("bin_shape_prologued", [option_shape_prologued])

	option_op = metafunc.config.option.bin_op
	if 'bin_op' in metafunc.fixturenames and option_op is not None:
		metafunc.parametrize("bin_op", [option_op])
