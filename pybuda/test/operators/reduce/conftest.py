# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Conftest for parameters setup for reduce operators
#

def pytest_addoption(parser):
    # model
    parser.addoption(
        "--red_model", 
        action="store", 
        default="model_1", 
        help="The name of the file in which the model is located."
    )
    # training
    parser.addoption(
        "--red_train", 
        action="store", 
        default=True, 
        help="Training or Inference."
    )
    # recompute
    parser.addoption(
        "--red_recompute", 
        action="store", 
        default=True, 
        help="Recompute or not."
    )
    # shape
    parser.addoption(
        "--red_shape", 
        action="store", 
        default=[1, 16, 32, 64], 
        help="Shape of the tensor."
    )
    # operation
    parser.addoption(
        "--red_op", 
        action="store", 
        default="ReduceSum", 
        help="ReduceSum or ReduceAvg"
    )
    # dimension
    parser.addoption(
        "--red_dim", 
        action="store", 
        default=2, 
        help="Shape of the tensor."
    )
    # keep dimensions
    parser.addoption(
        "--red_keepdim", 
        action="store", 
        default=True, 
        help="Shape of the tensor."
    )

def pytest_generate_tests(metafunc):

	option_model = metafunc.config.option.red_model
	if 'red_model' in metafunc.fixturenames and option_model is not None:
		metafunc.parametrize("red_model", [option_model])

	option_train = metafunc.config.option.red_train
	if 'red_train' in metafunc.fixturenames and option_train is not None:
		metafunc.parametrize("red_train", [option_train])

	option_recompute = metafunc.config.option.red_recompute
	if 'red_recompute' in metafunc.fixturenames and option_recompute is not None:
		metafunc.parametrize("red_recompute", [option_recompute])

	option_shape = metafunc.config.option.red_shape
	if 'red_shape' in metafunc.fixturenames and option_shape is not None:
		metafunc.parametrize("red_shape", [option_shape])

	option_op = metafunc.config.option.red_op
	if 'red_op' in metafunc.fixturenames and option_op is not None:
		metafunc.parametrize("red_op", [option_op])

	option_dim = metafunc.config.option.red_dim
	if 'red_dim' in metafunc.fixturenames and option_dim is not None:
		metafunc.parametrize("red_dim", [option_dim])

	option_keepdim = metafunc.config.option.red_keepdim
	if 'red_keepdim' in metafunc.fixturenames and option_keepdim is not None:
		metafunc.parametrize("red_keepdim", [option_keepdim])