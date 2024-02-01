# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Conftest for parameters setup for matmul operators
#

def pytest_addoption(parser):
    # model
    parser.addoption(
        "--mm_model", 
        action="store", 
        default="model_1", 
        help="The name of the file in which the model is located."
    )
    # training
    parser.addoption(
        "--mm_train", 
        action="store", 
        default=True, 
        help="Training or Inference."
    )
    # recompute
    parser.addoption(
        "--mm_recompute", 
        action="store", 
        default=True, 
        help="Recompute or not."
    )
    # shape
    parser.addoption(
        "--mm_shape", 
        action="store", 
        default=[1, 16, 32, 64], 
        help="Shape of the tensor."
    )

def pytest_generate_tests(metafunc):

	option_model = metafunc.config.option.mm_model
	if 'mm_model' in metafunc.fixturenames and option_model is not None:
		metafunc.parametrize("mm_model", [option_model])

	option_train = metafunc.config.option.mm_train
	if 'mm_train' in metafunc.fixturenames and option_train is not None:
		metafunc.parametrize("mm_train", [option_train])

	option_recompute = metafunc.config.option.mm_recompute
	if 'mm_recompute' in metafunc.fixturenames and option_recompute is not None:
		metafunc.parametrize("mm_recompute", [option_recompute])

	option_shape = metafunc.config.option.mm_shape
	if 'mm_shape' in metafunc.fixturenames and option_shape is not None:
		metafunc.parametrize("mm_shape", [option_shape])