# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Conftest for parameters setup for reshape operators
#

def pytest_addoption(parser):
    # model
    parser.addoption(
        "--resh_model", 
        action="store", 
        default="model_1", 
        help="The name of the file in which the model is located."
    )
    # training
    parser.addoption(
        "--resh_train", 
        action="store", 
        default=True, 
        help="Training or Inference."
    )
    # recompute
    parser.addoption(
        "--resh_recompute", 
        action="store", 
        default=True, 
        help="Recompute or not."
    )
    # old shape
    parser.addoption(
        "--resh_oshape", 
        action="store", 
        default=[1, 16, 32, 64], 
        help="Shape of the tensor."
    )
    # new shape
    parser.addoption(
        "--resh_nshape", 
        action="store", 
        default=[1, 32, 32, 32], 
        help="New shape of the tensor."
    )

def pytest_generate_tests(metafunc):

	option_model = metafunc.config.option.resh_model
	if 'resh_model' in metafunc.fixturenames and option_model is not None:
		metafunc.parametrize("resh_model", [option_model])

	option_train = metafunc.config.option.resh_train
	if 'resh_train' in metafunc.fixturenames and option_train is not None:
		metafunc.parametrize("resh_train", [option_train])

	option_recompute = metafunc.config.option.resh_recompute
	if 'resh_recompute' in metafunc.fixturenames and option_recompute is not None:
		metafunc.parametrize("resh_recompute", [option_recompute])

	option_oshape = metafunc.config.option.resh_oshape
	if 'resh_oshape' in metafunc.fixturenames and option_oshape is not None:
		metafunc.parametrize("resh_oshape", [option_oshape])

	option_nshape = metafunc.config.option.resh_nshape
	if 'resh_nshape' in metafunc.fixturenames and option_nshape is not None:
		metafunc.parametrize("resh_nshape", [option_nshape])