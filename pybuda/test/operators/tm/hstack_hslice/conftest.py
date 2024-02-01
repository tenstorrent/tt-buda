# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Conftest for parameters setup for hstack, hslice operators
#

def pytest_addoption(parser):
    # model
    parser.addoption(
        "--tmh_model", 
        action="store", 
        default="model_1", 
        help="The name of the file in which the model is located."
    )
    # training
    parser.addoption(
        "--tmh_train", 
        action="store", 
        default=True, 
        help="Training or Inference."
    )
    # recompute
    parser.addoption(
        "--tmh_recompute", 
        action="store", 
        default=True, 
        help="Recompute or not."
    )
    # shape
    parser.addoption(
        "--tmh_shape", 
        action="store", 
        default=[1, 16, 32, 64], 
        help="Shape of the tensor."
    )
    # slice
    parser.addoption(
        "--tmh_slice", 
        action="store", 
        default=4, 
        help="Number of slices to create."
    )

def pytest_generate_tests(metafunc):

	option_model = metafunc.config.option.tmh_model
	if 'tmh_model' in metafunc.fixturenames and option_model is not None:
		metafunc.parametrize("tmh_model", [option_model])

	option_train = metafunc.config.option.tmh_train
	if 'tmh_train' in metafunc.fixturenames and option_train is not None:
		metafunc.parametrize("tmh_train", [option_train])

	option_recompute = metafunc.config.option.tmh_recompute
	if 'tmh_recompute' in metafunc.fixturenames and option_recompute is not None:
		metafunc.parametrize("tmh_recompute", [option_recompute])

	option_shape = metafunc.config.option.tmh_shape
	if 'tmh_shape' in metafunc.fixturenames and option_shape is not None:
		metafunc.parametrize("tmh_shape", [option_shape])

	option_slice = metafunc.config.option.tmh_slice
	if 'tmh_slice' in metafunc.fixturenames and option_slice is not None:
		metafunc.parametrize("tmh_slice", [option_slice])