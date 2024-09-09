# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Conftest for parameters setup for element-wise unary operators
#

import json


def pytest_addoption(parser):
    # model
    parser.addoption(
        "--un_model", 
        action="store", 
        default="model_4", 
        help="The name of the file in which the model is located."
    )
    # training
    parser.addoption(
        "--un_train", 
        action="store", 
        default=True, 
        help="Training or Inference."
    )
    # recompute
    parser.addoption(
        "--un_recompute", 
        action="store", 
        default=True, 
        help="Recompute or not."
    )
    # shape
    parser.addoption(
        "--un_shape", 
        action="store", 
        default=[1, 32, 64], 
        help="Shape of the tensor."
    )
    # operation
    parser.addoption(
        "--un_op",
        action="store",
        default='Sqrt',
        help="Unary element-wise operation which we want to perform."
    )
	# kwargs
    parser.addoption(
        "--un_kwargs_json",
        action="store",
        default='{}',
        help="Additional arguents, in JSON format, for given operation. If they are needed."
	)

def pytest_generate_tests(metafunc):

	option_model = metafunc.config.option.un_model
	if 'un_model' in metafunc.fixturenames and option_model is not None:
		metafunc.parametrize("un_model", [option_model])

	option_train = metafunc.config.option.un_train
	if 'un_train' in metafunc.fixturenames and option_train is not None:
		metafunc.parametrize("un_train", [option_train])

	option_recompute = metafunc.config.option.un_recompute
	if 'un_recompute' in metafunc.fixturenames and option_recompute is not None:
		metafunc.parametrize("un_recompute", [option_recompute])

	option_shape = metafunc.config.option.un_shape
	if 'un_shape' in metafunc.fixturenames and option_shape is not None:
		shape = eval(option_shape) if type(option_shape) == str else option_shape
		metafunc.parametrize("un_shape", [shape])

	option_op = metafunc.config.option.un_op
	if 'un_op' in metafunc.fixturenames and option_op is not None:
		metafunc.parametrize("un_op", [option_op])

	option_kwargs = metafunc.config.option.un_kwargs_json
	if 'un_kwargs' in metafunc.fixturenames and option_kwargs is not None:
		metafunc.parametrize("un_kwargs", [json.loads(option_kwargs)])