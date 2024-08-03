## RGG Random Graph Generator

Random Graph Generator supports testing of randomly generated graphs. Tests based on RGG can be triggered as regular pytests and executed in a single run of pytest. Tests are performed as a bulk operation for the specified number of tests.

### Execution

For each random model RGG passes through steps:

 - Generate random model for specific random_seed
 - Verify model via verify_module

Source code of each randomly generated model with a pytest function can be automatically stored in a folder, ex `test/random_tests/` if configured.

## Run

Entrypoint for RGG pytests is in `test_graphs.py` module

Example command for running PyBuda RGG tests generated via random graph algorithm

```shell
LOGURU_LEVEL=DEBUG RANDOM_TEST_COUNT=5 MIN_DIM=3 MAX_DIM=4 MIN_OP_SIZE_PER_DIM=16 MAX_OP_SIZE_PER_DIM=64 MIN_MICROBATCH_SIZE=1 MAX_MICROBATCH_SIZE=8 NUM_OF_NODES_MIN=5 NUM_OF_NODES_MAX=10 NUM_OF_FORK_JOINS_MAX=5 CONSTANT_INPUT_RATE=20 SAME_INPUTS_PERCENT_LIMIT=10 pytest -svv pybuda/test/random/test_graphs.py::test_random_graph_algorithm_pybuda
```

Example command for running PyTorch RGG tests generated via random graph algorithm

```shell
LOGURU_LEVEL=DEBUG RANDOM_TEST_COUNT=5 MIN_DIM=4 MAX_DIM=4 MIN_OP_SIZE_PER_DIM=4  MAX_OP_SIZE_PER_DIM=8  MIN_MICROBATCH_SIZE=1 MAX_MICROBATCH_SIZE=1 NUM_OF_NODES_MIN=3 NUM_OF_NODES_MAX=5  NUM_OF_FORK_JOINS_MAX=5 CONSTANT_INPUT_RATE=20 SAME_INPUTS_PERCENT_LIMIT=10 pytest -svv pybuda/test/random/test_graphs.py::test_random_graph_algorithm_pytorch
```

## Configuration

Configuration of RGG is supported via `RandomizerConfig`

Parameters includes configuration of:

 - framework
 - number of tests
 - number of nodes
 - min and max size of an operand dimension
 - ...

For more details about configuration please take a look at `pybuda/test/random/rgg/config.py`.

Please refer to full list of supported enviroment variables in [README.debug.md](../README.debug.md)

## Development

Entrypoint for RGG impplementation is `process_test` module

Parameters of process_test pytest:

 - test_index - index of a test
 - random_seed - random seed of a test
 - test_device - target test device
 - randomizer_config - test configation parameters
 - graph_builder_type - algorithm
 - framework - target framework
