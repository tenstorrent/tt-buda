## RGG Random Graph Generator

Random Graph Generator supports testing of randomly generated graphs. Tests based on RGG can be triggered as regular pytests and executed in a single run of pytest. Tests are performed as a bulk operation for the specified number of tests.

### Execution

For each random model RGG passes through steps:

 - Generate random model for specific random_seed
 - Verify model via verify_module

Source code of each randomly generated model with a pytest function can be automatically stored in a folder, ex `test/random_tests/` if configured.

## Run

Entrypoint for RGG pytests is `test_graphs.py` module

```bash
pytest -svv pybuda/test/random/test_graphs.py`
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

## Development

Entrypoint for RGG impplementation is `process_test` module

Parameters of process_test pytest:

 - test_index - index of a test
 - random_seed - random seed of a test
 - test_device - target test device
 - randomizer_config - test configation parameters
 - graph_builder_type - algorithm
 - framework - target framework
