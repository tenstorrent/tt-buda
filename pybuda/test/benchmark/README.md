PyBuda Benchmark Infrastructure
===============================

`benchmark.py` allows easy way to benchmark performance of a support model, while varying configurations and compiler options. The script will measure
real end-to-end time on host, starting post-compile at first input pushed, and ending at last output received. 

The script optionally outputs a .json file with benchmark results and options used to run the test. If the file already exists, the results will be appended,
allowing the user to run multiple benchmark back-to-back, like:

```
pybuda/test/benchmark/benchmark.py -m bert -c tiny -o perf.json
pybuda/test/benchmark/benchmark.py -m bert -c base -o perf.json
pybuda/test/benchmark/benchmark.py -m bert -c large -o perf.json
```

`perf.json` will have performance results for all 3 configurations of bert.

To see which models and configurations are currently supported, run:

```
pybuda/test/benchmark/benchmark.py --list
```

Full Usage
----------
```
usage: benchmark.py [-h] [-m MODEL] [-c CONFIG] [-t] [-df {Fp32,Fp16,Fp16_b,Bfp8,Bfp8_b,Bfp4,Bfp4_b}] [-mf {LoFi,HiFi2,HiFi3,HiFi4}] [-opt {0,1,2,3}]
                    [--loop_count LOOP_COUNT] [--chips CHIPS] [--layers LAYERS] [--trace {none,light,verbose}] [-l] [-o OUTPUT]

Benchmark a model on TT hardware

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model to benchmark (i.e. bert)
  -c CONFIG, --config CONFIG
                        Model configuration to benchmark (i.e. tiny, base, large)
  -t, --training        Benchmark training
  -df {Fp32,Fp16,Fp16_b,Bfp8,Bfp8_b,Bfp4,Bfp4_b}, --dataformat {Fp32,Fp16,Fp16_b,Bfp8,Bfp8_b,Bfp4,Bfp4_b}
                        Set data format
  -mf {LoFi,HiFi2,HiFi3,HiFi4}, --math_fidelity {LoFi,HiFi2,HiFi3,HiFi4}
                        Set math fidelity
  -opt {0,1,2,3}, --backend_opt_level {0,1,2,3}
                        Set backend optimization level
  --loop_count LOOP_COUNT
                        Set the number of times to loop through the model. By default, it will be 5x the number of chips.
  --chips CHIPS         Number of chips to run benchmark on. 0 to run on all available.
  --layers LAYERS       Number of layers to run on models where this is applicable (i.e. nlp encoders/decoders)
  --trace {none,light,verbose}
                        Performance trace to be generated during the run.
  -l, --list            List available models and configurations
  -o OUTPUT, --output OUTPUT
                        Output json file to write results to, optionally. If file already exists, results will be appended.
```

Adding Models
-------------

To add a model, add a new file to `models/` directory (or add to an existing one) and create a function with the name of your model, and decorate it with
`@benchmark_model` decorator. If your model supports configurations, add `configs=[....]` parmeter to the decorator to define the legal configs. For example:

```python
@benchmark_model(configs=["tiny", "base", "large"])
def bert(training: bool, config: str, force_num_layers: Optional[int] = None):
```

`config` is an optional parameter to your model, but should be there if you've defined legal configs in the decorator. Similarly, `force_num_layers` is
optional, and should be used if your model supports the concept of modifiable number of layers.

The function should then create the `Module` that will run on TT device, and optionally `PyTorchModule`s for CPU to run before and after the module on device. 

The function needs to return a list of modules - [cpu-pre, tt, cpu-post], or smaller if there's no cpu-post or cpu-pre, and a list of torch tensors that can
be used as inputs. The benchmark module will push those same inputs over and over again.

