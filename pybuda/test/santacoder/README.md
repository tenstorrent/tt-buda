# KV-cache prefill and token-by-token decoding

## Installation

    pip install -r requirements.txt

These should be the pybuda-head compatible torch and transformers packages, but for reproducibility the versions are also stored here.

## Usage

    $ python prefill.py
    Saved KV cache to kv_cache.pt

    $ python decode.py --device silicon --fuse --precision fp16 --amp-level 2 --arch wormhole_b0
    <hf warnings about uninitialized tensors - ignore, the warning is mistaken, all tensors are initialized>
    <tvm warnings about accuracy - probably true, am curious as to the cause>
    <pybuda+bbe compilation>

     print("Hello World!")

This will download the SantaCoder model and run token-by-token decoding on it using a kv-cache, printing the output to stdout as it goes. You can change the default prompt/stop tokens with `--prompt` and `--stop`. Using `--prompt -` will allow you to enter a prompt from stdin. The output `print("Hello World!")` is an exact match for the CPU output.

The options:

    * `--precision fp16`: Santacoder is trained with CUDA AMP FP16, so FP16A is a good match on our hardware. With this setting the example output exactly matches CPU in FP32.
    * `--amp-level 2`: Uses bfp8_a for all matmuls and buffers and fp16 for fused ops, as used for BERT Large. With this setting the example output exactly matches CPU in FP32.
    * `--fuse`: Use op fusion. Without this pybuda adds a lot of weird ops between the attention mask and its consumers. But we want fusion anyway, it's good.

## Loading and saving TTImages

*Note: this is implemented but doesn't work with current pybuda at time of writing.*

You can compile once and save a TTImage to save recompiling the model every run:

    $ python decode.py ... --save image.tti

This is surprisingly slow and there is no progress bar. Ignore "fatal: Needed a single revision" - this comes from a non-essential call to git and is actually not fatal. To use this image in a future run:

    $ python decode.py ... --load image.tti

Typical timings:

|Mode|Time|
|:-:|:-:|
|Normal|3m23|
|Save|13m13|
|Load|@2m33: "Harvested device detected, but identity_map is false. This should not happen."|

Also available is a benchmark script, which can repeats prefill/decode steps on cpu or cuda and produces perf information, but this isn't maintained to the same degree and should be considered deprecated:

    $ python benchmark.py --device cpu
    Device: cpu
    Creating model... 
    Done (28.9 seconds)
    Prefill batch : (1, 2037) = 0.0 MB (int64)

    Time per batch       : 31017.3 ms
    Time per token       : 15.2 ms
    Batches per second   : 0.03 BPS
    Sequences per second : 0.03 SPS
    Tokens per second    : 65.67 TPS

    Decode kv_cache: 2x 24x (1, 128, 2037) = 47.7 MB (float32)

    Decode : 100%|███████████████████████████████████| 10/10 [00:11<00:00,  1.19s/it]
    Time per batch       : 1202.3 ms
    Time per token       : 1202.3 ms
    Batches per second   : 0.83 BPS
    Sequences per second : 0.83 SPS
    Tokens per second    : 0.83 TPS

    Total time (1 prefill + 10 decodes): 43.0 seconds @ 47.6 tokens/s

