# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import inspect
import pybuda
from loguru import logger
import torch
from collections import OrderedDict
import transformers
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.auto.tokenization_auto import AutoTokenizer
from pybuda.pybudaglobal import align_up_tile
from pybuda.tensor import remove_microbatch

class NLPPipelineWrapper(torch.nn.Module):
    """
    Wrapper for transformers nlp pipeline. Provide to pipeline(...) call as model.
    """
    def __init__(self, model, tokenizer, name="pb_model", use_cache=None, fp32_fallback=pybuda.DataFormat.Float16_b, forward_fn=None, max_length=None):
        super().__init__()

        #pybuda.config._get_global_compiler_config().verify_pybuda_codegen_vs_framework = False
        self.original_fwd = model.forward
        self.forward_args = list(inspect.signature(model.forward).parameters.keys())
        self.forward_args_dict = list(inspect.signature(model.forward).parameters.items())
        self.forward_fn = forward_fn
        if forward_fn is None:
            self.module = pybuda.PyTorchModule(name, model, redirect_forward=False)
            self.ttdevice = pybuda.TTDevice("tt0", module=self.module, fp32_fallback=fp32_fallback)
        self.pad_token_id = tokenizer.pad_token_id
        self.config = model.config
        self.model = model
        self.ordered_kwargs = None
        self.tensor_input_names = []
        self.generated_input_names = []
        self.orig_len = None
        self.use_cache = use_cache
        self.idx = 0
        if max_length is None:
            # Ideally this should be determined by model's largest seqlen, but
            # we will need padding for prime number of tiles. 
            max_length = 256

        self.max_length = max_length
    def tt_forward(self, *inputs, **kwargs):
        logger.info("Starting TT forward")

        for k in self.forward_args:
            if (k in kwargs) and kwargs[k] is not None and not isinstance(kwargs[k], bool):
                self.tensor_input_names.append(k)
                if k == 'encoder_outputs':
                    inputs = (*inputs, kwargs[k].last_hidden_state)
                else:
                    inputs = (*inputs, kwargs[k])

        if self.forward_fn is not None:
            logits = self.forward_fn(inputs)
        else:
            inputs = list(inputs)
            inputs = [i.int() if isinstance(i, torch.Tensor) and not torch.is_floating_point(i) else i for i in inputs]
            self.ttdevice.push_to_inputs(inputs)
            output_q = pybuda.run_inference(_sequential=True)
            logits = output_q.get()[0].value()
            logits = logits[:, :self.orig_len, :]

        return CausalLMOutputWithCrossAttentions(logits=logits)


    # This compilation forward is only used for torchscript tracing.
    # It is a simple pass-through while grouping the inputs into a dictionary,
    # which is then passed to the original forward function
    def compilation_forward(self, *inputs, **kwargs):
        assert len(self.tensor_input_names) == len(inputs)
        # Since kwargs are ordered, we can zip them with the list of inputs
        input_dict = OrderedDict(zip(self.tensor_input_names, inputs))
        self.ordered_kwargs.update(input_dict)

        # We need to feed dictionary input because some arguments are left as None,
        # If we feed them as a list, there will be a mismatch.
        if ("encoder_outputs" in self.ordered_kwargs):
            self.ordered_kwargs["encoder_outputs"] = (self.ordered_kwargs["encoder_outputs"], )

        if "return_dict" in self.forward_args:
            self.ordered_kwargs["return_dict"] = False # we don't want dictionaries when tracing

        if "use_cache" in self.forward_args:
            self.ordered_kwargs["use_cache"] = self.use_cache

        if len(self.generated_input_names):
            for k in self.generated_input_names:
                self.ordered_kwargs[k] = remove_microbatch([self.ordered_kwargs[k],])[0].value()

        return self.original_fwd(**self.ordered_kwargs)


    def first_forward(self, *inputs, **kwargs):
        if "return_dict" in kwargs:
            self.ordered_kwargs["return_dict"] = False # we don't want dictionaries when tracing

        if self.forward_fn is not None or self.ttdevice._compiled:
            self.model.forward = self.original_fwd
            return self.tt_forward(*inputs, **self.ordered_kwargs)
        else:
            # Assign the compilation_forward for torchscript tracing
            self.module.forward = self.compilation_forward
            self.model.forward = self.compilation_forward
            out = self.tt_forward(*inputs, **self.ordered_kwargs)
            self.model.forward = self.first_forward
        return out

    def prepare_inputs_for_generation(self, *inputs, **kwargs):
        # Convert kwargs to ordered dict so that we could save input names
        # and pair with the list of inputs in compilation_forward
        input_ids = kwargs["input_ids"] if "input_ids" in kwargs else inputs[0]

        # Prepare inputs
        orig_len = len(input_ids[0])
        
        total_length = self.max_length

        pad_len = total_length - orig_len

        if "cache_position" in kwargs:            
            # Cache positions indicate the positions of input sequence tokens within the sequence. They're utilized to
            # update cache positions and to deduce the complete sequence length.
            #
            # However, cache_position is presumed to be unpadded (for accurate sequence length calculations). Consequently,
            # it poses issues during compilation, particularly as we assume tile-aligned dimensions at some point. Therefore, 
            # cache_position is expected to be arranged from (1, orig_len) and serves as a constant for the model if not defined 
            # as input. Hence, we're removing it from the kwargs and relying on the model's default.
            #
            # For more details, refer to the following code snippet:
            # - build/python_env/lib/python3.10/site-packages/transformers/generation/utils.py:2404 =>
            # => model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
            #
            # This displays that cache_position is generated internally during pipeline setup, and is not expected to be
            # provided as input for the model.
            
            logger.warning("Removing cache_position from kwargs. It is not expected to be provided as input for the model.")
            kwargs.pop("cache_position", None)

        input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=self.pad_token_id)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[:, -pad_len:] = 0

        ordered_kwargs = OrderedDict()
        if not ("encoder_outputs" in kwargs and "decoder_input_ids" not in kwargs):
            ordered_kwargs["input_ids"] = input_ids
        if "attention_mask" in kwargs:
            ordered_kwargs["attention_mask"] = attention_mask.float()

        if "encoder_outputs" in kwargs:
            encoder_outputs = kwargs["encoder_outputs"]
            encoder_pad_len = align_up_tile(encoder_outputs["last_hidden_state"].shape[-2]) - encoder_outputs["last_hidden_state"].shape[-2]
            encoder_outputs["last_hidden_state"] = torch.nn.functional.pad(encoder_outputs["last_hidden_state"], (0, 0, 0, encoder_pad_len, 0, 0))

            if "decoder_input_ids" in kwargs:
                decoder_input_ids = kwargs["decoder_input_ids"]
                orig_len = kwargs["decoder_input_ids"].shape[-1]
                decoder_input_ids_pad_len = total_length - orig_len
                decoder_input_ids = kwargs["decoder_input_ids"]
                decoder_input_ids = torch.nn.functional.pad(decoder_input_ids, (0, decoder_input_ids_pad_len, 0, 0), value=self.pad_token_id)
                decoder_attention_mask = torch.ones_like(decoder_input_ids)
                decoder_attention_mask[:, -decoder_input_ids_pad_len:] = 0
                ordered_kwargs["decoder_input_ids"] = decoder_input_ids.int()
                ordered_kwargs["decoder_attention_mask"] = decoder_attention_mask.float()
                self.generated_input_names.append("decoder_attention_mask")
            else:
                decoder_input_ids = inputs[0]
                org_len = decoder_input_ids.shape[-1]
                decoder_input_ids_pad_len = total_length - orig_len
                decoder_input_ids = torch.nn.functional.pad(decoder_input_ids, (0, decoder_input_ids_pad_len, 0, 0), value=self.pad_token_id)
                ordered_kwargs["decoder_input_ids"] = decoder_input_ids.int()

            if "attention_mask" in ordered_kwargs:
                ordered_kwargs["attention_mask"] = None

            ordered_kwargs["encoder_outputs"] = encoder_outputs

        for k,v in kwargs.items():
            if k not in ordered_kwargs:
                ordered_kwargs[k] = v

        for fwd_key, fwd_val in self.forward_args_dict:
            # checking whether there is any required args in model forward function
            # if true updating ordered_kwargs with value set to None
            if not (fwd_val.kind in [inspect.Parameter.VAR_POSITIONAL,inspect.Parameter.VAR_KEYWORD] or fwd_val.default is not inspect.Parameter.empty):
                ordered_kwargs[fwd_key] = None

        self.ordered_kwargs = ordered_kwargs
        self.orig_len = orig_len

        self.model.forward = self.first_forward

        return ordered_kwargs 


    @classmethod
    def from_pretrained(cls, name, pipeline, use_cache, forward_fn=None, max_length=None):
        """
        Returns model and tokenizer for the given pipeline
        """

        tasks = transformers.pipelines.SUPPORTED_TASKS
        lookup_name = pipeline
        if pipeline.startswith("translation"):
            lookup_name = "translation"
        model = tasks[lookup_name]["pt"][0].from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)

        wrapper = NLPPipelineWrapper(model, tokenizer, name.replace("-", "_"), use_cache=use_cache, forward_fn=forward_fn, max_length=max_length)
        model.prepare_inputs_for_generation = wrapper.prepare_inputs_for_generation
        return model, tokenizer

def pipeline(pipeline_type: str, *args, **kwargs):
    m = kwargs["model"]
    forward_fn = None
    pybuda_max_length = None
    if "forward_fn" in kwargs:
        forward_fn = kwargs.pop("forward_fn")

    if "pybuda_max_length" in kwargs:
        pybuda_max_length = kwargs.pop("pybuda_max_length")

    use_cache = None if "use_cache" not in kwargs else kwargs["use_cache"]
    if isinstance(m, str):
        model, tokenizer = NLPPipelineWrapper.from_pretrained(m, pipeline_type, use_cache=use_cache, forward_fn=forward_fn, max_length=pybuda_max_length)
        kwargs["model"] = model
        if "tokenizer" not in kwargs:
            kwargs["tokenizer"] = tokenizer

    elif isinstance(m, torch.nn.Module):
        kwargs["model"].prepare_inputs_for_generation = NLPPipelineWrapper(m, kwargs["tokenizer"], m.__class__.__name__, use_cache=use_cache, forward_fn=forward_fn, max_length=pybuda_max_length).prepare_inputs_for_generation

    else:
        raise RuntimeError("Unsupported model type")

    return transformers.pipeline(pipeline_type, *args, **kwargs)