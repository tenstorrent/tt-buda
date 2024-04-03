# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import ray
from ray import serve
import torch
import pybuda
from transformers.pipelines import pipeline
import pybuda
from loguru import logger
import json

from pybuda import PyTorchModule
from transformers import BertModel, BertConfig, BertForPreTraining, BertTokenizer, BertForQuestionAnswering
from pybuda.verify import VerifyConfig
from pybuda._C.backend_api import BackendType, BackendDevice
from dataclasses import dataclass

# Embedding wrapper that extends and passes attention mask through - to run on host
class EmbWrapper(torch.nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
    def forward(self, input_ids, attention_mask, token_type_ids):
        attention_mask = attention_mask * 1.0
        emb_output = self.bert.embeddings(input_ids, token_type_ids)
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        return emb_output, extended_attention_mask

# Wrapper for encoders + QA output - to run on TT
class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, emb_output, extended_attention_mask):
        out = self.model.bert.encoder(emb_output, extended_attention_mask)
        out = self.model.qa_outputs(out.last_hidden_state)
        return out

@dataclass
class TestDevice:
    devtype: BackendType
    arch: BackendDevice


@serve.deployment(route_prefix="/bert-qa", name="bert-qa")
class BertQADeploy:
    def __init__(self):
        size = "base"

        if size == "tiny":
            model_name = "mrm8488/bert-tiny-finetuned-squadv2"
            #context = "Manuel Romero has been working hardly in the repository hugginface/transformers lately"
            #input_q = {"context": context, "question": "For which company has worked Manuel Romero?"}
        elif size == "base":
            # https://huggingface.co/phiyodr/bert-base-finetuned-squad2
            model_name = "phiyodr/bert-base-finetuned-squad2"
            #context = "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."

            #input_q = {"context": context, "question": "What discipline did Winkelmann create?"}
        else:
            raise RuntimeError("Unknown size")

        model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
        tokenizer = BertTokenizer.from_pretrained(model_name, pad_to_max_length=True)
        self.nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

        test_device = TestDevice(devtype=BackendType.Golden, arch=BackendDevice.Wormhole_B0)


        # Create pipeline, with encoders on TT
        self.cpu0 = pybuda.CPUDevice("cpu0", module=PyTorchModule("bert_embeddings", EmbWrapper(model.bert)))
        tt1 = pybuda.TTDevice("tt1", 
                devtype=test_device.devtype, arch=test_device.arch, module=PyTorchModule("encoder", EncoderWrapper(model)))

            
    async def __call__(self, request):
        data = await request.body()

        import urllib.parse
        input_q = urllib.parse.parse_qs(data)

        input_q = json.loads(data)

        examples = self.nlp._args_parser(input_q)
        preprocess_params, _, postprocess_params = self.nlp._sanitize_parameters()

        inputs = []
        for model_inputs in self.nlp.preprocess(examples[0], **preprocess_params):
            inputs.append( {
                "data": (model_inputs["input_ids"], model_inputs["attention_mask"], model_inputs["token_type_ids"]),
                "example": model_inputs["example"],
                "inputs": model_inputs})

        input = inputs[0]
        logger.info("Running on TT")
        self.cpu0.push_to_inputs(input["data"])
        output_q = pybuda.run_inference(_verify_cfg=VerifyConfig.disabled(), _sequential=True)

        outputs = output_q.get()
        logits = outputs[0].value().detach()
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous().type(torch.float32)
        end_logits = end_logits.squeeze(-1).contiguous().type(torch.float32)
        
        res = {"start": start_logits, "end": end_logits, "example": input["example"], **input["inputs"]}

        tt_answer = self.nlp.postprocess([res], **postprocess_params)

        return tt_answer

import ray
from ray import serve

# Connect to the running Ray Serve instance.
ray.init(address='auto', namespace="serve-example", ignore_reinit_error=True)
serve.start(detached=True)

# Deploy the model.
BertQADeploy.deploy()
