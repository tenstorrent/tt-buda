# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda._C import MathFidelity
import pytest
from typing import List, Optional, Union

import torch
from torch import nn
from loguru import logger
import os


from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

import pybuda
from pybuda import (
    PyTorchModule,
    TTDeviceImage,
    BackendType,
    VerifyConfig,
    BackendDevice,
    PyBudaModule,
)

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from test.utils import download_model
from PIL import Image



class UnetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, latent_model_input, timestep, text_embeddings):
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings)["sample"]
        return noise_pred

def test_unet(test_device):
    import os
    # os.environ["PYBUDA_RELOAD_GENERATED_MODULES"] = "1"
    os.environ["PYBUDA_MAX_GRAPH_CUT_RETRY"] = "2000"
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"
    os.environ["PYBUDA_PAD_SPARSE_MM_WEIGHT_MM"] = "{10:12, 20:24, 30:32, 40:48, 60:64}"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"
    pipe = download_model(StableDiffusionPipeline.from_pretrained, "CompVis/stable-diffusion-v1-4")

    mod = PyTorchModule("sd_unet", UnetWrapper(pipe.unet))
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.enable_enumerate_u_kt = False
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.retain_tvm_python_files = True
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.default_dram_parameters = True

    compiler_cfg.balancer_policy = "Ribbon"



    input_shape = ((1, 4, 64, 64), (1,), (1, 77, 768))
    # ttdevice = pybuda.TTDevice("tt0", module=mod,arch=BackendDevice.Wormhole_B0, devtype=BackendType.Silicon)
    # TTDeviceImage = ttdevice.compile_to_image(
    #     img_path="device_images/sd_unet_final.tti",
    #     training=False,
    #     sample_inputs=[torch.randn(shape) for shape in input_shape],
    # )

    verify_module(
        mod,
        input_shape,
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,),
    )


class UnetCrossAttentionWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.attn = unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1

    def forward(self, hidden_states,):
        hidden_states = self.attn(hidden_states)

        return hidden_states

def test_unet_CrossAttention(test_device):
    pytest.skip()
    pipe = download_model(StableDiffusionPipeline.from_pretrained, "CompVis/stable-diffusion-v1-4")

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.balancer_op_override("softmax_11.dc.subtract.1", "t_stream_shape", (16,1))
    compiler_cfg.place_on_new_epoch("softmax_11.dc.subtract.1")
    compiler_cfg.place_on_new_epoch("matmul_17")
    compiler_cfg.place_on_new_epoch("hstack_19.dc.sparse_matmul.4.lc2")
    compiler_cfg.place_on_new_epoch("softmax_11.dc.reduce_max.0")
    compiler_cfg.place_on_new_epoch("softmax_11.dc.reciprocal.6_s_brcst_m1_0_0.lc1")
    compiler_cfg.place_on_new_epoch("softmax_11.dc.multiply.7")
    compiler_cfg.graph_solver_self_cut_type = "ConsumerOperandDataEdgesFirst"
    compiler_cfg.retain_tvm_python_files = True
    compiler_cfg.enable_auto_fusing = False

    compiler_cfg.default_df_override = pybuda.DataFormat.Bfp8_b
    compiler_cfg.default_math_fidelity = MathFidelity.LoFi


    compiler_cfg.balancer_policy = "Ribbon"

    mod = PyTorchModule("sd_unet_cross_attention", UnetCrossAttentionWrapper(pipe.unet))

    input_shape = ((2, 4096, 320), )
    verify_module(
        mod,
        input_shape,
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        ),
    )



    
class UNetWrapper(torch.nn.Module):
    def __init__(self, unet) -> None:
        super().__init__()
        self.unet = unet

    def forward(
        self,
        text_embeddings,
        uncond_embeddings,
        latents,
        t,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        batch_size = 1,
        guidance_scale: Optional[float] = 7.5,
    ):
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        noise_pred = self.unet(latent_model_input, timestep=t, encoder_hidden_states=text_embeddings)["sample"]

        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred, latents, t



class UnetConvInWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample,):
        sample = self.unet.conv_in(sample)

        return sample

def test_unet_conv_in(test_device):
    pytest.skip()
    pipe = download_model(StableDiffusionPipeline.from_pretrained, "CompVis/stable-diffusion-v1-4")

    mod = PyTorchModule("sd_unet_convin", UnetConvInWrapper(pipe.unet))

    input_shape = ((2, 4, 64, 64), )
    verify_module(
        mod,
        input_shape,
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        ),
    )

class UnetMidBlockWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, emb, hidden_states):
        sample = self.unet.mid_block(sample, emb, encoder_hidden_states=hidden_states)

        return sample

def test_unet_mid_block(test_device):
    pytest.skip()
    pipe = download_model(StableDiffusionPipeline.from_pretrained, "CompVis/stable-diffusion-v1-4")

    mod = PyTorchModule("sd_unet_mid_block", UnetMidBlockWrapper(pipe.unet))

    input_shape = ((2, 1280, 8, 8), (2, 1280), (2, 77, 768))
    verify_module(
        mod,
        input_shape,
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        ),
    )



class UnetUpBlockWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet 

    def forward(self, sample, emb, res1, res2, res3, res4, res5, res6, encoder_hidden_states):
        sample = sample.reshape(-1, 1280, 8, 8)
        res1 = res1.reshape(-1, 1280, 8, 8)
        res2 = res2.reshape(-1, 1280, 8, 8)
        res3 = res3.reshape(-1, 1280, 8, 8)

        res4 = res4.reshape(-1, 640, 16, 16)
        res5 = res5.reshape(-1, 1280, 16, 16)
        res6 = res6.reshape(-1, 1280, 16, 16)
        sample = self.unet.up_blocks[0](sample, temb=emb, res_hidden_states_tuple=(res1, res2, res3))
        sample = self.unet.up_blocks[1](sample, temb=emb, res_hidden_states_tuple=(res4, res5, res6), encoder_hidden_states=encoder_hidden_states)

        return sample


def test_unet_up_block(test_device):
    import os
    os.environ["PYBUDA_PAD_SPARSE_MM_WEIGHT_MM"] = "{10:12, 20:24, 40:48}"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_MAX_GRAPH_CUT_RETRY"] = "100"
    os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{64*1024}"
    pipe = download_model(StableDiffusionPipeline.from_pretrained, "CompVis/stable-diffusion-v1-4")

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.retain_tvm_python_files = True
    compiler_cfg.enable_enumerate_u_kt = False
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.graph_solver_self_cut_type = "ConsumerOperandDataEdgesFirst"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    input_shape = ((1, 1, 1280, 64), (1, 1280), (1, 1, 1280, 64), (1, 1, 1280, 64), (1, 1, 1280, 64),
                    (1, 1, 640, 256),(1, 1, 1280, 256),(1, 1, 1280, 256),(1, 77, 768))
    mod = PyTorchModule("sd_unet_up_block", UnetUpBlockWrapper(pipe.unet))

    verify_module(
        mod,
        input_shape,
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        ),
    )


def test_unet_section(test_device):
    pytest.skip()
    class Test(PyBudaModule):
        def __init__(self):
            super().__init__("unet_section")
            self.add_constant("const_760")
            self.set_constant("const_760", torch.ones(1, 32, 1, 1))
            
            self.add_constant("const_761")
            self.set_constant("const_761", torch.ones(1, 32, 10, 4096))
            
            self.add_constant("const_762")
            self.set_constant("const_762", torch.ones(1, 32, 10, 4096))
            
        def forward(self, x):
            softmax_210 = pybuda.op.Softmax("", x, dim=-1) # (1, 1, 320, 4096)
            reshape_211 = pybuda.op.Reshape("", softmax_210, shape=(1, 32, 10, 4096))
            reduce_avg_212 = pybuda.op.ReduceAvg("", reshape_211, dim=-2) # (1, 32, 1, 4096)
            reduce_avg_213 = pybuda.op.ReduceAvg("", reduce_avg_212, dim=-1) # (1, 32, 1, 1)
            subtract_214 = pybuda.op.Subtract("", reshape_211, reduce_avg_213) # (1, 32, 10, 4096)
            subtract_215 = pybuda.op.Subtract("", reshape_211, reduce_avg_213) # (1, 32, 10, 4096)
            multiply_216 = pybuda.op.Multiply("", subtract_215, subtract_215) # (1, 32, 10, 4096)
            reduce_avg_217 = pybuda.op.ReduceAvg("", multiply_216, dim=-2) # (1, 32, 1, 4096)
            reduce_avg_218 = pybuda.op.ReduceAvg("", reduce_avg_217, dim=-1) # (1, 32, 1, 1)
            add_220 = pybuda.op.Add("", reduce_avg_218, self.get_constant("const_760")) # (1, 32, 1, 1)
            sqrt_221 = pybuda.op.Sqrt("", add_220) # (1, 32, 1, 1)
            reciprocal_222 = pybuda.op.Reciprocal("", sqrt_221) # (1, 32, 1, 1)
            multiply_223 = pybuda.op.Multiply("", subtract_214, reciprocal_222) # (1, 32, 10, 4096)
            multiply_226 = pybuda.op.Multiply("", multiply_223, self.get_constant("const_761")) # (1, 32, 10, 4096)
            add_228 = pybuda.op.Add("", multiply_226, self.get_constant("const_762")) # (1, 32, 10, 4096)
            sigmoid_229 = pybuda.op.Sigmoid("", add_228) # (1, 32, 10, 4096)
            multiply_230 = pybuda.op.Multiply("", add_228, sigmoid_229) # (1, 32, 10, 4096)
            reshape_231 = pybuda.op.Reshape("", multiply_230, shape=(1, 1, 320, 4096))
            softmax_232 = pybuda.op.Softmax("", reshape_231, dim=-1) # (1, 1, 320, 4096)
            return softmax_232


    input_shape = ((1, 1, 320, 4096),)
    verify_module(
        Test(),
        input_shape,
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        ),
    )
    





class UnetAttnDownBlockWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet 

    def forward(self, sample, emb, encoder_hidden_states):
        sample = sample.reshape(-1, 320, 64, 64)
        sample, _ = self.unet.down_blocks[0](sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
        sample, _ = self.unet.down_blocks[1](sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
        # sample, _ = self.unet.down_blocks[2](sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
        # sample, _ = self.unet.down_blocks[3](sample, temb=emb,)

        return sample

def test_unet_down_block(test_device):

    import os
    os.environ["PYBUDA_PAD_SPARSE_MM_WEIGHT_MM"] = "{10:12, 20:24, 40:48}"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["PYBUDA_MAX_GRAPH_CUT_RETRY"] = "100"
    os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{80*1024}"
    pipe = download_model(StableDiffusionPipeline.from_pretrained, "CompVis/stable-diffusion-v1-4")

    input_shape = ((1, 1, 320, 4096), (1, 1280), (1, 77, 768))
    mod = PyTorchModule("sd_unet_downblock", UnetAttnDownBlockWrapper(pipe.unet))

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["PYBUDA_RIBBON_LEGACY"] = "1"
    compiler_cfg.graph_solver_self_cut_type = "ConsumerOperandDataEdgesFirst"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    verify_module(
        mod,
        input_shape,
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        ),
    )

def stable_diffusion_preprocessing(
    pipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback = None,
    callback_steps: int = 1,
    cross_attention_kwargs = None,
):

    # 0. Default height and width to unet
    height = height or pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = width or pipeline.unet.config.sample_size * pipeline.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    pipeline.check_inputs(
        prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
    )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipeline._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds = pipeline._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    # 4. Prepare timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = pipeline.unet.in_channels
    latents = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)

    return latents, timesteps, extra_step_kwargs, prompt_embeds, extra_step_kwargs


class UnetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, latent_model_input, timestep, text_embeddings):
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings)["sample"]
        return noise_pred

def initialize_compiler_overrides():
    # os.environ["PYBUDA_RELOAD_GENERATED_MODULES"] = "1"
    os.environ["PYBUDA_MAX_GRAPH_CUT_RETRY"] = "2000"

    os.environ["PYBUDA_PAD_SPARSE_MM_WEIGHT_MM"] = "{10:12, 20:24, 30:32, 40:48, 60:64}"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.enable_enumerate_u_kt = False
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.retain_tvm_python_files = True
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.balancer_policy = "Ribbon"

    compiler_cfg.place_on_new_epoch("matmul_134")
    compiler_cfg.place_on_new_epoch("matmul_268")
    compiler_cfg.place_on_new_epoch("matmul_2159")
    compiler_cfg.place_on_new_epoch("matmul_2295")
    compiler_cfg.place_on_new_epoch("matmul_2431")


def denoising_loop(
    pipeline,
    latents,
    timesteps,
    prompt_embeds,
    extra_step_kwargs,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    callback = None,
    callback_steps: int = 1,
    ttdevice = None,
):
    assert ttdevice is not None, "Please provide a TTDevice"


    do_classifier_free_guidance = guidance_scale > 1.0
    num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
                timestep_ = torch.cat([t.unsqueeze(0)] * 2).float()
            else:
                latent_model_input = latents
                timestep_ = t

            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            ttdevice.push_to_inputs(*[latent_model_input.detach()[0:1], timestep_.detach()[0:1], prompt_embeds.detach()[0:1]])
            output_q = pybuda.run_inference(_sequential=True)
            noise_pred_0 = output_q.get()[0].value().detach()

            ttdevice.push_to_inputs(*[latent_model_input.detach()[1:2], timestep_.detach()[1:2], prompt_embeds.detach()[1:2]])
            output_q = pybuda.run_inference(_sequential=True)
            noise_pred_1 = output_q.get()[0].value().detach()

            noise_pred = torch.cat([noise_pred_0, noise_pred_1], dim=0)

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)


    return latents


def stable_diffusion_postprocessing(
    pipeline,
    latents,
    prompt_embeds,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
):
    
    device = pipeline._execution_device
    has_nsfw_concept = None
    if output_type == "latent":
        image = latents
        has_nsfw_concept = None
    elif output_type == "pil":
        # 8. Post-processing
        latents = 1 / pipeline.vae.config.scaling_factor * latents
        image = pipeline.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()

        # 9. Run safety checker
        # image, has_nsfw_concept = pipeline.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        image = pipeline.numpy_to_pil(image)
    else:
        # 8. Post-processing
        latents = 1 / pipeline.vae.config.scaling_factor * latents
        image = pipeline.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()

        # 9. Run safety checker
        # image, has_nsfw_concept = pipeline.run_safety_checker(image, device, prompt_embeds.dtype)

    # Offload last model to CPU
    if hasattr(pipeline, "final_offload_hook") and pipeline.final_offload_hook is not None:
        pipeline.final_offload_hook.offload()

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)



def test_stable_diffusion_pipeline():

    # Step 0: Load model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")


    # Step 1: Compile/Load PyBuda module
    save_path = "sd_unet_final.tti"

    if not os.path.exists(save_path):
        # Need to compile to tti
        initialize_compiler_overrides()
        input_shape = ((1, 4, 64, 64), (1,), (1, 77, 768))
        tt_module = pybuda.PyTorchModule("sd_unet_script", UnetWrapper(pipe.unet),)
        device = pybuda.TTDevice("tt0", module=tt_module,arch=BackendDevice.Wormhole_B0, devtype=BackendType.Silicon)
        tti_img = device.compile_to_image(
            img_path="sd_unet_final.tti",
            training=False,
            sample_inputs=[torch.randn(shape) for shape in input_shape],
        )

    if os.path.exists(save_path):
        device_img: TTDeviceImage = TTDeviceImage.load_from_disk(save_path)
        ttdevice = pybuda.TTDevice.load_image(img=device_img)
    else:
        raise RuntimeError("Could not load compiled TTI file")

    # Step 2: Run inference
    while True:
        prompt = input("Please enter a prompt for image generation: ")
        if prompt == "exit":
            break
        
        print("Generating image for prompt: ", prompt)
        num_inference_steps = 50

        
        latents, timesteps, extra_step_kwargs, prompt_embeds, extra_step_kwargs = stable_diffusion_preprocessing(
                                                                            pipe,
                                                                            prompt,
                                                                            num_inference_steps=num_inference_steps,
                                                                        )
        
        latents = denoising_loop(
            pipe,
            latents,
            timesteps,
            prompt_embeds,
            extra_step_kwargs,
            num_inference_steps=num_inference_steps,
            ttdevice=ttdevice,
        )

        output = stable_diffusion_postprocessing(
            pipe,
            latents,
            prompt_embeds,
        )
        output.images[0].save(prompt.replace(" ", "_") + ".png")

