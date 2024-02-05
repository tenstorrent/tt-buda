# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda._C import MathFidelity
import pytest
from typing import List, Optional, Union

import torch
from torch import nn
from loguru import logger
import os
import argparse


from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

import pybuda
from pybuda import (
    TTDeviceImage,
    BackendType,
    BackendDevice,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from PIL import Image




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
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"
    os.environ["PYBUDA_RIBBON2"] = "1"

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.enable_enumerate_u_kt = False
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.retain_tvm_python_files = True
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.balancer_policy = "Ribbon"



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



def run_stable_diffusion_pipeline():
    parser = argparse.ArgumentParser()
    # General run functionality
    parser.add_argument("--compile", help="Compile stable diffusion unet into a TTI", action="store_true")
    parser.add_argument("--run", help="Run stable diffusion pipeline", action="store_true")
    parser.add_argument("-n", "--num_inference_steps", help="Number of denoising loop iterations", default=50)
    args = parser.parse_args()
    num_inference_steps = args.num_inference_steps

    if not args.compile and not args.run:
        raise RuntimeError(f"Please specify mode of operation: use \"--compile\" or \"--run\"")
    

    # Step 0: Load model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

    if args.compile:
        logger.info(f"Compiling stable diffusion unet into TTI")
        initialize_compiler_overrides()
        input_shape = ((1, 4, 64, 64), (1,), (1, 77, 768))
        tt_module = pybuda.PyTorchModule("sd_unet_demo", UnetWrapper(pipe.unet),)
        device = pybuda.TTDevice("tt0", module=tt_module,arch=BackendDevice.Wormhole_B0, devtype=BackendType.Silicon)
        tti_img = device.compile_to_image(
            img_path="sd_unet_demo.tti",
            training=False,
            sample_inputs=[torch.randn(shape) for shape in input_shape],
        )

        return


    if args.run:
        save_path = "sd_unet_demo.tti"
        device_img: TTDeviceImage = TTDeviceImage.load_from_disk(save_path)
        ttdevice = pybuda.TTDevice.load_image(img=device_img)


        while True:
            prompt = input("Please enter a prompt for image generation: ")
            if prompt == "exit":
                break
            
            print("Generating image for prompt: ", prompt)

            
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

        return 



if __name__ == "__main__":
    run_stable_diffusion_pipeline()

