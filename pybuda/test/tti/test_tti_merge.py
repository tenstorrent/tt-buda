# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import subprocess as sp
import argparse
import os
from loguru import logger
from pybuda.tools.tti_merge import merge_models

if __name__ == "__main__":
    try:
        # Remove this config var, else dir names are too long in CI
        if "PYBUDA_CI_DIR" in os.environ:
            print("Found var")
            del os.environ["PYBUDA_CI_DIR"]
            
        parser =  argparse.ArgumentParser()
        parser.add_argument("--models_to_merge", type = str, help = "List of models to generate TTIs for and merge.", default = "*", nargs = "*")
        parser.add_argument("--device_cfg", type = str, help = "Choose between wh_n150 and wh_n300", required = True)
        parser.add_argument("--merge_ttis", type = bool, help = "Merge Generated TTIs into a single image", default = False)
        parser.add_argument("--disable_host_queues", type = bool, help = "Disable activations in host memory", default = False)
        parser.add_argument("--disable_memory_optimizations", type = bool, help = "Disable low level memory optimizations done during model fusion", default = False)
        args = parser.parse_args()
        
        tti_build_dir = "device_images_to_merge/"
        valid_models = {
            "resnet" : os.path.abspath("pybuda/test/benchmark/benchmark.py") + " -m resnet -c resnet50 -df Fp16_b -mf HiFi3 -o perf.json -mb 128 --save_tti " + tti_build_dir + "/resnet.tti --arch wormhole_b0 --device-config ",
            "mobilenet_v1" : os.path.abspath("pybuda/test/benchmark/benchmark.py") + " -m mobilenet_v1 -c 224 -df Fp16_b -mf HiFi2 -o perf.json -mb 256 --save_tti " + tti_build_dir + "/mobilenet_v1.tti --arch wormhole_b0 --device-config ",
            "mobilenet_v2" : "pybuda/test/benchmark/benchmark.py -m mobilenet_v2 -c 224 -df Fp16_b -mf HiFi2 -o perf.json -mb 256 --save_tti " + tti_build_dir + "/mobilenet_v2.tti --arch wormhole_b0 --device-config ",
            "mobilenet_v3" : "pybuda/test/benchmark/benchmark.py -m mobilenet_v3_timm -c large -df Fp16_b -mf HiFi2 -o perf.json -mb 256 --save_tti " + tti_build_dir + "/mobilenet_v3.tti --arch wormhole_b0 --device-config ",
            "hrnet" : "pybuda/test/benchmark/benchmark.py -m hrnet -c v2_w64 -df Fp16_b -mf HiFi3 -o perf.json -mb 128 --save_tti " + tti_build_dir + "/hrnet.tti --arch wormhole_b0 --device-config ",
            "vit" : "pybuda/test/benchmark/benchmark.py -m vit -c base -df Fp16_b -mf HiFi2 -o perf.json -mb 64 --save_tti " + tti_build_dir + "/vit.tti --arch wormhole_b0 --device-config ",
            "deit" : "pybuda/test/benchmark/benchmark.py -m deit -c base -df Fp16_b -mf HiFi2 -o perf.json -mb 128 --save_tti " + tti_build_dir + "/deit.tti --arch wormhole_b0 --device-config ",
            "unet" : "pybuda/test/benchmark/benchmark.py -m unet -c 256 -mb 48 -df Fp16_b -mf HiFi3 -o perf.json -mb 16 --save_tti " + tti_build_dir + "/unet.tti --arch wormhole_b0 --device-config ",
            "inception" : "pybuda/test/benchmark/benchmark.py -m inception_v4 -c 224 -df Fp16_b -mf HiFi3 -o perf.json -mb 32 --save_tti " + tti_build_dir + "/inception.tti --arch wormhole_b0 --device-config ",
            "bert_large" : "pybuda/test/benchmark/benchmark.py -m bert -c large_tc -df Fp16_b -mf HiFi3 -o perf.json -mb 64 --save_tti " + tti_build_dir + "/bert_large.tti --arch wormhole_b0 --device-config ",
        }
        
        os.makedirs(tti_build_dir, exist_ok=True)
        harvesting_flag = ['--env']
        if args.device_cfg == "wh_n150":
            harvesting_flag = harvesting_flag + ['PYBUDA_FORCE_EMULATE_HARVESTED=1 TT_BACKEND_HARVESTED_ROWS=2048']
        elif args.device_cfg == "wh_n300":
            harvesting_flag = harvesting_flag + ['PYBUDA_FORCE_EMULATE_HARVESTED=1 TT_BACKEND_HARVESTED_ROWS=2050']
        else:
            logger.exception("Unsupported device cfg: {}", args.device_cfg)
        
        # # Generate TTIs
        tti_locations = []
        if args.disable_host_queues:
            os.environ["PYBUDA_ENABLE_INPUT_QUEUES_ON_HOST"] = "0"
            os.environ["PYBUDA_ENABLE_OUTPUT_QUEUES_ON_HOST"] = "0"
        os.environ["PYBUDA_TTI_BACKEND_FORMAT"] = "1"

        models_to_merge = []
        if args.models_to_merge == "*":
            models_to_merge = valid_models
        else:
            for model in args.models_to_merge:
                assert model in valid_models, "Model: " + model + "is not in the list of valid models."
                models_to_merge.append(model)
        # Generate TTIs
        for model in models_to_merge:
            assert model in valid_models, "Model: " + model + "is not in the list of valid models."
            logger.info("Generating TTI for {}", model)
            cmd = "python3 " + valid_models[model] + args.device_cfg
            cmd = cmd.split(" ")
            cmd = cmd + harvesting_flag
            sp.run(cmd)
            tti_locations.append(os.path.abspath(os.path.join(tti_build_dir, model + ".tti")))
        # Merge TTIs
        logger.info("Merging TTIs")
        if args.disable_memory_optimizations:
            merge_models(tti_locations, "wormhole_b0", "merged_model.tti", False, False)
        else:
            merge_models(tti_locations, "wormhole_b0", "merged_model.tti")
    except Exception as e:
        logger.exception(e)
        