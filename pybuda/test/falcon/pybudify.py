# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import sys
import os
import torch


class PyBudify(torch.nn.Module):
    def __init__(self, pt_module, device='silicon', arch='wormhole_b0', precision='fp32', amp_level=0, micro_batch_size=1, fuse=False, num_chips=1, perf=None,
                 verify=False, log_level='ERROR', tti_save=None, tti_load=None, concurrent=False, place='none', fracture=False, fracture_attn=0, fracture_mlp=0,
                 netlist_name="pybudify_module", padded_fracture=False, padded_fracture_p=False, padded_fracture_full=False, lora=False, host_queues=False,
                 odkv=False, masked_odkv=False, num_layers=32, matmuls='bfp8', decode_mode=True):
        super().__init__()

        self.device = device
        self.bound_module = pt_module
        self.tti_save = tti_save
        self.tti_load = tti_load
        self.concurrent = concurrent
        self.odkv = odkv
        self.masked_odkv = masked_odkv

        if device != 'pytorch':
            os.environ["LOGGER_LEVEL"] = log_level
            os.environ["LOGURU_LEVEL"] = log_level

            # pybuda workarounds
            os.environ["GOLDEN_WORMHOLE_B0"] = "1"            # golden should always simulate a B0 as that's all we use now
            os.environ["PYBUDA_ENABLE_STABLE_SOFTMAX"] = "1"  # improved accuracy - pybuda team surprised we need it though
            os.environ["PYBUDA_CONVERT_PARAMS_TO_TVM"] = "0"  # faster compile times... why would this ever be 1?
            os.environ["TT_BACKEND_TIMEOUT"] = "0"            # default is too aggressive for large models?

            # os.environ["ENABLE_ETH_SERIALIZATON"] = "1"
            # os.environ["PYBUDA_ENABLE_BROADCAST_SPLITTING"] = "1"
            # os.environ["PYBUDA_DISABLE_FORK_JOIN_BUF"] = "1"
            # os.environ["PYBUDA_DRAM_PICK_CAPACITY"] = "1"
            # os.environ["WHA0_DISABLE_RELAY_BUFS"] = "1"
            # os.environ["PYBUDA_FUSE_STOP_ON_RECIPROCAL"] = "1"
            # os.environ["PYBUDA_PLACER_SNAKE"] = "1" Not what we want for dual chip placement
            # os.environ["PYBUDA_DISABLE_INTERACTIVE_PLACER"] = "1" # Until interactive placer supports multi-chip placement overrides
            # os.environ["PYBUDA_PLACER_SNAKE"] = "1"
            # os.environ["PYBUDA_ETH_LINKS_NEBULA"] = "1"

            pybuda = self.pybuda = __import__('pybuda') # let us set log levels before importing pybuda

        if device == 'pytorch':
            pass
        else:
            devtype = { 'golden' : pybuda.BackendType.Golden,
                        'silicon': pybuda.BackendType.Silicon,
                      }[device]

            module = pybuda.PyTorchModule(netlist_name, self.bound_module)

            if precision == 'fp32':
                fallback = pybuda.DataFormat.Float32
            elif precision == 'fp16':
                fallback = pybuda.DataFormat.Float16
            elif precision == 'bf16':
                fallback = pybuda.DataFormat.Float16_b
            elif precision == 'fp8':
                fallback = pybuda.DataFormat.Bfp8
            elif precision == 'fp8b':
                fallback = pybuda.DataFormat.Bfp8_b
            else:
                raise ValueError('Precision "%s" not implemented' % precision)

            if matmuls == 'bfp8':
                # Lower-precision: All matmuls inputs lowered to bfp8
                pybuda.config.configure_mixed_precision(
                    name_regex="matmul_.*",
                    input_df={0: [pybuda.DataFormat.Bfp8_b, True], 1: [pybuda.DataFormat.Bfp8_b, True], 2: [pybuda.DataFormat.Bfp8_b, True]})
            elif matmuls == '2nd_mlp_bfp8':
                # Higher-precision: 2nd matmul in MLP lower weights to bfp8
                # Layer 0 has different offset
                mm_offset = 80
                pybuda.config.configure_mixed_precision(
                    name_regex=f"matmul_{23}",
                    input_df={0: [pybuda.DataFormat.Bfp8_b, False], 1: [pybuda.DataFormat.Bfp8_b, False], 2: [pybuda.DataFormat.Bfp8_b, False]})
                for i in range(31):
                    pybuda.config.configure_mixed_precision(
                        name_regex=f"matmul_{104+i*mm_offset}",
                        input_df={0: [pybuda.DataFormat.Bfp8_b, False], 1: [pybuda.DataFormat.Bfp8_b, False], 2: [pybuda.DataFormat.Bfp8_b, False]})
            elif matmuls == 'weight_bfp8_act_bf16':
                # Experiment 4 - Weights-bfp8 [Current best alpaca-eval score for finetune decode demo]: All matmul weights BPF8. All activations BFP16
                pybuda.config.configure_mixed_precision(
                    name_regex="matmul_.*",
                    input_df={0: [pybuda.DataFormat.Bfp8_b, False], 1: [pybuda.DataFormat.Bfp8_b, False], 2: [pybuda.DataFormat.Bfp8_b, False]})
            else:
                raise ValueError('Matmul precision "%s" not implemented' % matmuls)

            if decode_mode:
                # Required for decode or we get invalid DF error. Important: DO not set intermed, acc_df or we hang on prefill.
                pybuda.config.configure_mixed_precision(
                    op_type="splice",
                    output_df=pybuda.DataFormat.Float16_b,
                    input_df={0: [pybuda.DataFormat.Float16_b, True], 1: [pybuda.DataFormat.Float16_b, True], 2: [pybuda.DataFormat.Float16_b, True]})

            if lora:
                os.environ['TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE'] = "147456"

            if fracture:
                # Inner-dimension fracturing for Q proj and FF2
                os.environ['TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE'] = "147456" #"98304"#"49152"

                q_offset = 70
                h4h_offset = 70
                for layer_num in range(len(self.bound_module.layers)):
                    pybuda.config.insert_fracture_group([(f"matmul_{26+layer_num*q_offset}", pybuda.k_dim, 2)])
                    pybuda.config.insert_fracture_group([(f"matmul_{23+layer_num*h4h_offset}", pybuda.k_dim, 4)])

            if padded_fracture:
                offset = 73
                factor = 2
                for layer_num in range(len(self.bound_module.layers)):
                    pybuda.config.insert_fracture_group([(f"matmul_{18+layer_num*offset}", -1, factor), (f"matmul_{23+layer_num*offset}", pybuda.k_dim, factor)])

                    if layer_num > 0 and layer_num < len(self.bound_module.layers)-1:
                        pybuda.set_epoch_break(f'multiply_{0+layer_num*offset}')

                # 4 bit precision for fracturing required otherwise DRAM error occurs for 32 layers
                pybuda.config.configure_mixed_precision(
                    name_regex="fractured_1_matmul_.*",
                    input_df={0: [pybuda.DataFormat.Bfp8_b, True], 1: [pybuda.DataFormat.Bfp4_b, True], 2: [pybuda.DataFormat.Bfp8_b, True]})
                pybuda.config.configure_mixed_precision(
                    name_regex="fractured_0_matmul_.*",
                    input_df={0: [pybuda.DataFormat.Bfp8_b, True], 1: [pybuda.DataFormat.Bfp4_b, True], 2: [pybuda.DataFormat.Bfp8_b, True]})

            if padded_fracture_p:
                offset = 73
                for layer_num in range(len(self.bound_module.layers)):
                    if layer_num < len(self.bound_module.layers)//2:
                        pybuda.config.insert_fracture_group([(f"matmul_{18+layer_num*offset}", -1, 2), (f"matmul_{23+layer_num*offset}", pybuda.k_dim, 2)])

                    if layer_num > 0 and layer_num < len(self.bound_module.layers)-1:
                        pybuda.set_epoch_break(f'multiply_{0+layer_num*offset}')

            # Running padded fracture full (Dragon's exploration)
            if padded_fracture_full:

                # 4 bit precision for fracturing required otherwise DRAM error occurs for 32 layers
                # TODO change this to only affect MLP fractured matmuls and not attention
                pybuda.config.configure_mixed_precision(
                    name_regex="fractured_*._matmul_.*",
                    input_df={0: [pybuda.DataFormat.Bfp8_b, True], 1: [pybuda.DataFormat.Bfp4_b, True], 2: [pybuda.DataFormat.Bfp8_b, True]})

                # Attn fracture
                if fracture_attn > 0:
                    attn_constr = []
                    attn_factor = fracture_attn
                    q_offset = 73

                    for layer_num in range(len(self.bound_module.layers)):
                        # Since we move around the users dimension, full attn fracturing won't be possible in a single group
                        pybuda.config.insert_fracture_group([
                            # Q
                            # (f"matmul_{26+layer_num*q_offset}", -2, attn_factor),
                            # K
                            # (f"matmul_{39+layer_num*q_offset}", -2, attn_factor),
                            #V
                            # (f"matmul_{64+layer_num*q_offset}", -2, attn_factor),
                            # QK
                            # (f"matmul_{52+layer_num*q_offset}", -3, attn_factor),
                            (f"matmul_{51+layer_num*q_offset}", -3, attn_factor),
                            # QK.V
                            # (f"matmul_{70+layer_num*q_offset}", -3, attn_factor),
                            (f"matmul_{65+layer_num*q_offset}", -3, attn_factor),
                            # K cache
                            # (f"past_key", -3, attn_factor),
                            (f"concatenate_{48+layer_num*q_offset}", -3, attn_factor),
                            # V cache
                            # (f"past_value", -3, attn_factor),
                            (f"concatenate_{61+layer_num*q_offset}", -3, attn_factor),
                            #
                            # fracturing this matmul leads to unspported fracture reshape_66 op
                            # (f"matmul_{68+layer_num*q_offset}", -2, attn_factor),
                        ])
                        entries = [f'multiply_{15 + layer_num*q_offset}']
                        # ops = [[f'concatenate_{48 + layer_num*q_offset}',
                        ops = [[f'matmul_{51 + layer_num*q_offset}',
                                # f'concatenate_{61 + layer_num*q_offset}',
                                f'matmul_{65 + layer_num*q_offset}',
                                # f'matmul_{68 + layer_num*q_offset}',
                                ]]
                        exits = [f'matmul_{68 + layer_num*q_offset}']
                        # exits = [f'add_{70 + layer_num*q_offset}']
                        attn_constr = self.add_sched(pybuda, entries, exits, ops, attn_factor, attn_constr)
                    pybuda.config.add_schedule_constraint(attn_constr)

                # MLP fracture
                if fracture_mlp > 0:
                    mlp_constr = []
                    mlp_factor = fracture_mlp
                    h4h_offset = 73
                    for layer_num in range(len(self.bound_module.layers)):
                        mlp_offset = layer_num * h4h_offset
                        print(f"MLP Fracture: Layer: {layer_num}, matmul offset = {18+mlp_offset} & {23+mlp_offset}")

                        # Manual scheduling to support MLP fracture 2-chip on full size falcon-7B
                        pybuda.set_epoch_break(f'softmax_{55+mlp_offset}.dc.reduce_max.0')
                        pybuda.config.override_op_placement(f"concatenate_{48+mlp_offset}.dc.concatenate.2", chip_id=1, temporal_epoch_break=True)

                        # MLP fracture
                        pybuda.config.insert_fracture_group([
                            # Can't do fracturing of weights due to transpose
                            # mlp.dense_h_to_4h
                            (f"matmul_{18+mlp_offset}", -1, mlp_factor),
                            # mlp.dense_4h_to_h
                            (f"matmul_{23+mlp_offset}", pybuda.k_dim, mlp_factor),
                            ])

                        entries = [f'multiply_{15 + mlp_offset}']
                        ops = [[f'matmul_{18 + mlp_offset}',
                                f'matmul_{23 + mlp_offset}']]
                        exits = [f'add_{70 + mlp_offset}']
                        mlp_constr = self.add_sched(pybuda, entries, exits, ops, mlp_factor, mlp_constr)
                    pybuda.config.add_schedule_constraint(mlp_constr)


            perf_level = { None    : None,
                          'none'   : None,
                          'light'  : pybuda.PerfTraceLevel.LIGHT,
                          'verbose': pybuda.PerfTraceLevel.VERBOSE }[perf]
            pybuda.set_configuration_options(default_df_override=fallback,
                                             accumulate_df=pybuda.DataFormat.Float32,
                                             amp_level=amp_level,
                                             enable_auto_fusing=fuse,
                                             performance_trace=perf_level,
                                             backend_opt_level=4,
                                             enable_auto_transposing_placement=True
                                             )
            compiler_cfg = pybuda.config._get_global_compiler_config()
            compiler_cfg.input_queues_on_host = host_queues

            if self.masked_odkv:
                # pybuda.config.override_t_stream_dir(f"concatenate_50.dc.sparse_matmul.4.lc2", "c")
                # pybuda.config.override_t_stream_dir(f"concatenate_67.dc.sparse_matmul.4.lc2", "c")

                # pybuda.config.set_epoch_break("transpose_58.dc.sparse_matmul.4.lc2")
                # pybuda.config.set_epoch_break("matmul_64")

                # pybuda.config.add_schedule_constraint(['transpose_58.dc.sparse_matmul.4.lc2', 'add_59'])
                names = []
                if num_layers == 1:
                    names_start_idx = 56
                    names_end_idx = 57
                elif num_layers == 16:
                    names_start_idx = 716
                    names_end_idx = 747
                elif num_layers == 24:
                    names_start_idx = 1068
                    names_end_idx = 1115
                elif num_layers == 32:
                    names_start_idx = 1420 #1418
                    names_end_idx = 1483 # 1481
                else:
                    raise Exception("Unsupported num_layers. Please use either 1,16 or 32.")

                for i in range (names_end_idx-names_start_idx+1):
                    names.append('input__'+str(names_start_idx+i))
                print(f'names" {names}')

                names_dict = { name: (i+1) for i, name in enumerate(names) }

                compiler_cfg = pybuda.config._get_global_compiler_config()
                compiler_cfg.loopback_outputs = names_dict

                # pybuda.config.insert_fracture_group([(f"concatenate_50", 2, 2)])
                # pybuda.config.insert_fracture_group([(f"concatenate_67", 2, 2)])

                # pybuda.config.configure_mixed_precision(
                #     name_regex="concatenate_50.dc.sparse_matmul.4.lc2",
                #     input_df={0: [pybuda.DataFormat.Bfp8_b, True], 1: [pybuda.DataFormat.Bfp8_b, True], 2: [pybuda.DataFormat.Bfp8_b, True]})

                # pybuda.config.configure_mixed_precision(
                #     name_regex="concatenate_50.dc.sparse_matmul.4.lc2",
                #     input_df={0: [pybuda.DataFormat.Bfp8_b, True], 1: [pybuda.DataFormat.Bfp8_b, True], 2: [pybuda.DataFormat.Bfp8_b, True]})
            elif self.odkv:
                # pybuda.config.override_t_stream_dir(f"concatenate_50.dc.sparse_matmul.4.lc2", "c")
                # pybuda.config.override_t_stream_dir(f"concatenate_67.dc.sparse_matmul.4.lc2", "c")
                names = []
                if num_layers == 1:
                    names_start_idx = 54
                    names_end_idx = 55
                elif num_layers == 32:
                    names_start_idx = 1418
                    names_end_idx = 1481
                else:
                    raise Exception("Unsupported num_layers. Please use either 1 or 32.")

                for i in range (names_end_idx-names_start_idx+1):
                    names.append('input__'+str(names_start_idx+i))
                print(f'names" {names}')
                names_dict = { name: (i+1) for i, name in enumerate(names) }

                compiler_cfg = pybuda.config._get_global_compiler_config()

                # pybuda.config.insert_fracture_group([(f"concatenate_50", 2, 2)])
                # pybuda.config.insert_fracture_group([(f"concatenate_67", 2, 2)])

                # pybuda.config.configure_mixed_precision(
                #     name_regex="concatenate_50.dc.sparse_matmul.4.lc2",
                #     input_df={0: [pybuda.DataFormat.Bfp8_b, True], 1: [pybuda.DataFormat.Bfp8_b, True], 2: [pybuda.DataFormat.Bfp8_b, True]})

                # pybuda.config.configure_mixed_precision(
                #     name_regex="concatenate_50.dc.sparse_matmul.4.lc2",
                #     input_df={0: [pybuda.DataFormat.Bfp8_b, True], 1: [pybuda.DataFormat.Bfp8_b, True], 2: [pybuda.DataFormat.Bfp8_b, True]})

                compiler_cfg.loopback_outputs = names_dict

            pybuda_arch = { 'grayskull': pybuda.BackendDevice.Grayskull,
                            'wormhole_b0': pybuda.BackendDevice.Wormhole_B0 }[arch]

            if tti_load is not None:
                self.tt0 = pybuda.TTDevice.load_image(img_path=tti_load)
            else:
                self.tt0 = pybuda.TTDevice('tt0', module=module,
                                            fp32_fallback=fallback,
                                            arch=pybuda_arch,
                                            devtype=devtype,
                                            chip_ids=list(range(num_chips)))

            mp = torch.multiprocessing.get_context('spawn')
            self.output_q = mp.Queue()

            if verify:
                self.verify_cfg = pybuda.VerifyConfig(verify_all=True,
                                                      verify_last=True,
                                                      devtype=pybuda.BackendType.Silicon,
                                                      arch=pybuda_arch,)
            else:
                self.verify_cfg = None

            self.initialized = False
            self.micro_batch_size = micro_batch_size


    def run_async(self, *args):
        """ Send inputs to pybuda and run forward pass asynchronously.
            Outputs can be read from self.output_q. """
        assert self.device != 'pytorch', "run_async() is only supported for pybuda devices"
        if self.odkv or self.masked_odkv:
            self.ensure_initialized(*args)
            # print(f'pybuda pushing data')
            self.pybuda.sync()
            in_args = list(args[0]) + list(args[1]) + list(args[2]) + list(args[3])
            self.tt0.push_to_inputs(in_args)   # don't pass in kv over and over again
            self.pybuda.run_generate(input_count=1, write_index=0) #, _sequential=True)
        else:
            self.ensure_initialized(*args)
            self.tt0.push_to_inputs(*args)
            self.pybuda.run_forward(input_count=1)


    def ensure_initialized(self, *args):
        if not self.initialized and self.device != 'pytorch':
            if self.tti_save is not None:
                self.tt0.compile_to_image(
                    img_path=self.tti_save,
                    training=False,
                    sample_inputs=args,
                    microbatch_count=self.micro_batch_size,
                )
                print(f'Saved image to {self.tti_save}')
                sys.exit(0)
            self.pybuda.initialize_pipeline(training=False,
                                    sample_inputs=args,
                                    output_queue=self.output_q,
                                    microbatch_count=self.micro_batch_size,
                                    _sequential=self.concurrent==False,
                                    _verify_cfg=self.verify_cfg,
                                    )
        self.initialized = True


    def __call__(self, *args, **kwargs):
        # import pdb; pdb.set_trace()
        if self.concurrent:
            self.run_async(*args)
            return self.output_q

        if self.device == 'pytorch':
            result = self.bound_module(*args, **kwargs)
        else:
            self.ensure_initialized(*args)

            if self.masked_odkv:
                # print('run_generate1')
                self.pybuda.sync()
                in_args = list(args[0]) + list(args[1]) + list(args[2]) + list(args[3]) + list(args[4]) + list(args[5])
                self.tt0.push_to_inputs(in_args)   # don't pass in kv over and over again
                self.pybuda.run_generate(input_count=1, write_index=0, _sequential=True)
            elif self.odkv:
                self.pybuda.sync()
                in_args = list(args[0]) + list(args[1]) + list(args[2]) + list(args[3])
                self.tt0.push_to_inputs(in_args)   # don't pass in kv over and over again
                self.pybuda.run_generate(input_count=1, write_index=0, _sequential=True)
            else:
                self.tt0.push_to_inputs(*args)
                self.pybuda.run_forward(input_count=1, _sequential=True)
            ys = self.output_q.get()
            outputs = tuple([ y.value().float() for y in ys if isinstance(y, self.pybuda.tensor.TensorFromPytorch)])
            if len(outputs) == 1:
                outputs = outputs[0]
            if self.verify_cfg:
                baseline = self.bound_module(*args, **kwargs)
                if len(outputs) != len(baseline):
                    print(f'Num outputs: {len(outputs)}, expected: {len(baseline)}')
                for i, (real, expected) in enumerate(zip(outputs, baseline)):
                    pcc = torch.corrcoef(torch.stack([real.reshape(-1), expected.reshape(-1)]))[0,1]
                    print('PCC tensor %d: %.4f' % (i, pcc))


            result = outputs
        return result

    def add_sched(self, pybuda, entries, exits, ops, factor, constr):
        for elem in entries:
            constr.append(elem)
        for lst in ops:
            for f in range(factor):
                for i,op in enumerate(lst):
                    fop = f'fractured_{f}_{op}'
                    if i == 0:
                        if f == 0:
                            print(f"[add_sched]: Override op temp. epoch: {fop}, chip {f}")
                            pybuda.config.override_op_placement(fop, chip_id=f, temporal_epoch_break=True)
                        else:
                            print(f"[add_sched]: Override op spatial epoch: {fop}, chip {f}")
                            pybuda.config.override_op_placement(fop, chip_id=f, spatial_epoch_break=True)
                    constr.append(fop)
        # for elem in exits:
            # constr.append(elem)
        # pybuda.config.override_op_placement(exits[0], temporal_epoch_break=True)
        print(f"[add_sched] sched: {constr}")
        return constr

