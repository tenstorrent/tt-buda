# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import sys
import os
import torch
from amp_configs import amp_config
from placement import manual_placer


class PyBudify(torch.nn.Module):
    def __init__(self, pt_module, device='silicon', arch='wormhole_b0', precision='fp32', amp_config_file=None, micro_batch_size=1, fuse=False, num_chips=1, perf=None, verify=False, log_level='ERROR', tti_save=None, tti_load=None,
                 prefill_kvs=[], write_index=0, num_layers=None, netlist_name="pybudify_module", opt_level=0, nlp_target_cycles=-1, placement_config_file=None):
        super().__init__()

        self.device = device
        self.bound_module = pt_module
        self.tti_save = tti_save
        self.tti_load = tti_load
        self.prefill_kvs = prefill_kvs
        self.write_index = write_index

        if device != 'pytorch':
            # pybuda workarounds
            os.environ["GOLDEN_WORMHOLE_B0"] = "1"
            # os.environ["PYBUDA_ENABLE_BROADCAST_SPLITTING"] = "1"
            #os.environ["PYBUDA_DISABLE_FORK_JOIN_BUF"] = "1"
            # os.environ["PYBUDA_DRAM_PICK_CAPACITY"] = "1"
            os.environ["WHA0_DISABLE_RELAY_BUFS"] = "1"
            os.environ["PYBUDA_ENABLE_STABLE_SOFTMAX"] = "1"
            os.environ["PYBUDA_FUSE_STOP_ON_RECIPROCAL"] = "1"
            os.environ["PYBUDA_PLACER_SNAKE"] = "1"
            os.environ["LOGGER_LEVEL"] = log_level
            os.environ["LOGURU_LEVEL"] = log_level
            # os.environ["PYBUDA_DISABLE_FORK_JOIN_BUF"] = "1"
            os.environ["PYBUDA_ENABLE_OUTPUT_QUEUES_ON_HOST"] = "0"
            os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"
            # os.environ["TT_BACKEND_FORCE_SW_TILIZE"] = "1"

            if nlp_target_cycles > 0:
                os.environ["PYBUDA_NLP_MANUAL_TARGET"] = str(nlp_target_cycles)

            pybuda = self.pybuda = __import__('pybuda') # let us set log levels before importing pybuda

        if device == 'pytorch':
            pass
        else:
            devtype = { 'golden' : pybuda.BackendType.Golden,
                        'silicon': pybuda.BackendType.Silicon,
                      }[device]

            module = pybuda.PyTorchModule(netlist_name, self.bound_module)

            assert amp_config_file is not None, "amp_config_file must be specified for PyBudify"
            assert num_layers is not None, "num_layers must be specified for PyBudify"
            # apply_amp_settings manages all dataformats
            amp_config.apply_amp_settings(pybuda, amp_config_file, num_layers)

            num_layers = 32 if num_layers is None else num_layers

            if placement_config_file is not None:
                manual_placer(pybuda.config, placement_config_file, loops=num_layers)


            if self.prefill_kvs:

                # if not fuse:

                OP_OFFSET = 77
                INDEX_OFFSET = 4
                INDEX_START = num_layers * OP_OFFSET

                for layer_num in range(num_layers):
                    k = OP_OFFSET * layer_num
                    j = INDEX_OFFSET * layer_num
                    pybuda.config.add_schedule_constraint([f'concatenate_{30+k}.dc.concatenate.0', f'index_{INDEX_START+j}.dc.select.0', f'index_{INDEX_START+j}.dc.buffer.1', f'{netlist_name}.output_hstack_{INDEX_START+1+j}_tm_nop', f'matmul_{33+k}',
                                                        f'concatenate_{44+k}.dc.concatenate.0', f'index_{INDEX_START+2+j}.dc.select.0', f'index_{INDEX_START+2+j}.dc.buffer.1', f'{netlist_name}.output_hstack_{INDEX_START+3+j}_tm_nop', f'matmul_{48+k}'])
                    
                    pybuda.config.override_op_size(f'{netlist_name}.output_hstack_{INDEX_START + 1 +j}_tm_nop', (1,4))
                    # Removed these since they don't play nice with fusion
                    # if layer_num < (num_layers - 1):
                    #     pybuda.config.add_schedule_constraint([f'add_{76+k}', f'layers.{layer_num+1}.input_layernorm.weight_s_brcst_m2_0_0.lc1'])
                # Ensure all decoders start at same op
                # pybuda.config.set_epoch_break(f'layers.{layer_num}.input_layernorm.weight_s_brcst_m2_0_0.lc1')

                compiler_cfg = pybuda.config._get_global_compiler_config()
                compiler_cfg.loopback_outputs = generate_loopback_dict(num_layers) # IM SORRY JON
                
                                         
            perf_level = { None    : None,
                          'none'   : None,
                          'light'  : pybuda.PerfTraceLevel.LIGHT,
                          'verbose': pybuda.PerfTraceLevel.VERBOSE }[perf]
            pybuda.set_configuration_options(enable_auto_fusing=fuse, performance_trace=perf_level, backend_opt_level=opt_level, input_queues_on_host=False)

            pybuda_arch = { 'grayskull': pybuda.BackendDevice.Grayskull,
                            'wormhole_b0': pybuda.BackendDevice.Wormhole_B0 }[arch]
            
            if tti_load is not None:
                self.tt0 = pybuda.TTDevice.load_image(img_path=tti_load)
            else:
                self.tt0 = pybuda.TTDevice('tt0', module=module,
                                            # fp32_fallback=fallback,
                                            arch=pybuda_arch,
                                            devtype=devtype,
                                            chip_ids=list(range(num_chips)))
                    
            mp = torch.multiprocessing.get_context('spawn')
            self.output_q = mp.Queue()

            if verify:
                def debug_tensors(t1, t2):                    
                    """ return True if the tensors match and print them if they don't and return false """
                    max_err = torch.max(torch.abs(t1 - t2)).item()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    print()
                    print('YO DAWG I GOT YO TENSORS')
                    print('Expected:', t1.shape)
                    print(t1.tolist())
                    print('Observed:', t2.shape)
                    print(t2.tolist())
                    max_d0_err = torch.max(torch.abs(t1[0] - t2[0]))
                    print('Max abs error for [0, ...] is', max_d0_err)
                    print('Max abs error overall is', max_err)
                    print()
                    sys.stdout.flush()
                    return max_err > 1
#                    return True

                self.verify_cfg = pybuda.VerifyConfig(verify_all=True,
                                                      verify_last=True,
                                                      devtype=device,
                                                      arch=pybuda_arch,
#                                                      golden_compare_callback=debug_tensors,
                                                      intermediates=False,
                                                      golden_ignore_df_precision=False)
            else:
                self.verify_cfg = None

            self.initialized = False
            self.micro_batch_size = micro_batch_size


    def __call__(self, *args, **kwargs):
        # print(f'In pybudify: num args: {len(args)}')
        # print(args)
        if self.device == 'pytorch':
            # Make sure to reuse args
            if self.prefill_kvs:
                pt_args = list(args) + self.prefill_kvs
            else:
                pt_args = args
            result = self.bound_module(*pt_args, **kwargs)
        else:
            if not self.initialized:
                if self.prefill_kvs:
                    init_args = list(args) + self.prefill_kvs
                else:
                    init_args = args

                if self.tti_save is not None:
                    self.tt0.compile_to_image(
                        img_path=self.tti_save,
                        training=False,
                        sample_inputs=init_args,
                        microbatch_count=self.micro_batch_size,
                    )
                    print(f'Saved image to {self.tti_save}')
                    sys.exit(0)
                self.pybuda.initialize_pipeline(training=False,
                                        sample_inputs=init_args,
                                        output_queue=self.output_q,
                                        microbatch_count=self.micro_batch_size,
                                        _sequential=True, # FIXME: can we implement concurrent mode and still have a wrapper?
                                        _verify_cfg=self.verify_cfg,
                                        )
                self.initialized = True
                self.pybuda.sync()

            self.tt0.push_to_inputs(*args)
            if self.prefill_kvs:
                self.pybuda.run_generate(input_count=1, write_index=self.write_index, _sequential=True)
            else:
                self.pybuda.run_forward(input_count=1, _sequential=True)
            ys = self.output_q.get()
            outputs = tuple([ y.value().float() for y in ys if isinstance(y, self.pybuda.tensor.TensorFromPytorch)])
            if self.verify_cfg: 
                baseline = self.bound_module(*args, **kwargs)
                if len(outputs) != len(baseline):
                    print(f'Num outputs: {len(outputs)}, expected: {len(baseline)}')
                for i, (real, expected) in enumerate(zip(outputs, baseline)):
                    pcc = torch.corrcoef(torch.stack([real.view(-1), expected.view(-1)]))[0,1]
                    print('PCC tensor %d: %.3f (%d nans)' % (i, pcc, torch.isnan(real).sum().item()))

            result = outputs

        return result

def generate_loopback_dict(num_layers):
    """ Names found by looking in generated_modules/pybudify_module.py """
    names = 'k_past_1, v_past_1, k_past_5, v_past_5, k_past_9, v_past_9, k_past_13, v_past_13, k_past_17, v_past_17, k_past_21, v_past_21, k_past_25, v_past_25, k_past_29, v_past_29, k_past_33, v_past_33, k_past_37, v_past_37, k_past_41, v_past_41, k_past_45, v_past_45, k_past_49, v_past_49, k_past_53, v_past_53, k_past_57, v_past_57, k_past_61, v_past_61, k_past_65, v_past_65, k_past_69, v_past_69, k_past_73, v_past_73, k_past_77, v_past_77, k_past_81, v_past_81, k_past_85, v_past_85, k_past_89, v_past_89, k_past_93, v_past_93, k_past_97, v_past_97, k_past_101, v_past_101, k_past_105, v_past_105, k_past_109, v_past_109, k_past_113, v_past_113, k_past_117, v_past_117, k_past_121, v_past_121, k_past_125, v_past_125'
    # names = "k_past_1, v_past_1"
    # print('WARNING: using hardcoded names for loopback dict! ONLY WORKS FOR ONE LAYER')
    names = names.split(', ')

    # Select only the first num_layer
    names = names[:2*num_layers]

    # with open('generated_modules/pybudify_module.py') as f:
    #     lines = f.read()
    #     ret = [ line.split('def forward(self, ')[1] for line in lines if 'def forward(self, ' in line ][0]
    #     vars = ret.split(', ')[4:]

    # if vars != names:
    #     print('loopback names appear to have changed, using the new values but CHECK THIS:')
    #     print(', '.join(vars))
    #     names = vars

    return { name: (i+1) for i, name in enumerate(names) }
