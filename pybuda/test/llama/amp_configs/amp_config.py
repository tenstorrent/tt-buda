# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import json
import os

def apply_mlp(pybuda, config):
    # Config could be df string or a dict of df strings
    if isinstance(config, str):
        df = str_to_dataformat(pybuda, config)
        gate_df = up_df = down_df = df
    else:
        gate_df = str_to_dataformat(pybuda, config['gate'])
        up_df = str_to_dataformat(pybuda, config['up'])
        down_df = str_to_dataformat(pybuda, config['down'])

    # MLP dataformat is applied to MLP weights with this regex
    pybuda.config.configure_mixed_precision(
        output_df=gate_df,
        name_regex=".*mlp.gate_proj.weight.*",
        input_df={0: [gate_df, True]})

    pybuda.config.configure_mixed_precision(
        output_df=up_df,
        name_regex=".*mlp.up_proj.weight.*",
        input_df={0: [up_df, True]})

    pybuda.config.configure_mixed_precision(
        output_df=down_df,
        name_regex=".*mlp.down_proj.weight.*",
        input_df={0: [down_df, True]})


def apply_attn(pybuda, config):
    # Config could be df string or a dict of df strings
    if isinstance(config, str):
        df = str_to_dataformat(pybuda, config)
        q_df = k_df = v_df = o_df = df
    else:
        q_df = str_to_dataformat(pybuda, config['q'])
        k_df = str_to_dataformat(pybuda, config['k'])
        v_df = str_to_dataformat(pybuda, config['v'])
        o_df = str_to_dataformat(pybuda, config['o'])

    # Attention dataformat is applied to attention weights with this regex
    pybuda.config.configure_mixed_precision(
        output_df=q_df,
        name_regex=".*self_attn.q_proj.weight.*",
        input_df={0: [q_df, True]})
    pybuda.config.configure_mixed_precision(
        output_df=k_df,
        name_regex=".*self_attn.k_proj.weight.*",
        input_df={0: [k_df, True]})
    pybuda.config.configure_mixed_precision(
        output_df=v_df,
        name_regex=".*self_attn.v_proj.weight.*",
        input_df={0: [v_df, True]})
    pybuda.config.configure_mixed_precision(
        output_df=o_df,
        name_regex=".*self_attn.o_proj.weight.*",
        input_df={0: [o_df, True]})


def apply_cache(pybuda, config, num_layers):
    # Config could be df string or a dict of df strings
    if isinstance(config, str):
        df = str_to_dataformat(pybuda, config)
        key_df = df
        value_df = df
    else:
        key_df = str_to_dataformat(pybuda, config['key'])
        value_df = str_to_dataformat(pybuda, config['value'])

    pybuda.config.configure_mixed_precision(
        output_df=key_df,
        name_regex="k_past_.*",
        input_df={0: [key_df, True]})

    pybuda.config.configure_mixed_precision(
        output_df=value_df,
        name_regex="v_past_.*",
        input_df={0: [value_df, True]})
    

    # Also let's loop over the concatenate ops and make sure they are using this DF.
    # Otherwise we get garbage outputs. Bad.
    # TODO: Figure out a more programmatic way to figure out these op names
    OP_OFFSET = 77
    INDEX_START = num_layers * OP_OFFSET
    HSTACK_OFFSET = 4
    for i in range(num_layers):
        k = OP_OFFSET * i
        j = HSTACK_OFFSET * i
        # special-case key ops
        pybuda.config.configure_mixed_precision(
            output_df=key_df,
            name_regex=f'concatenate_{30+k}.dc.concatenate.0',
            input_df={0: [key_df, True], 1: [key_df, True]})

        # Write-view also needs overriding
        pybuda.config.configure_mixed_precision(
            output_df=key_df,
            name_regex=f".*output_hstack_{INDEX_START + 1 +j}.*",
            input_df={0: [key_df, True]})

        # special-case value ops
        pybuda.config.configure_mixed_precision(
            output_df=value_df,
            name_regex=f'concatenate_{44+k}.dc.concatenate.0',
            input_df={0: [value_df, True], 1: [value_df, True]})

        # Write-view also needs overriding
        pybuda.config.configure_mixed_precision(
            output_df=value_df,
            name_regex=f".*output_hstack_{INDEX_START + 3 +j}.*",
            input_df={0: [value_df, True]})


def apply_matmul_acc(pybuda, df):
    pybuda.config.configure_mixed_precision(
        op_type="matmul",
        intermediate_df=df,
        accumulate_df=df,
    )


def apply_default(pybuda, df):
    # Default dataformat is applied to all other weights with this regex
    pybuda.set_configuration_options(default_df_override=df, accumulate_df=df)


def apply_attn_mask(pybuda, df):
    # MLP dataformat is applied to MLP weights with this regex
    pybuda.config.configure_mixed_precision(
    output_df=df,
    name_regex="attention_mask",
    input_df={0: [df, True]})


def str_to_dataformat(pybuda, df_str):
    if df_str == 'fp32':
        df = pybuda.DataFormat.Float32
    elif df_str == 'fp16':
        df = pybuda.DataFormat.Float16
    elif df_str == 'bf16':
        df = pybuda.DataFormat.Float16_b
    elif df_str == 'fp8':
        df = pybuda.DataFormat.Bfp8
    elif df_str == 'fp8b':
        df = pybuda.DataFormat.Bfp8_b
    elif df_str == 'fp4b':
        df = pybuda.DataFormat.Bfp4_b
    elif df_str == 'fp2b':
        df = pybuda.DataFormat.Bfp2_b
    else:
        raise ValueError('Precision "%s" not implemented' % precision)
    return df

def apply_amp_settings(pybuda, config_file, num_layers):
    print('Applying AMP from file ', config_file, flush=True)
    # Open config json
    with open(config_file) as f:
        config = json.load(f)

    '''
    For now, this file has hard-coded ideas of what AMP means, as it applies to Llama.
    Ex: MLP amp means set MLP weights to some df.
    '''
    for k, v in config.items():
        if k == "mm_acc_df":
            apply_matmul_acc(pybuda, str_to_dataformat(pybuda, v))
        elif k == "mlp_df":
            apply_mlp(pybuda, v)
        elif k == "attn_df":
            apply_attn(pybuda, v)
        elif k == "cache_df":
            apply_cache(pybuda, v, num_layers)
        elif k == "default_df":
            apply_default(pybuda, str_to_dataformat(pybuda, v))
        elif k == "attn_mask_df":
            apply_attn_mask(pybuda, str_to_dataformat(pybuda, v))
        else:
            raise ValueError('Config "%s" not implemented' % k)

