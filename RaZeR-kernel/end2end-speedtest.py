import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from razer_linear import RazerLinear
try:
    from marlin import Layer as MarlinLinear
except ImportError:
    print("Marlin will be required for the Marlin implementation")
import numpy as np
import os
from tqdm import tqdm
import argparse




def test_model_aio(model, tokenizer, description=None, NUM_REPEATS=5):
    device = model.device
    if description:
        print(description)

    # prompt = '''count from 1 to 1000, seperated by commas: 1, 2'''
    prompt = '''once upon a time, '''
    input_ids = tokenizer(prompt, return_tensors='pt').to(device)
    max_new_tokens = 200


    # warmup
    with torch.no_grad():
        for _ in tqdm(range(5), desc="Warmup"):
            _ = model.generate(**input_ids, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, temperature=0.001)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    ttfts = []
    torch.cuda.synchronize()
    for i in tqdm(range(NUM_REPEATS), desc="TTFT"):
        start_event.record()
        with torch.no_grad():
            output = model.generate(**input_ids, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id, temperature=0.001)
        end_event.record()

        # Wait for the events to be recorded
        torch.cuda.synchronize()

        ttft = start_event.elapsed_time(end_event)
        ttfts.append(ttft)

    ttft = np.mean(ttfts)
    print(f"Time to first token (TTFT): {ttft:.4f} ms")


    tpots = []
    pbar = tqdm(range(NUM_REPEATS), desc="TPOT")
    for i in pbar:
        start_event.record()
        with torch.no_grad():
            output = model.generate(**input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, temperature=0.001)
        end_event.record()

        # Wait for the events to be recorded
        torch.cuda.synchronize()
        pbar.set_postfix(custom_info=f'input token length: {input_ids["input_ids"].shape[1]}, output token length: {output.shape[1]}, number of generated tokens: {output.shape[1] - input_ids["input_ids"].shape[1]}')
        tpot = (start_event.elapsed_time(end_event) - ttft) / (output.shape[1] - input_ids['input_ids'].shape[1])
        tpots.append(tpot)

    tpot = np.mean(tpots)
    print(f"Time per output token (TPOT): {tpot:.4f} ms")
    return ttft, tpot

def replace_layers_razer(model, omni_parameters = None):
    device = model.device
    num_layers = len(model.model.layers)
    if omni_parameters is not None:
        assert len(omni_parameters) == num_layers

    for i in tqdm(range(num_layers), desc="Bitmod: quantizing and replacing layers"):
        layer = model.model.layers[i]
        # replace self_atten: q_proj, k_proj, v_proj, o_proj, mlp: gate_proj, up_proj, down_proj
        q_proj_omni_dict = {'symbound_factor': omni_parameters[i]['self_attn.q_proj.weight_quantizer.symbound_factor']} if omni_parameters is not None else None
        q_proj_razer = RazerLinear(layer.self_attn.q_proj, 4, 128, q_proj_omni_dict)
        q_proj_razer.to(device)
        q_proj_razer.construct()
        setattr(layer.self_attn, 'q_proj', q_proj_razer)

        k_proj_omni_dict = {'symbound_factor': omni_parameters[i]['self_attn.k_proj.weight_quantizer.symbound_factor']} if omni_parameters is not None else None
        k_proj_razer = RazerLinear(layer.self_attn.k_proj, 4, 128, k_proj_omni_dict)
        k_proj_razer.to(device)
        k_proj_razer.construct()
        setattr(layer.self_attn, 'k_proj', k_proj_razer)

        v_proj_omni_dict = {'symbound_factor': omni_parameters[i]['self_attn.v_proj.weight_quantizer.symbound_factor']} if omni_parameters is not None else None
        v_proj_razer = RazerLinear(layer.self_attn.v_proj, 4, 128, v_proj_omni_dict)
        v_proj_razer.to(device)
        v_proj_razer.construct()
        setattr(layer.self_attn, 'v_proj', v_proj_razer)

        o_proj_omni_dict = {'symbound_factor': omni_parameters[i]['self_attn.o_proj.weight_quantizer.symbound_factor']} if omni_parameters is not None else None
        o_proj_razer = RazerLinear(layer.self_attn.o_proj, 4, 128, o_proj_omni_dict)
        o_proj_razer.to(device)
        o_proj_razer.construct()
        setattr(layer.self_attn, 'o_proj', o_proj_razer)

        gate_proj_omni_dict = {'symbound_factor': omni_parameters[i]['mlp.gate_proj.weight_quantizer.symbound_factor']} if omni_parameters is not None else None
        gate_proj_razer = RazerLinear(layer.mlp.gate_proj, 4, 128, gate_proj_omni_dict)
        gate_proj_razer.to(device)
        gate_proj_razer.construct()
        setattr(layer.mlp, 'gate_proj', gate_proj_razer)

        up_proj_omni_dict = {'symbound_factor': omni_parameters[i]['mlp.up_proj.weight_quantizer.symbound_factor']} if omni_parameters is not None else None
        up_proj_razer = RazerLinear(layer.mlp.up_proj, 4, 128, up_proj_omni_dict)
        up_proj_razer.to(device)
        up_proj_razer.construct()
        setattr(layer.mlp, 'up_proj', up_proj_razer)

        down_proj_omni_dict = {'symbound_factor': omni_parameters[i]['mlp.down_proj.weight_quantizer.symbound_factor']} if omni_parameters is not None else None
        down_proj_razer = RazerLinear(layer.mlp.down_proj, 4, 128, down_proj_omni_dict)
        down_proj_razer.to(device)
        down_proj_razer.construct()
        setattr(layer.mlp, 'down_proj', down_proj_razer)

    torch.cuda.empty_cache()
    print("Model quantized and replaced successfully")

def replace_layers_marlin(model):

    def generate_scale_marlin(w:torch.Tensor):
        # w = w.t()
        group_size = 128
        dim1, dim2 = w.shape
        w_reg = w.reshape([-1, group_size])
        # compute the range of the w_reg, keep dimsion
        w_max = w_reg.max(dim=1, keepdim=True).values
        w_min = w_reg.min(dim=1, keepdim=True).values
        # compute the scale
        scale = (w_max - w_min) / 15
        # reshape back to dim1, dim2 // group_size
        scale = scale.reshape([dim1, dim2 // group_size])
        return scale

    device = model.device
    num_layers = len(model.model.layers)
    for i in tqdm(range(num_layers), desc="Marlin: quantizing and replacing layers"):
        layer = model.model.layers[i]


        original_q_proj = layer.self_attn.q_proj
        q_proj_marlin = MarlinLinear(layer.self_attn.q_proj.in_features, layer.self_attn.q_proj.out_features, 128)
        q_proj_marlin.to(device)
        q_proj_marlin.pack(original_q_proj, generate_scale_marlin(original_q_proj.weight))
        setattr(layer.self_attn, 'q_proj', q_proj_marlin)
        del original_q_proj.weight
        del original_q_proj

        original_k_proj = layer.self_attn.k_proj
        k_proj_marlin = MarlinLinear(layer.self_attn.k_proj.in_features, layer.self_attn.k_proj.out_features, 128)
        k_proj_marlin.to(device)
        k_proj_marlin.pack(original_k_proj, generate_scale_marlin(original_k_proj.weight))
        setattr(layer.self_attn, 'k_proj', k_proj_marlin)
        del original_k_proj.weight
        del original_k_proj

        original_v_proj = layer.self_attn.v_proj
        v_proj_marlin = MarlinLinear(layer.self_attn.v_proj.in_features, layer.self_attn.v_proj.out_features, 128)
        v_proj_marlin.to(device)
        v_proj_marlin.pack(original_v_proj, generate_scale_marlin(original_v_proj.weight))
        setattr(layer.self_attn, 'v_proj', v_proj_marlin)
        del original_v_proj.weight
        del original_v_proj

        original_o_proj = layer.self_attn.o_proj
        o_proj_marlin = MarlinLinear(layer.self_attn.o_proj.in_features, layer.self_attn.o_proj.out_features, 128)
        o_proj_marlin.to(device)
        o_proj_marlin.pack(original_o_proj, generate_scale_marlin(original_o_proj.weight))
        setattr(layer.self_attn, 'o_proj', o_proj_marlin)
        del original_o_proj.weight
        del original_o_proj

        original_gate_proj = layer.mlp.gate_proj
        gate_proj_marlin = MarlinLinear(layer.mlp.gate_proj.in_features, layer.mlp.gate_proj.out_features, 128)
        gate_proj_marlin.to(device)
        gate_proj_marlin.pack(original_gate_proj, generate_scale_marlin(original_gate_proj.weight))
        setattr(layer.mlp, 'gate_proj', gate_proj_marlin)
        del original_gate_proj.weight
        del original_gate_proj

        original_up_proj = layer.mlp.up_proj
        up_proj_marlin = MarlinLinear(layer.mlp.up_proj.in_features, layer.mlp.up_proj.out_features, 128)
        up_proj_marlin.to(device)
        up_proj_marlin.pack(original_up_proj, generate_scale_marlin(original_up_proj.weight))
        setattr(layer.mlp, 'up_proj', up_proj_marlin)
        del original_up_proj.weight
        del original_up_proj

        original_down_proj = layer.mlp.down_proj
        down_proj_marlin = MarlinLinear(layer.mlp.down_proj.in_features, layer.mlp.down_proj.out_features, 128)
        down_proj_marlin.to(device)
        down_proj_marlin.pack(original_down_proj, generate_scale_marlin(original_down_proj.weight))
        setattr(layer.mlp, 'down_proj', down_proj_marlin)
        del original_down_proj.weight
        del original_down_proj

    torch.cuda.empty_cache()
    print("Model quantized and replaced successfully")


def main():
    parser = argparse.ArgumentParser(description='Test the speed of razer quantized model')
    parser.add_argument('--model', type=str, help='The path to the model', required=True)
    parser.add_argument('--omni', type=str, default='', help='The path to the omni parameters')
    parser.add_argument('--num_repeats', type=int, default=10, help='The number of repeats for each test')
    parser.add_argument('--impl', type=str, default='razer', help='The implementation to test', choices=['razer', 'marlin'])
    args = parser.parse_args()

    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError("CUDA device not available. Please check your GPU setup.")
    

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.half()  # Convert model parameters to FP16
    model.to(device)
    model = model.eval()
    print(f"Model {args.model.split()[-1]} loaded successfully")

    ttft_fp16, tpot_fp16 = test_model_aio(model, tokenizer, 'fp16 model performance', args.num_repeats)

    if args.impl == 'razer':
        if args.omni != '':
            omni_parameters = torch.load(args.omni, weights_only=True)
        else:
            omni_parameters = None
        replace_layers_razer(model, omni_parameters)
    elif args.impl == 'marlin':
        replace_layers_marlin(model)
    else:
        raise ValueError("Invalid implementation")
    
    # model.to(device)
    
    ttft_quantized, tpot_quantized = test_model_aio(model, tokenizer, f'{args.impl} quantized model performance', args.num_repeats)

    ttft_speedup = ttft_fp16 / ttft_quantized
    tpot_speedup = tpot_fp16 / tpot_quantized

    print(f"TTFT speedup: {ttft_speedup:.4f}")
    print(f"TPOT speedup: {tpot_speedup:.4f}")





if __name__ == '__main__':
    main()