# OmniQuant-RaZeR

OmniQuant-RaZeR is the implementation of RaZeR datatype with [OmniQuant](https://github.com/OpenGVLab/OmniQuant) framework.

## Installation 

Run the following commands in the `OmniQuant-RaZeR` directory to install the package:

```bash
cd OmniQuant-RaZeR
conda create -n omniquant-razer python=3.11 -y
conda activate omniquant-razer
pip install --upgrade pip 
python -m pip install -e .
```

## OmniQuant-RaZeR model zoo

Verified models with RaZeR datatype are listed below:

| Models    | Sizes                           | W3A16   | W4A16 |
| -------   | ------------------------------- | -----   | ------|
| LLaMA-2   | 7B/13B                          | ✅      | ✅     |
| LLaMA-3   | 8B                              | ✅      | ✅     |
| LLaMA-3.2 | 1B/3B                           | ✅      | ✅     |


## Usage

We used weight only quantization for the RaZeR datatype. After installing the package, you can run the following command to quantize the model and evaluate the performance:

```bash
# llama-2-7b 3 bits
python main.py --model path/to/llama-2-7b-hf --epochs 20 --output_dir ./log/llama-2-7b-hf-w3a16g128 --eval_ppl --wbits 3 --abits 16 --group_size 128 --lwc --lwc_lr 0.015 --aug_loss --datatype mod
# llama-2-7b 4 bits
python main.py --model path/to/llama-2-7b-hf --epochs 20 --output_dir ./log/llama-2-7b-hf-w4a16g128 --eval_ppl --wbits 4 --abits 16 --group_size 128 --lwc --lwc_lr 0.015 --aug_loss --datatype mod
```
The flag `--datatype` specifies the quantization datatype, and can be `int` for integer quantization, or `fp` for floating point quantization, or `mod` for RaZeR quantization.
The flag `--eval_ppl` will evaluate the Wikitext and C4 perplexity after running the OmniQuant weight quantization. You may check the perplexity results in the log file under `--output_dir`.

The comprehensive scripts to reproduce the results are available in the `scripts` directory. (please modify the model path before running the scripts)

