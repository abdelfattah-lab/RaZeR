# The Power of Negative Zero: Datatype Customization for Quantized Large Language Models

**R**edund**a**nt **Ze**ro **R**emapping (**RaZeR**) is an algorithm that maps negative zero in the floating point representation to a useful value, adding extra precision when bit width is limited. RaZeR is designed to be used in quantized large language models, where the precision of the weights is crucial for the model's performance. In this repository, we provide the implementation of RaZeR in CUDA with PyTorch, and the code to reproduce the experiments in the paper.

Current version focus on the GEMV operation, and the GEMM optimization will be released soon.

## Installation

```
conda create -n razer python=3.11 -y
conda activate razer

pip install -r requirements.txt
python setup.py build_ext --inplace
```

To compare the speed of Razer with Marlin, you need to install the Marlin library. Please follow the instructions in the [Marlin repository](https://github.com/IST-DASLab/marlin).

## Usage

### GEMV benchmark

After installing Razer and Marlin, you can run the following command to compare the speed of Razer with Marlin on the GEMV operation. Please check the `benchmark.py` file for more details and customization.

```
python benchmark.py
```

### end-to-end inference speedtest

use the `end2end-speedtest.py` for end-to-end inference speedtest. It runs the inference with the specified implementation. The `--impl` argument can be `razer` or `marlin`. The `--num_repeats` argument specifies the number of times to repeat the inference. The `--model` argument specifies the model to use

```
# example usage, please modify the model path accordingly
python end2end-speedtest.py --model /data/models/llama-2-7b-hf --impl razer --num_repeats 5
```

<!-- optionally it can use the omniquant parameter for a better quantization, but this does not affect the speedtest. -->

### other customization integration

The `razer_linear.py` contains the implementation of the RaZeR linear layer. You can integrate it with your model by replacing the `torch.nn.Linear` with `razer_linear.RazerLinear`.


``` python
# example usage to replace the linear layer in the transformer model
k_proj_razer = RazerLinear(layer.self_attn.k_proj, 4, 128)
k_proj_razer.to(device)
k_proj_razer.construct()
setattr(layer.self_attn, 'k_proj', k_proj_razer)
```