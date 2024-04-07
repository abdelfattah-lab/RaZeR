import os
from pathlib import Path
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9+PTX'

extra_compile_args = {
    "cxx": [
        "-g", 
        "-O3", 
        "-fopenmp", 
        "-lgomp", 
        "-std=c++17",
        # "-DENABLE_BF16"
    ],
    "nvcc": [
        # "-O0", "-G", "-g", # uncomment for debugging
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",  # TODO
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        # "--threads=8"
        "-g",
        "-lineinfo",
    ],
}

setup(
    name="RazerKernel",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="razer_kernels",
            sources=[
                "csrc/pybind.cpp",
                "csrc/razer_gpu_symmetric.cu",
                "csrc/razer_gpu_symmetric_512.cu",
                "csrc/razer_gpu_symmetric_256.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)