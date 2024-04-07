
import numpy as np
import torch
import marlin
import razer_kernels
import time

def benchmark(f, warmup=5, iter=100):
    for i in range(warmup + iter):
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(2.)
    return res


def get_problem(m, n, k):
    print(f'get_problem: m: {m}, n: {n}, k: {k}')
    groupsize = 128
    dev = torch.device('cuda:0')
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B_fp16 = torch.randn((k, n), dtype=torch.half, device=dev)
    B_marlin = torch.randint(low=-2**31, high=2**31, size=(k * n // 8,), device=dev)
    B_bitmod = torch.randint(-2**31, 2**31, (n, k // 8), device=dev, dtype=torch.int32)
    scaling_marlin = torch.rand((k // groupsize, n), dtype=torch.half, device=dev)
    scaling_bitmod = torch.rand((n, k // groupsize), dtype=torch.half, device=dev)
    bitmods = torch.randint(-2**31, 2**31, (n, k // 8 // groupsize), device=dev, dtype=torch.int32)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B_fp16, B_marlin, B_bitmod, scaling_marlin, scaling_bitmod, bitmods, C


def benchmark_dense(A, B_fp16, C, iter = 100):
    res = benchmark(lambda: torch.matmul(A, B_fp16, out=C), iter=iter)
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 2 * B_fp16.numel() + 2 * C.numel()) / res / 10 ** 9
    }

def benchmark_marlin(A, B_marlin, C, scaling_marlin, thread_k, thread_n, sms, iter = 100):
    # print(f'marlin test initiated, A: {A.shape}, B: {B_marlin.shape}, C: {C.shape}, scaling: {scaling_marlin.shape}, thread_k: {thread_k}, thread_n: {thread_n}, sms: {sms}')
    workspace = torch.zeros(C.shape[1] // 128 * 16, device=torch.device('cuda:0'))
    res = benchmark(lambda: marlin.mul(A, B_marlin, C, scaling_marlin, workspace, thread_k, thread_n, sms), iter=iter)
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 4 * B_marlin.numel() + 2 * C.numel() + 2 * scaling_marlin.numel()) / res / 10 ** 9
    }

def benchmark_bitmod(A, B_bitmod, C, scaling_bitmod, mods, iter = 100):
    if A.shape[1] % 1024 == 0:
        res = benchmark(lambda: razer_kernels.razer_gpu_symmetric_perf(A, B_bitmod, scaling_bitmod, mods, C), iter=iter)
    elif A.shape[1] % 512 == 0:
        res = benchmark(lambda: razer_kernels.razer_gpu_symmetric_512_perf(A, B_bitmod, scaling_bitmod, mods, C), iter=iter)
    elif A.shape[1] % 256 == 0:
        res = benchmark(lambda: razer_kernels.razer_gpu_symmetric_256_perf(A, B_bitmod, scaling_bitmod, mods, C), iter=iter)
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 4 * B_bitmod.numel() + 2 * C.numel() + 2 * scaling_bitmod.numel() + 4 * mods.numel()) / res / 10 ** 9
    }





SMS = -1

def main_square():


    # M = 1
    # K = 4096
    # N = 4096

    total_results = []
    for side_length in [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]:
        M = 1
        N = side_length
        K = side_length

        A, B_fp16, B_marlin, B_bitmod, scaling_marlin, scaling_bitmod, bitmods, C = get_problem(M, N, K)

        result_fp16 = benchmark_dense(A, B_fp16, C)
        result_marlin = benchmark_marlin(A, B_marlin, C, scaling_marlin, -1, -1, SMS)
        result_bitmod = benchmark_bitmod(A, B_bitmod, C, scaling_bitmod, bitmods)

        print(f"FP16: {result_fp16}")
        print(f"Marlin: {result_marlin}")
        print(f"Bitmod: {result_bitmod}")
        total_results.append(f'{side_length}, {result_fp16["TFLOP/s"]}, {result_marlin["TFLOP/s"]}, {result_bitmod["TFLOP/s"]}, {1e6 * result_fp16["s"]}, {1e6 * result_marlin["s"]}, {1e6 * result_bitmod["s"]}')
    

    print('side_length, fp16_TFLOP/s, marlin_TFLOP/s, bitmod_TFLOP/s, fp16_us, marlin_us, bitmod_us')
    for res in total_results:
        print(res)

def main_llama():

    # M      N             K
    # batch, out_features, in_features
    size_dicts ={
        'llama-3.2-1b-instruct': [(1, 2048, 2048), (1, 512, 2048), (1, 8192, 2048), (1, 2048, 8192)],
        'llama-3.2-3b-instruct': [(1, 3072, 3072), (1, 1024, 3072), (1, 8192, 3072), (1, 3072, 8192)],
        'llama-3-8b-hf' : [(1, 4096, 4096), (1, 1024, 4096), (1, 14336, 4096), (1, 4096, 14336)],
        'llama-2-7b-hf' : [(1, 4096, 4096), (1, 11008, 4096), (1, 4096, 11008)],
        'llama-2-13b-hf': [(1, 5120, 5120), (1, 13824, 5120), (1, 5120, 13824)],
    }

    result_dict = {}
    for model_type in size_dicts.keys():
        sizes = size_dicts[model_type]
        results = []
        for dim3 in sizes:
            M, N, K = dim3

            A, B_fp16, B_marlin, B_bitmod, scaling_marlin, scaling_bitmod, bitmods, C = get_problem(M, N, K)

            result_fp16 = benchmark_dense(A, B_fp16, C)
            print(f"FP16: {result_fp16}")
            result_marlin = benchmark_marlin(A, B_marlin, C, scaling_marlin, -1, -1, SMS)
            print(f"Marlin: {result_marlin}")
            result_bitmod = benchmark_bitmod(A, B_bitmod, C, scaling_bitmod, bitmods)
            print(f"Bitmod: {result_bitmod}")

            fp16_us = 1e6 * result_fp16["s"]
            marlin_us = 1e6 * result_marlin["s"]
            bitmod_us = 1e6 * result_bitmod["s"]
            results.append((fp16_us, marlin_us, bitmod_us))

        result_dict[model_type] = results

    print(result_dict)

if __name__ == '__main__':
    main_square()