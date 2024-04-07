#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


// used for linear layer inference
torch::Tensor razer_gpu_symmetric(
    at::Tensor fA,              // [B, M, K], fp16
    at::Tensor qB,              // [N, K_packed], int32, of 4-bit quantized values
    at::Tensor scaling_factors, // [N, K / group_size], fp16, group_size = 128
    at::Tensor bitmod          // [N, (K / group_size) / 8], int4 packed in int32 format, each int4 upper 2 bit is always 0, and lower 2 bits indicate the bitmod for this group
);

torch::Tensor razer_gpu_symmetric_512(
    at::Tensor fA,              // [B, M, K], fp16
    at::Tensor qB,              // [N, K_packed], int32, of 4-bit quantized values
    at::Tensor scaling_factors, // [N, K / group_size], fp16, group_size = 128
    at::Tensor bitmod          // [N, (K / group_size) / 8], int4 packed in int32 format, each int4 upper 2 bit is always 0, and lower 2 bits indicate the bitmod for this group
);

torch::Tensor razer_gpu_symmetric_256(
    at::Tensor fA,              // [B, M, K], fp16
    at::Tensor qB,              // [N, K_packed], int32, of 4-bit quantized values
    at::Tensor scaling_factors, // [N, K / group_size], fp16, group_size = 128
    at::Tensor bitmod          // [N, (K / group_size) / 8], int4 packed in int32 format, each int4 upper 2 bit is always 0, and lower 2 bits indicate the bitmod for this group
);



// used for performance evaluation
void razer_gpu_symmetric_perf(
    at::Tensor fA,              // [M, K], fp16
    at::Tensor qB,              // [N, K_packed], int32, of 4-bit quantized values
    at::Tensor scaling_factors, // [N, K / group_size], fp16, group_size = 128
    at::Tensor bitmod,          // [N, (K / group_size) / 8], int4 packed in int32 format, each int4 upper 2 bit is always 0, and lower 2 bits indicate the bitmod for this group
    at::Tensor out              // [M, N], fp16, result
);


void razer_gpu_symmetric_512_perf(
    at::Tensor fA,              // [M, K], fp16
    at::Tensor qB,              // [N, K_packed], int32, of 4-bit quantized values
    at::Tensor scaling_factors, // [N, K / group_size], fp16, group_size = 128
    at::Tensor bitmod,          // [N, (K / group_size) / 8], int4 packed in int32 format, each int4 upper 2 bit is always 0, and lower 2 bits indicate the bitmod for this group
    at::Tensor out        // [M, N], fp16, result
);


void razer_gpu_symmetric_256_perf(
    at::Tensor fA,              // [M, K], fp16
    at::Tensor qB,              // [N, K_packed], int32, of 4-bit quantized values
    at::Tensor scaling_factors, // [N, K / group_size], fp16, group_size = 128
    at::Tensor bitmod,          // [N, (K / group_size) / 8], int4 packed in int32 format, each int4 upper 2 bit is always 0, and lower 2 bits indicate the bitmod for this group
    at::Tensor out        // [M, N], fp16, result
);