#include "razerz_gpu.h"

// changed bit packing method, end2end speedup: 2.0600x

__device__ __forceinline__ float fp42float_v4(unsigned int x, unsigned int mod) {
    // Default case computations
    unsigned int sign_default = x << 31;
    unsigned int exp_man_combined_raw = x & 0xE;
    unsigned int exp_man_default = (0x1f8 + exp_man_combined_raw) << 21;
    unsigned int result_default = (sign_default | exp_man_default);

    
    const unsigned int mod_results[4] = {0x40a00000, 0x41000000, 0xc0a00000, 0xc1000000};
    unsigned int result_special = mod_results[mod];

    unsigned int nz_result = (result_default == 0xbf400000) ? result_special : result_default;

    unsigned int final_result = (result_default == 0x3f400000) ? 0 : nz_result;

    return __uint_as_float(final_result);
}


__device__  __forceinline__ float warp_reduce_sum(float acc) {
    // Use warp shuffle down to accumulate the sum across the warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    return acc; // Now, the first thread in the warp (lane 0) will have the sum
}

#define BLOCK_X 32    // K direction, 32 threads together for one result element
#define BLOCK_Y 1   // M direction, gemv only 1 length, thickness is 1
#define BLOCK_Z 4    // N direction, 4 threads parallel.


__global__ void razer_gpu_symmetric_1024_host(
    const float4 * __restrict__ A,        // [M, K] fp16 activations, row-major
    const uint4  * __restrict__ qB,     // [K_packed, N] int32, of 4-bit quantized values weights, column-major
    const float4 * __restrict__ scaling_factors, // [K / group_size, N] fp16 scaling factors, column-major
    const uint32_t* __restrict__ bitmod,      // [(K / 128) / 8, N] int4 packed in int32 format, each int4 upper 2 bit is always 0, and lower 2 bits indicate the bitmod for this group, column-major
    half* __restrict__ output,    // [M, N] fp16 output, row-major
    const int M,
    const int N,
    const int K_logical
){
    
    __shared__ half A_shared[1024];
    float4* A_shared_float = reinterpret_cast<float4*>(A_shared);

    __shared__ half scaling_factors_shared[32]; 
    float4* scaling_factors_shared_float = reinterpret_cast<float4*>(scaling_factors_shared);

    __shared__ uint32_t bitmod_shared[4]; // one bitmod only oocupies 4 bits, a int32 is enough for 4 groups



    int m = threadIdx.y + blockIdx.y * blockDim.y; // M index of result
    int n = threadIdx.z + blockIdx.z * blockDim.z; // N index of result
    if (m >= M || n >= N){
        return;
    }

    float acc = 0;

    


    for (int k = 0; k < K_logical; k += 1024){
        // load B matrix to shared memory
        uint4 qb = qB[n * K_logical / 32 + k / 32 + threadIdx.x];

        float4 temp = __ldg(&A[m * K_logical / 8 + k / 8 + threadIdx.x + threadIdx.z * 32]);
        __syncthreads();
        A_shared_float[threadIdx.x + threadIdx.z * 32] = temp;
        
        // load scaling factors to shared memory
        if (threadIdx.x == 0){
            scaling_factors_shared_float[threadIdx.z] = scaling_factors[(k /1024 + n * K_logical /1024)]; //   ~ /128 / 8
            int mybitmods = bitmod[(k + (n) * K_logical) / 1024]; //   ~ /128 / 8
            bitmod_shared[threadIdx.z] = mybitmods;
        }
        __syncthreads();
        
        uint32_t myMod = (bitmod_shared[threadIdx.z] >> ((threadIdx.x / 4) * 4)) & 0xF;
        
        float acc_local = 0;
        #pragma unroll
        for (int i = 0; i < 8; i ++){
            uint32_t qb4 = (qb.x >> (i * 4));
            float b = fp42float_v4(qb4, myMod);
            float a = __half2float(A_shared[threadIdx.x * 32 + i]);
            acc_local += a * b;
        }

        #pragma unroll
        for (int i = 0; i < 8; i ++){
            uint32_t qb4 = (qb.y >> (i * 4));
            float b = fp42float_v4(qb4, myMod);
            float a = __half2float(A_shared[threadIdx.x * 32 + i + 8]);
            acc_local += a * b;
        }

        #pragma unroll
        for (int i = 0; i < 8; i ++){
            uint32_t qb4 = (qb.z >> (i * 4));
            float b = fp42float_v4(qb4, myMod);
            float a = __half2float(A_shared[threadIdx.x * 32 + i + 16]);
            acc_local += a * b;
        }

        #pragma unroll
        for (int i = 0; i < 8; i ++){
            uint32_t qb4 = (qb.w >> (i * 4));
            float b = fp42float_v4(qb4, myMod);
            float a = __half2float(A_shared[threadIdx.x * 32 + i + 24]);
            acc_local += a * b;
        }

        acc += acc_local * __half2float(scaling_factors_shared[threadIdx.z * 8 + threadIdx.x / 4]);            

    }

    float total_sum = warp_reduce_sum(acc);

    if (threadIdx.x == 0){
        output[m * N + n] = __float2half(total_sum);
    }
}



torch::Tensor razer_gpu_symmetric(
    at::Tensor fA,              // [B, M, K], fp16
    at::Tensor qB,              // [N, K_packed], int32, of 4-bit quantized values
    at::Tensor scaling_factors, // [N, K / group_size], fp16, group_size = 128
    at::Tensor bitmod          // [N, (K / group_size) / 8], int4 packed in int32 format, each int4 upper 2 bit is always 0, and lower 2 bits indicate the bitmod for this group
){

    // printf("gemm_mod_gpu_v6_symmetric\n");
    fA = fA.contiguous();
    qB = qB.contiguous();
    scaling_factors = scaling_factors.contiguous();

    // Check inputs
    TORCH_CHECK(fA.is_cuda(), "fA must be a CUDA tensor");
    TORCH_CHECK(qB.is_cuda(), "qB must be a CUDA tensor");
    TORCH_CHECK(scaling_factors.is_cuda(), "scaling_factors must be a CUDA tensor");
    TORCH_CHECK(fA.dtype() == torch::kHalf, "fA must be of type torch.float16");
    TORCH_CHECK(qB.dtype() == torch::kInt32, "qB must be of type torch.int32");
    TORCH_CHECK(scaling_factors.dtype() == torch::kHalf, "scaling_factors must be of type torch.float16");

    // Get dimensions
    auto original_shape = fA.sizes();
    auto last_dim = original_shape[original_shape.size() - 1];
    fA = fA.reshape({-1, last_dim});
    int M = fA.size(0);
    int N = qB.size(0);
    int K_packed = qB.size(1);
    int K = K_packed * 8;
    TORCH_CHECK(K % (128) == 0, "K must be a multiple of 128 (alignment)");

    const float4* fA_ptr = reinterpret_cast<const float4*>(fA.data_ptr<at::Half>());
    const uint4* qB_ptr = reinterpret_cast<const uint4*>(qB.data_ptr<int32_t>());
    const float4* scaling_factors_ptr = reinterpret_cast<const float4*>(scaling_factors.data_ptr<at::Half>());
    const uint32_t* bitmod_ptr = reinterpret_cast<const uint32_t*>(bitmod.data_ptr<int32_t>());

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(fA.device());
    at::Tensor output = torch::empty({M, N}, options);

    __half* output_ptr = reinterpret_cast<__half*>(output.data_ptr<at::Half>());

    //                 K-32     M       N
    dim3 block_size(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid_size(1, (M + block_size.y - 1) / block_size.y, (N + block_size.z - 1) / block_size.z);
    razer_gpu_symmetric_1024_host<<<grid_size, block_size>>>(
        fA_ptr,
        qB_ptr,
        scaling_factors_ptr,
        bitmod_ptr,
        output_ptr,
        M,
        N,
        K
    );
    auto output_shape = original_shape.vec();
    output_shape[output_shape.size() - 1] = N;
    output = output.reshape(output_shape);

    return output;
}


void razer_gpu_symmetric_perf(
    at::Tensor fA,              // [M, K], fp16
    at::Tensor qB,              // [N, K_packed], int32, of 4-bit quantized values
    at::Tensor scaling_factors, // [N, K / group_size], fp16, group_size = 128
    at::Tensor bitmod,          // [N, (K / group_size) / 8], int4 packed in int32 format, each int4 upper 2 bit is always 0, and lower 2 bits indicate the bitmod for this group
    at::Tensor out        // [M, N], fp16, result
){


    // Get dimensions
    int M = fA.size(0);
    int N = qB.size(0);
    int K_packed = qB.size(1);
    int K = K_packed * 8;

    const float4* fA_ptr = reinterpret_cast<const float4*>(fA.data_ptr<at::Half>());
    const uint4* qB_ptr = reinterpret_cast<const uint4*>(qB.data_ptr<int32_t>());
    const float4* scaling_factors_ptr = reinterpret_cast<const float4*>(scaling_factors.data_ptr<at::Half>());
    const uint32_t* bitmod_ptr = reinterpret_cast<const uint32_t*>(bitmod.data_ptr<int32_t>());
    half* output_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());

    //                 K-32     M       N
    dim3 block_size(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid_size(1, (M + block_size.y - 1) / block_size.y, (N + block_size.z - 1) / block_size.z);
    razer_gpu_symmetric_1024_host<<<grid_size, block_size>>>(
        fA_ptr,
        qB_ptr,
        scaling_factors_ptr,
        bitmod_ptr,
        output_ptr,
        M,
        N,
        K
    );
}




