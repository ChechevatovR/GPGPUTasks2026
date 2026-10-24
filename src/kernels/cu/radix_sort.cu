#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// ================== Prefix Sum Build Oracle =====================

__host__ __device__ uint oracle_base_idx(uint level, uint n) {
    switch (level) {
        case 0:
            return 0; // Из input
        case 1:
            return 0; // Из oracle level 0
        case 2:
            return n/2;
        case 3:
            return n/2 + n/4;
        case 4:
            return n/2 + n/4 + n/8;
        case 5:
            return n/2 + n/4 + n/8 + n/16;
        case 6:
            return n/2 + n/4 + n/8 + n/16 + n/32;
        case 7:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 ;
        case 8:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128;
        case 9:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256;
        case 10:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512;
        case 11:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024;
        case 12:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048;
        case 13:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096;
        case 14:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192;
        case 15:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384;
        case 16:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768;
        case 17:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 ;
        case 18:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072;
        case 19:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144;
        case 20:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 ;
        case 21:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576;
        case 22:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152;
        case 23:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304;
        case 24:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304 + n/8388608;
        case 25:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304 + n/8388608 + n/16777216;
        case 26:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304 + n/8388608 + n/16777216 + n/33554432;
        case 27:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304 + n/8388608 + n/16777216 + n/33554432 + n/67108864;
        case 28:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304 + n/8388608 + n/16777216 + n/33554432 + n/67108864 + n/134217728;
        case 29:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304 + n/8388608 + n/16777216 + n/33554432 + n/67108864 + n/134217728 + n/268435456;
        case 30:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304 + n/8388608 + n/16777216 + n/33554432 + n/67108864 + n/134217728 + n/268435456 + n/536870912;
        case 31:
            return n/2 + n/4 + n/8 + n/16 + n/32 + n/64 + n/128 + n/256 + n/512 + n/1024 + n/2048 + n/4096 + n/8192 + n/16384 + n/32768 + n/65536 + n/131072 + n/262144 + n/524288 + n/1048576 + n/2097152 + n/4194304 + n/8388608 + n/16777216 + n/33554432 + n/67108864 + n/134217728 + n/268435456 + n/536870912 + n/1073741824;
        default:
            curassert(false, 754743);
    }
    return 0;
}

__global__ void prefix_sum_build_oracle(
    const uint* input,
    const uint input_n,
    uint* dst,
    const uint dst_n
) {
    __shared__ uint input_small[2 * GROUP_SIZE_REDUX];

    const uint input_index = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
    const uint dst_index = (blockIdx.x * blockDim.x) * 1 + threadIdx.x;

    // Берем блок из 64 элементов в инпуте
    // Закидываем его в shared
    input_small[threadIdx.x] = (input_index < input_n) ? input[input_index] : 0;
    input_small[threadIdx.x + GROUP_SIZE_REDUX] = (input_index + GROUP_SIZE_REDUX < input_n) ? input[input_index + GROUP_SIZE_REDUX] : 0;

    __syncthreads();

    if (dst_index < dst_n) {
        // Два соседних индекса суммируем -- следующий уровень дерева
        dst[dst_index] = input_small[2 * threadIdx.x] + input_small[2 * threadIdx.x + 1];
    }
}

// =========================== Prefix Sum Actual Compute ==========================

__global__ void prefix_sum_prefix_accumulation(
    const uint* input,
    const uint* oracle,
          uint* output,
    uint n
) {
    const uint index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= n) {
        return;
    }

    uint mask = 1 << 30;
    uint base = 0;
    uint accum = 0;
    for (uint level = 30; level > 0; level--) {
        if (index & mask) {
            uint base_adj = base >> level;
            uint oracle_idx = oracle_base_idx(level, n) + base_adj;
            accum += oracle[oracle_idx];
            base += mask;
        }
        mask >>= 1;
    }
    if (index & 1) {
        accum += input[base];
    }
    accum += input[index];
    output[index] = accum;
}

// ============================== Map ==============================

__global__ void mapscan(
    const uint* input,
          uint* output,
    uint pos,
    uint n
) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const uint digit = (input[idx] >> pos) & (BUCKETS - 1);

    for (uint bucket = 0; bucket < BUCKETS; bucket++) {
        output[bucket * n + idx] = (bucket == digit) ? 1 : 0;
    }
}

// ============================ Scatter ============================
__global__ void scatter(
    const uint* input,
    const uint* scanned,
          uint* output,
    uint n,
    uint pos
) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const uint digit = (input[idx] >> pos) & (BUCKETS - 1);
    uint offset = 0;
    for (uint bucket = 0; bucket < digit; bucket++) {
        offset += scanned[bucket * n + (n - 1)];
    }
    output[offset + scanned[digit * n + idx] - 1u] = input[idx];
}

// ============================== Host ==============================

namespace cuda {

void prefix_sum(
        const gpu::gpu_mem_32u &input,
        uint input_offset,
        gpu::gpu_mem_32u &output,
        uint output_offset,
        gpu::gpu_mem_32u &oracle,
        uint n
) {
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    uint input_n = n;
    uint src_n = n;
    uint dst_n = (src_n + 1) / 2;
    uint level = 0;

    dim3 gridDim;
    dim3 blockDim;
    blockDim.x = GROUP_SIZE_REDUX;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.y = 1;
    gridDim.z = 1;
    gridDim.x = (dst_n + GROUP_SIZE_REDUX - 1) / GROUP_SIZE_REDUX;

    ::prefix_sum_build_oracle<<<gridDim, blockDim, 0, stream>>>(
            input.cuptr() + input_offset,
            src_n,
            oracle.cuptr(),
            dst_n
    );
    src_n /= 2;
    dst_n = (src_n + 1) / 2;
    level++;

    while (src_n > 1) {
        gridDim.x = (dst_n + GROUP_SIZE_REDUX - 1) / GROUP_SIZE_REDUX;

        ::prefix_sum_build_oracle<<<gridDim, blockDim, 0, stream>>>(
                oracle.cuptr() + oracle_base_idx(level, n),
                src_n,
                oracle.cuptr() + oracle_base_idx(level + 1, n),
                dst_n
        );

        src_n /= 2;
        dst_n /= 2;
        level++;
    }

    // Расчет префиксных сумм по оракулу
    blockDim.x = GROUP_SIZE;
    gridDim.x = (n + GROUP_SIZE - 1) / GROUP_SIZE;

    ::prefix_sum_prefix_accumulation<<<gridDim, blockDim, 0, stream>>>(
            input.cuptr() + input_offset,
            oracle.cuptr(),
            output.cuptr() + output_offset,
            n
    );

    CUDA_CHECK_KERNEL(stream);
}

gpu::gpu_mem_32u radix_sort_do_it(
        const gpu::gpu_mem_32u &input,
        uint n
) {
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    gpu::gpu_mem_32u work_a(n);
    gpu::gpu_mem_32u work_b(n);
    gpu::gpu_mem_32u flags(BUCKETS * n);
    gpu::gpu_mem_32u scanned(BUCKETS * n);
    gpu::gpu_mem_32u oracle(n);

//     work_a.fill(460235);
//     work_b.fill(598719);

    input.copyToN(work_a, n);

    dim3 gridDim;
    dim3 blockDim;
    blockDim.x = GROUP_SIZE;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.y = 1;
    gridDim.z = 1;
    gridDim.x = (n + GROUP_SIZE - 1) / GROUP_SIZE;

    gpu::gpu_mem_32u cur = work_a;
    gpu::gpu_mem_32u next = work_b;

    for (uint pos = 0; pos < 8 * sizeof(uint); pos += WIDTH) {
        ::mapscan<<<gridDim, blockDim, 0, stream>>>(cur.cuptr(), flags.cuptr(), pos, n);

        for (uint bucket = 0; bucket < BUCKETS; bucket++) {
            prefix_sum(flags, bucket * n, scanned, bucket * n, oracle, n);
        }

        ::scatter<<<gridDim, blockDim, 0, stream>>>(cur.cuptr(), scanned.cuptr(), next.cuptr(), n, pos);

        std::swap(cur, next);
    }

    CUDA_CHECK_KERNEL(stream);

    return cur;
}
} // namespace cuda
