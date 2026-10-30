#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication() // TODO input/output buffers
{
    // TODO
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize) // TODO input/output buffers
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        // TODO input/output buffers
        // input_buffer.cuptr(),
        );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
