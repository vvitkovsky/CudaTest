#include "cudaSum.h"
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const uint16_t* A, const uint16_t* B, uint16_t* C, uint32_t numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

cudaError_t cudaWarpSum(uint16_t* first, uint16_t* second, uint16_t* output, uint32_t num) {
	if (!first || !second || !output)
		return cudaErrorInvalidDevicePointer;

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(first, second, output, num);

	return CUDA(cudaGetLastError());
}