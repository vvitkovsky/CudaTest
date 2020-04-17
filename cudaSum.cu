#include "cudaSum.h"
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

static const int gThreadsPerBlock = 256;

__global__ void
vectorAdd(const uint16_t* A, const uint16_t* B, uint16_t* C, uint32_t numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void
vectorAdd(uint16_t* A, const uint16_t* B, int32_t numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        A[i] += B[i];
    }
}

cudaError_t cudaWarpSum(uint16_t* first, uint16_t* second, uint16_t* output, uint32_t num) {
	if (!first || !second || !output)
		return cudaErrorInvalidDevicePointer;

    int blocksPerGrid = (num + gThreadsPerBlock - 1) / gThreadsPerBlock;
    vectorAdd<<<blocksPerGrid, gThreadsPerBlock >>>(first, second, output, num);

	return CUDA(cudaGetLastError());
}

cudaError_t cudaWarpSum(uint16_t* aDst, uint16_t* aSrc, uint32_t num) {
    if (!aDst || !aSrc)
        return cudaErrorInvalidDevicePointer;

    int blocksPerGrid = (num + gThreadsPerBlock - 1) / gThreadsPerBlock;
    vectorAdd<<<blocksPerGrid, gThreadsPerBlock>>>(aDst, aSrc, num);

    return CUDA(cudaGetLastError());
}