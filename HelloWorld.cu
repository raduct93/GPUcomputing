#include "Utils.h"

// ===============
// The CUDA kernel
// ===============

__global__ void helloWorldKernel(const int nRequestedGridSize)
{
	// We can only call "printf()" from a kernel if the device supports at least CC 2.0
#if	__CUDA_ARCH__ >= 200

	// blockIdx, threadIdx, blockDim, gridDim - built-in CUDA variables
	int bSize = blockDim.x;							// The size of one block
	int bIdx = blockIdx.x;							// The block index of the current thread
	int tIdx = threadIdx.x;							// The thread index of the current thread
	int gIdx = blockIdx.x * bSize + threadIdx.x;	// The grid index of the current thread

	// Disable the threads which are not needed
	if (gIdx >= nRequestedGridSize)
	{
		return;
	}

	printf("'Hello World!' from thread (TIdx: %d, BIdx: %d, GIdx: %d)" NEW_LINE,
		tIdx + 1, bIdx + 1, gIdx + 1);

#endif
}

// =================
// Run the CUDA grid
// =================

void runDeviceGrid(const int nBlocks, const int nThreads, const int nGridSize)
{
	// Launch the grid asynchronously
	helloWorldKernel<<<nBlocks, nThreads>>>(nGridSize);

	// Wait for the grid to finish
	SAFE_CUDA_CALL(cudaDeviceSynchronize());
}

// =======================
// Application entry point
// =======================

int _01_Hello_World(int argCount, char ** argValues)
{
	int vGridConf[3];
	const char * vErrMessages[2] =	{"Error: The number of threads must be greater than 0.",
									 "Error: The grid size must be greater than 0."};

	// Extract and validate the number of blocks and threads to launch
	validateArguments(argCount, 2, argValues, vGridConf, vErrMessages);

	printf("Starting application (B: %d, T: %d, G: %d):" NEW_LINE, vGridConf[0], vGridConf[1], vGridConf[2]);

	// Launch the CUDA grid
	runDeviceGrid(vGridConf[0], vGridConf[1], vGridConf[2]);

	printf("The application has finished." NEW_LINE);

	WAIT_AND_EXIT(0);
}