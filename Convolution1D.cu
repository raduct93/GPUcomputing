#include "Utils.h"

const int MAX_FILTER_SIZE = 127;

// The filter will also be stored in constant memory
__constant__ float d_cF[MAX_FILTER_SIZE];

// ============
// CUDA kernels
// ============

// The original kernel
__global__ void deviceConvolutionKernel(const float * d_a, const float * d_f, float * d_b, 
	const int aSize, const int fSize)
{
	int tIdx = blockIdx.x * blockDim.x + threadIdx.x;	// Global ID of the current thread
	const int fHalfSize = fSize / 2;					// Half the size of the filter

	// Ignore threads that will not be used
	if (tIdx >= aSize)
	{
		return;
	}

	// Border threads just copy the input to the output (filter cannot be applied)
	if ((tIdx < fHalfSize) || (tIdx >= aSize - fHalfSize))
	{
		d_b[tIdx] = d_a[tIdx];
		return;
	}

	// All other threads apply the filter
	float out = 0.0F;

	for (int i = 0; i < fSize; ++i)
	{
		out += d_a[tIdx - fHalfSize + i] * d_f[i];
	}

	// Write the filtered element
	d_b[tIdx] = out;
}

// The improved kernel
__global__ void deviceConvolutionKernelOptimized(const float * d_a, float * d_b, const int aSize, const int fSize)
{
	__shared__ float tile[512 + MAX_FILTER_SIZE - 1];	// Block-shared tile holding array entries	
	const int fHalfSize = fSize >> 1;					// Half the size of the filter
	int tIdx = threadIdx.x;								// Local thread index
	int gBIdx = blockIdx.x * blockDim.x;				// Global block index
	int gIdx = gBIdx + tIdx;							// Global thread index
	
	// The global bounds of the array (all elements in between them are loaded to shared memory)
	int gMinIndex = max(gBIdx - fHalfSize, 0);
	int gMaxIndex = min(gBIdx + blockDim.x + fHalfSize, aSize) - 1;

	// An offset used to align active threads to their respective elements in shared memory
	int offset = gMinIndex + fHalfSize - gBIdx;

	// Load all required elements in shared memory (note the coalesced memory access)
	for (int crtIdx = gMinIndex + tIdx; crtIdx <= gMaxIndex; crtIdx += blockDim.x)
	{
		tile[crtIdx - gMinIndex] = d_a[crtIdx];
	}

	// Wait for the entire tile to be populated
	__syncthreads();

	// Unneeded threads can be ignored
	if (gIdx < aSize)
	{
		if ((gIdx < fHalfSize) || (gIdx >= aSize - fHalfSize))
		{
			// Global edge threads just copy the input to the output
			d_b[gIdx] = d_a[gIdx];
		}
		else
		{
			// Global inner threads apply the filter
			float out = 0.0F;

			for (int i = 0; i < fSize; ++i)
			{
				out += tile[tIdx - offset + i] * d_cF[i];
			}

			d_b[gIdx] = out;
		}
	}
}

// ========================================
// Command-line argument validation wrapper
// ========================================

void validateArgumentsWrapper(int argCount, char ** argValues, int * vGridConf)
{
	const char * vErrMessages[3] = {"Error: The number of threads must be greater than 0.",
									"Error: The array size must be greater than 0.",
									"Error: The filter size must be greater than 0."};

	validateArguments(argCount, 3, argValues, vGridConf, vErrMessages);

	VALIDATE(vGridConf[3] % 2 == 1, "Error: The filter size must be an odd value.");
	VALIDATE(vGridConf[3] <= vGridConf[2], "Error: The filter size cannot exceed the array size.");
	VALIDATE(vGridConf[3] <= MAX_FILTER_SIZE, "Error: The filter size is too big.");
}

// =============================
// Host-based filter application
// =============================

float hostFilter(const float * h_a, const float * h_f, float * h_b, const int aSize, const int fSize)
{
	const int fHalfSize = fSize / 2;
	long long startMoment;
	float pcFreq;

	// Start the timer
	hostTimerStart(&pcFreq, &startMoment);

	for(int i = 0; i < aSize; ++i)
	{
		if ((i >= fHalfSize) && (i < aSize - fHalfSize))
		{
			// An element is filtered only if the entire filter can be applied to it
			float out = 0.0F;
			
			for (int j = 0; j < fSize; ++j)
			{
				out += h_f[j] * h_a[i - fHalfSize + j];
			}

			h_b[i] = out;
		}
		else
		{
			// For border elements we don't apply the filter
			h_b[i] = h_a[i];
		}
	}

	// Return the time in microseconds
	return hostTimerStop(pcFreq, startMoment);
}

// ===============================
// Device-based filter application
// ===============================

float deviceFilter(const float * h_a, const float * h_f, float * h_b, const int aSize, const int fSize, 
	int nBlocks, int nThreadsPerBlock, const int bOptimized)
{
	float * d_a = NULL;	// Device-based copy of input array
	float * d_b = NULL;	// Device-generated output array
	float * d_f = NULL;	// Device-based copy of the filter (unoptimized version only)
	int byteSize = aSize * sizeof(float);
	cudaEvent_t start, stop;
	float time;

	// Create events for measuring elapsed time on the device
	SAFE_CUDA_CALL(cudaEventCreate(&start));
	SAFE_CUDA_CALL(cudaEventCreate(&stop));

	// Allocate and fill the device-based arrays
	generateDeviceData(byteSize, &d_a, h_a, TRUE);
	generateDeviceData(byteSize, &d_b, NULL, FALSE);

	if (bOptimized)
	{
		// The filter is copied to constant memory
		SAFE_CUDA_CALL(cudaMemcpyToSymbol(d_cF, h_f, fSize * sizeof(float)));
	}
	else
	{
		// The filter is copied to the device
		generateDeviceData(fSize * sizeof(float), &d_f, h_f, TRUE);
	}

	// Record the starting moment of the vector addition
	SAFE_CUDA_CALL(cudaEventRecord(start, 0));

	// Launch the kernel (asynchronously)
	if (bOptimized)
	{
		deviceConvolutionKernelOptimized<<<nBlocks, nThreadsPerBlock>>>(d_a, d_b, aSize, fSize);
	}
	else
	{
		deviceConvolutionKernel<<<nBlocks, nThreadsPerBlock>>>(d_a, d_f, d_b, aSize, fSize); 
	}

	// Record the ending moment of the vector addition and synchronized
	SAFE_CUDA_CALL(cudaEventRecord(stop, 0));
	SAFE_CUDA_CALL(cudaEventSynchronize(stop));

	// Calculate the elapsed time
	SAFE_CUDA_CALL(cudaEventElapsedTime(&time, start, stop));

	// Transfer the computed array back to the host
	SAFE_CUDA_CALL(cudaMemcpy(h_b, d_b, byteSize, cudaMemcpyDeviceToHost));

	// Free used resources on the device
	SAFE_CUDA_CALL(cudaEventDestroy(start));
	SAFE_CUDA_CALL(cudaEventDestroy(stop));
	SAFE_CUDA_CALL(cudaFree(d_a));
	SAFE_CUDA_CALL(cudaFree(d_b));
	
	if (d_f != NULL)
	{
		SAFE_CUDA_CALL(cudaFree(d_f));
	}

	return time * 1.0E+3F;
}

// =======================================
// Wrapper over the device kernel launcher
// =======================================

float deviceFilterWrapper(const float * h_a, const float * h_f, float * h_b_d, int aSize, int fSize, 
	int nBlocks, int nThreadsPerBlock, int bOptimized, const float * h_b_h, float hostTime)
{
	float devTime = deviceFilter(h_a, h_f, h_b_d, aSize, fSize, nBlocks, nThreadsPerBlock, bOptimized);

	printf("Device filtering time (%s kernel): %.2f us" NEW_LINE, bOptimized? "optimized" : "simple", devTime);

	// Compute the speed-up between the device and the host
	printf("Speed-up (%s kernel): %.2f" NEW_LINE, bOptimized? "optimized" : "simple", hostTime / devTime);

	// Check if the calculated arrays match
	compareResults(h_b_h, h_b_d, aSize);

	return devTime;
}

// =======================
// Application entry point
// =======================

int _03_1D_Convolution(int argCount, char ** argValues)
{
	int vGridConf[4];
	float * h_a;	// Host-based input array
	float * h_f;	// Host-based input filter
	float * h_b_h;	// Host-based output array generated by the host
	float * h_b_d;	// Host-based output array generated by the device
	float hostTime, devTime1, devTime2;
	
	// Validate all command-line arguments
	validateArgumentsWrapper(argCount, argValues, vGridConf);

	// Generate the host-based data
	generateHostData(vGridConf[2], &h_a, TRUE);
	generateHostData(vGridConf[3], &h_f, TRUE);
	generateHostData(vGridConf[2], &h_b_h, FALSE);
	generateHostData(vGridConf[2], &h_b_d, FALSE);

	printf("Data generation complete." NEW_LINE);

	// Perform the host-based filtering
	hostTime = hostFilter(h_a, h_f, h_b_h, vGridConf[2], vGridConf[3]);

	printf("Host addition time: %.2f us" NEW_LINE, hostTime);

	printf("Will launch (B: %d, T: %d, G: %d, F: %d)" NEW_LINE, 
		vGridConf[0], vGridConf[1], vGridConf[2], vGridConf[3]);

	// Perfrm the device-based filtering
	devTime1 = deviceFilterWrapper(h_a, h_f, h_b_d, vGridConf[2], vGridConf[3], vGridConf[0], vGridConf[1], 
		FALSE, h_b_h, hostTime);

	devTime2 = deviceFilterWrapper(h_a, h_f, h_b_d, vGridConf[2], vGridConf[3], vGridConf[0], vGridConf[1], 
		TRUE, h_b_h, hostTime);

	printf("Device kernel speed-up: %.2f" NEW_LINE, devTime1 / devTime2);

	free(h_a);
	free(h_f);
	free(h_b_h);
	free(h_b_d);

	WAIT_AND_EXIT(0);
}