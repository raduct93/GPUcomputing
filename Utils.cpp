#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <Windows.h>	// For the performance counters
#endif

#include "Utils.h"

static int bRandomSeedApplied = 0;	// Marks if the seed for the random generator has been set

// ===========================================================
// Use the first available CUDA device and read its properties
// ===========================================================

void getFirstDeviceProperties(struct cudaDeviceProp * devProps)
{
	int devCount = 0;

	memset(devProps, 0, sizeof(struct cudaDeviceProp));

	// Get the number of available CUDA devices
	SAFE_CUDA_CALL(cudaGetDeviceCount(&devCount));
	VALIDATE(devCount > 0, "Error: No CUDA-capable device found.");
	
	// Use the first available device
	SAFE_CUDA_CALL(cudaSetDevice(0));

	// Get the properties of this device
	SAFE_CUDA_CALL(cudaGetDeviceProperties(devProps, 0));
}

// ===========================================
// Validate an array of command-line arguments
// ===========================================

void validateArguments(const int argCount, const int expectedArgCount, 
	char ** argValues, int * vValues, const char ** vErrMessages)
{
	struct cudaDeviceProp devProps;

	// We have (argCount - 1) pairs to process
	VALIDATE(argCount - 1 == expectedArgCount, "Error: The number of arguments is different than expected.");

	// By convention:
	// - vValues[0] holds the number of blocks, which must be calculated 
	// - vValues[1] holds the number of threads per block
	// - vValues[2] holds the size of the grid 
	for (int i = 1; i < argCount; ++i)
	{
		vValues[i] = atoi(argValues[i]);
		VALIDATE(vValues[i] > 0, vErrMessages[i - 1]);
	}

	// Extract the CUDA device properties
	getFirstDeviceProperties(&devProps);

	// Ensure the number of threads per block is valid
	VALIDATE(vValues[1] <= devProps.maxThreadsPerBlock, "Error: Too many threads per block.");

	// Compute the number of needed blocks
	vValues[0] = (vValues[2] + vValues[1] - 1) / vValues[1];

	// Ensure that the number of blocks is valid
	VALIDATE(vValues[0] <= devProps.maxGridSize[0], "Error: Too many blocks.");
}

// =============================================
// Create and optionally fill a host-based array
// =============================================

void generateHostData(int vSize, float ** vData, int mustFill)
{
	// Allocate memory for the array
	* vData = (float *)malloc(vSize * sizeof(float));

	// Ensure the array has been properly allocated
	VALIDATE((* vData) != NULL, "Error: Could not allocate enough memory.");

	// Filling the array is optional
	if (mustFill)
	{
		// Initialize the random generator seed, if necessary
		if (!bRandomSeedApplied)
		{
			bRandomSeedApplied = 1;
			srand((unsigned int)time(NULL));
		}

		// The generated numbers are between 0 and 1
		for (int i = 0; i < vSize; ++i)
		{
			(* vData)[i] = (float)rand() / RAND_MAX;
		}
	}
}

// =================================================
// Allocate and optionally fill a device-based array
// =================================================

void generateDeviceData(int byteSize, float ** vDevData, const float * vHostData, int copyFromHost)
{
	// First we allocate device memory for the device array
	SAFE_CUDA_CALL(cudaMalloc((void **)vDevData, byteSize));

	// If necessary, we also fill the device-based array
	if (copyFromHost && (vHostData != NULL))
	{
		SAFE_CUDA_CALL(cudaMemcpy(* vDevData, vHostData, byteSize, cudaMemcpyHostToDevice));
	}
}

// ================================
// Compare the contents of 2 arrays
// ================================

void compareResults(const float * h_a, const float * h_b, int size)
{
	int match = TRUE;

	for (int i = 0; i < size; ++i)
	{
		if (fabs(h_a[i] - h_b[i]) > 1.0E-5F)
		{
			match = FALSE;
			break;
		}
	}

	printf("Match? %s" NEW_LINE, match? "Yes" : "No");
}

// =====================================
// Measuring time accurately on the host
// =====================================

void hostTimerStart(float * pcFreq, long long * startMoment)
{
#ifdef _WIN32

	LARGE_INTEGER pcParam;

    VALIDATE(QueryPerformanceFrequency(&pcParam), "Error: QueryPerformanceFrequency has failed.");

	// Get frequency for microsecond timer
    * pcFreq = (float)pcParam.QuadPart * 1.0E-6F;

	// Get the starting moment
    QueryPerformanceCounter(&pcParam);
    * startMoment = pcParam.QuadPart;

#endif
}

float hostTimerStop(const float pcFreq, long long startMoment)
{
#ifdef _WIN32

	LARGE_INTEGER pcParam;

	// Get the current moment
    QueryPerformanceCounter(&pcParam);

	// Calculate the time difference
    return (float)(pcParam.QuadPart - startMoment) / pcFreq;

#else

	// Change this if running on a non-Windows OS
	return 0.0F;

#endif
}