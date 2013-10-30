#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

// =============
// Useful macros
// =============

#define NEW_LINE "\r\n"

// Wait for input from the user, then exit the application
#define WAIT_AND_EXIT(exit_code)	do { system("pause"); exit(exit_code); } while (0)

// CUDA call guard with error signaling
#define SAFE_CUDA_CALL(call)		do {											\
	cudaError_t status = (call);													\
	if (status != cudaSuccess)	{													\
		fprintf(stderr, "Call '%s' at '%s':%d failed with error: '%s'" NEW_LINE,	\
			#call, __FILE__, __LINE__, cudaGetErrorString(status));					\
		WAIT_AND_EXIT(1);															\
	} } while (0)

// Parameter validation macro
#define VALIDATE(cond, err_msg)		do {			\
	if (!(cond)) {									\
		fprintf(stderr, "%s%s", err_msg, NEW_LINE);	\
		WAIT_AND_EXIT(1);							\
	} } while (0)

#define TRUE	1
#define FALSE	0

// =============================
// Entry points for applications
// =============================

int _01_Hello_World(int argCount, char ** argValues);		// "Hello, World" application
int _02_Vector_Add(int argCount, char ** argValues);		// Vector addition application
int _03_1D_Convolution(int argCount, char ** argValues);	// 1D convolution application

// =================
// Support functions
// =================

// Validate an array of command-line arguments
void validateArguments(const int argCount, const int expectedArgCount, 
	char ** argValues, int * vValues, const char ** vErrMessages);

// Create and optionally fill a host-based array
void generateHostData(int vSize, float ** vData, int mustFill);
void generateDeviceData(int byteSize, float ** vDevData, const float * vHostData, int copyFromHost);
void compareResults(const float * h_a, const float * h_b, int size);

// Accurate time measurements on the host
void hostTimerStart(float * pcFreq, long long * startMoment);
float hostTimerStop(const float pcFreq, long long startMoment);

#endif	// _COMMON_H_