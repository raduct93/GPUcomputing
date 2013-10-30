#include "Utils.h"

// The application to run
const int APP_TO_RUN = 1;

// =======================
// Application entry point
// =======================

int main(int argCount, char ** argValues)
{
	switch (APP_TO_RUN)
	{
	case 1:
		return _01_Hello_World(argCount, argValues);

	case 2:
		return _02_Vector_Add(argCount, argValues);

	case 3:
		return _03_1D_Convolution(argCount, argValues);

	default:
		VALIDATE(0, "Error: Unknown application.");
		WAIT_AND_EXIT(1);
	}
}