Readme for the "GPU_Workshop" project
=====================================

This project was built using Visual Studio 2012. If you have an older version of Visual Studio (VS 2010 or VS 2008), you should just take the source files (*.cu, *.cpp, *.h) and create a new NVIDIA CUDA project with them. Please note that NVIDIA project and NSIGHT integration work only with VS 2008, VS 2010 or VS 2012.

These are the project files:
- "Start.cpp": this is the entry point; please set the constant "APP_TO_RUN" to either 1, 2 or 3 in order to select which application to run.
- "Utils.h", "Utils.cpp": helper functions and macros used to manage memory, measure time etc.
- "HelloWorld.cu" - the first application, which displays messages from all CUDA threads (please note that a GPU with Compute Capability of at least 2.0 is needed)
- "VectorAdd.cu" - the vector addition application
- "Convolution1D.cu" - the 1D convolution application

Please remember to set the target device the code will be compiled for (the Compute Capability, for short). For this, go to the project's settings, navigate to "CUDA C/C++", then to "Device" and edit the "Code Generation" tab. Some acceptable values are: "compute_10,sm_10", "compute_20,sm_20", "compute_30,sm_30" etc. Note that for VS 2010 or VS 2008 it might be necessary to navigate to other tabs.

Also, please remember to set the appropriate command-line arguments for each project. In particular:

- for the first project ("HelloWorld.cu") you will need 2 arguments: the number of threads per block and the total number of threads to run (i.e. the grid size in threads)
- for the second project ("VectorAdd.cu") you will need 2 arguments: the number of threads per block and the length of the arrays 
- for the third project ("Convolution1D.cu") you will need 3 arguments: the number of threads per block, the length of the input array and the length of the filter.

