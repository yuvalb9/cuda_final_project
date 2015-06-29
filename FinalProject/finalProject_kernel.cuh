

#ifndef __FINALPROJECT_KERNEL_H__
#define __FINALPROJECT_KERNEL_H__

// CUDA Runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>
#include <helper_functions.h>

extern "C" void  reduce(int boardHeight, int boardWidth, int numThreads, int numBlocks, char* d_idata, char* d_odata, int epochs);

#endif