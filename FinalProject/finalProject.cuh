#ifndef __FINALPROJECT_H__
#define __FINALPROJECT_H__

#include "Utils.cuh"
#include "finalProject_kernel.cuh"


// total global memory is 536,870,912 => max dim is 23170*23170
#define BOARD_WIDTH 1024	
#define BOARD_HIEGHT 1024
#define COLORS 16
#define EPOCHS 200



int main(int argc, char **argv);
bool runTest(int argc, char **argv);
char* calculateCPU(int boardHeight, int boardWidth, int colors, char * h_idata, int epochs);
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads);


#endif

/*

GeForce 210
=============
totalGlobalMem				536870912
sharedMemPerBlock				16384
regsPerBlock					16384
warpSize						   32
maxThreadsPerBlock				  512
maxThreadsDim			{512, 512, 64}
maxGridSize			 {65535, 65535, 1}
multiProcessorCount					2
maxThreadsPerMultiProcessor		 1024	int
sharedMemPerMultiprocessor		16384	unsigned int
regsPerMultiprocessor			16384	int


*/