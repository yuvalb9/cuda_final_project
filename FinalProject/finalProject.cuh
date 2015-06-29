#ifndef __FINALPROJECT_H__
#define __FINALPROJECT_H__

#include "Utils.cuh"
#include "finalProject_kernel.cuh"

#define BOARD_WIDTH 4
#define BOARD_HIEGHT 4
#define COLORS 16
#define EPOCHS 2



int main(int argc, char **argv);
bool runTest(int argc, char **argv);
char* calculateCPU(int boardHeight, int boardWidth, int colors, char * h_idata, int epochs);
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads);


#endif