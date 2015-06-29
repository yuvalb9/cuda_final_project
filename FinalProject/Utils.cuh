

#ifndef __UTILS_H__
#define __UTILS_H__

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <algorithm>
#include <chrono>
#include <stdio.h>
#include <iostream>
#include <fstream>

void printBoard(int boardHeight, int boardWidth, char* board);
void outputBoardToFile(char* board, int boardHeight, int boardWidth, int colors, const char* filePath);
unsigned int nextPow2(unsigned int x);

#endif