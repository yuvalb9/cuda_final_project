#include "finalProject_kernel.cuh"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}




__global__ void kernel0( char* lifeData, int worldWidth, 	int worldHeight, char* resultLifeData) 
{
	int worldSize = worldWidth * worldHeight;
	int colors = 16;
	//int cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	for (int cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		
		cellId < worldSize;
		cellId += blockDim.x * gridDim.x) {
		
		
		
		int x = cellId % worldWidth;									// x=0
		int yAbs = cellId - x;											// yabs = 0
		int xLeft = (x + worldWidth - 1) % worldWidth;					// xleft=3
		int xRight = (x + 1) % worldWidth;								// xright=1
		int yAbsUp = (yAbs + worldSize - worldWidth) % worldSize;		// yabsup=12
		int yAbsDown = (yAbs + worldWidth) % worldSize;					// yabsdown=4

		char currCellColor = lifeData[x + yAbs];
		char nextColor = (currCellColor + 1) % colors;

		if ((lifeData[xLeft + yAbsUp] == nextColor) || (lifeData[x + yAbsUp] == nextColor) || (lifeData[xRight + yAbsUp] == nextColor) || (lifeData[xLeft+yAbsDown] == nextColor) ||
			(lifeData[x+yAbsDown] == nextColor) || (lifeData[xRight + yAbsDown] == nextColor) || (lifeData[xLeft + yAbs] == nextColor) || (lifeData[xRight+yAbs] == nextColor))
		{
			resultLifeData[x + yAbs] = nextColor;
		}
		else
		{
			resultLifeData[x + yAbs] = currCellColor;
		}

	}
	
}



void  reduce(int boardHeight, int boardWidth, int numThreads, int numBlocks, char* d_idata, char* d_odata, int epochs)
{
	char* temp;
	for (size_t i = 0; i < epochs; i++) {
		cudaDeviceSynchronize();
		kernel0 << <numBlocks, numThreads >> >(d_idata, boardHeight, boardWidth, d_odata);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		std::swap(d_idata, d_odata);
		//temp = d_odata;
		//d_odata = d_idata;
		//d_idata = temp;
		
	}
}


