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



struct SharedMemory
{
	__device__ inline operator int *()
	{
		extern __shared__ int __smem[];
		return (int *)__smem;
	}

	__device__ inline operator const int *() const
	{
		extern __shared__ int __smem[];
		return (int *)__smem;
	}
};

// shared memory - danny!
__global__ void kernel3(char* lifeData, int worldWidth, int worldHeight, char* resultLifeData)
{
	int worldSize = worldWidth * worldHeight;
	int colors = 16;

	int *sdata = SharedMemory();

	for (int cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		cellId < worldSize;
		cellId += blockDim.x * gridDim.x)
	{

	}

	//int cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	for (int cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		cellId < worldSize;
		cellId += blockDim.x * gridDim.x)
	{
		int x = cellId % worldWidth;									// x=0
		int yAbs = cellId - x;											// yabs = 0
		int xLeft = (x + worldWidth - 1) % worldWidth;					// xleft=3
		int xRight = (x + 1) % worldWidth;								// xright=1
		int yAbsUp = (yAbs + worldSize - worldWidth) % worldSize;		// yabsup=12
		int yAbsDown = (yAbs + worldWidth) % worldSize;					// yabsdown=4

		// load left neighbors to SM

		sdata[cellId * 3 + 0] = lifeData[xLeft + yAbsUp];
		sdata[cellId * 3 + 1] = lifeData[xLeft + yAbs];
		sdata[cellId * 3 + 2] = lifeData[xLeft + yAbsDown];

		// if last thread - load 3 from current col and 3 from right

		if (cellId == blockDim.x) // - 1 ?
		{
			sdata[cellId * 4 + 0] = lifeData[x + yAbsUp];
			sdata[cellId * 4 + 1] = lifeData[x + yAbs];
			sdata[cellId * 4 + 2] = lifeData[x + yAbsDown];

			sdata[cellId * 5 + 0] = lifeData[xRight + yAbsUp];
			sdata[cellId * 5 + 1] = lifeData[xRight + yAbs];
			sdata[cellId * 5 + 2] = lifeData[xRight + yAbsDown];
		}

		__syncthreads();

		// now we are ready to work.

		// go to IF, and check neighbors in SM, and output to global memory.

		char currCellColor = sdata[cellId * 4];
		char nextColor = (currCellColor + 1) % colors;

		if ((sdata[cellId * 4 - 4] == nextColor) ||
			(sdata[cellId * 4 - 3] == nextColor) ||
			(sdata[cellId * 4 - 2] == nextColor) ||
			(sdata[cellId * 4 - 1] == nextColor) ||
			(sdata[cellId * 4 + 0] == nextColor) ||
			(sdata[cellId * 4 + 1] == nextColor) ||
			(sdata[cellId * 4 + 2] == nextColor) ||
			(sdata[cellId * 4 + 3] == nextColor) ||
			(sdata[cellId * 4 + 4] == nextColor))
		{
			resultLifeData[x + yAbs] = nextColor;
		}
		else
		{
			resultLifeData[x + yAbs] = currCellColor;
		}

	}

}



/// naive approach
__global__ void kernel2(char* lifeData, int worldWidth, int worldHeight, char* resultLifeData)
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

		if ((*(lifeData + xLeft + yAbsUp) == nextColor) || (*(lifeData + x + yAbsUp) == nextColor) || (*(lifeData + xRight + yAbsUp) == nextColor) || (*(lifeData + xLeft + yAbsDown) == nextColor) ||
			(*(lifeData + x + yAbsDown) == nextColor) || (*(lifeData + xRight + yAbsDown) == nextColor) || (*(lifeData + xLeft + yAbs) == nextColor) || (*(lifeData + xRight + yAbs) == nextColor))
		{
			*(resultLifeData + x + yAbs) = nextColor;
		}
		else
		{
			*(resultLifeData + x + yAbs) = currCellColor;
		}

	}

}



// remove all possible variables (we actually dont need  most of them
__global__ void kernel1(char* lifeData, int worldWidth, int worldHeight, char* resultLifeData)
{
#define WORLD_SIZE (worldWidth * worldHeight)

	//int worldSize = worldWidth * worldHeight;
	int colors = 16;
	//int cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	for (int cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		//cellId < worldSize;
		cellId < WORLD_SIZE;
	cellId += blockDim.x * gridDim.x) {


#define x (cellId % worldWidth)
		//		int x = cellId % worldWidth;									// x=0

#define yAbs  (cellId - x)
		//		int yAbs = cellId - x;											// yabs = 0
#define xLeft  ((x + worldWidth - 1) % worldWidth)
		//		int xLeft = (x + worldWidth - 1) % worldWidth;					// xleft=3

#define xRight ((x + 1) % worldWidth)
		//		int xRight = (x + 1) % worldWidth;								// xright=1
		//int yAbsUp = (yAbs + worldSize - worldWidth) % worldSize;		// yabsup=12
#define yAbsUp  ((yAbs + WORLD_SIZE - worldWidth) % WORLD_SIZE)
		//		int yAbsUp = (yAbs + WORLD_SIZE - worldWidth) % WORLD_SIZE;
		//int yAbsDown = (yAbs + worldWidth) % worldSize;					// yabsdown=4
#define yAbsDown ( (yAbs + worldWidth) % WORLD_SIZE)
		//		int yAbsDown = (yAbs + worldWidth) % WORLD_SIZE;					// yabsdown=4

		//char currCellColor = lifeData[x + yAbs];
#define currCellColor (lifeData[x + yAbs])
		//		char nextColor = (currCellColor + 1) % colors;
#define nextColor ((currCellColor + 1) % colors)

		if ((lifeData[xLeft + yAbsUp] == nextColor) || (lifeData[x + yAbsUp] == nextColor) || (lifeData[xRight + yAbsUp] == nextColor) || (lifeData[xLeft + yAbsDown] == nextColor) ||
			(lifeData[x + yAbsDown] == nextColor) || (lifeData[xRight + yAbsDown] == nextColor) || (lifeData[xLeft + yAbs] == nextColor) || (lifeData[xRight + yAbs] == nextColor))
		{
			resultLifeData[x + yAbs] = nextColor;
		}
		else
		{
			resultLifeData[x + yAbs] = currCellColor;
		}

	}

}
#undef x
#undef yAbs
#undef xLeft
#undef xRight
#undef yAbsUp
#undef yAbsDown
#undef currCellColor
#undef nextColor



/// naive approach
__global__ void kernel0(char* lifeData, int worldWidth, int worldHeight, char* resultLifeData)
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

		if ((lifeData[xLeft + yAbsUp] == nextColor) || (lifeData[x + yAbsUp] == nextColor) || (lifeData[xRight + yAbsUp] == nextColor) || (lifeData[xLeft + yAbsDown] == nextColor) ||
			(lifeData[x + yAbsDown] == nextColor) || (lifeData[xRight + yAbsDown] == nextColor) || (lifeData[xLeft + yAbs] == nextColor) || (lifeData[xRight + yAbs] == nextColor))
		{
			resultLifeData[x + yAbs] = nextColor;
		}
		else
		{
			resultLifeData[x + yAbs] = currCellColor;
		}

	}

}



void  reduce(int boardHeight, int boardWidth, int numThreads, int numBlocks, char** d_idata, char** d_odata, int epochs)
{
	char* temp;
	for (size_t i = 0; i < epochs; i++) {
		cudaDeviceSynchronize();
		kernel2 << <numBlocks, numThreads >> >(*d_idata, boardHeight, boardWidth, *d_odata);
		std::swap(*d_idata, *d_odata);
	}
	checkCudaErrors(cudaDeviceSynchronize());

}


