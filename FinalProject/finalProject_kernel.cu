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
	__device__ inline operator char *()
	{
		extern __shared__ char __smem[];
		return (char *)__smem;
	}

	__device__ inline operator const char *() const
	{
		extern __shared__ char __smem[];
		return (char *)__smem;
	}
};

// kernel 5 + some more bit wise operations...
__global__ void kernel6(char* lifeData, int worldWidth, int worldHeight, char* resultLifeData)
{
	int worldSize = worldWidth * worldHeight;
	int colors = 16;

	char *sdata = SharedMemory();

	for (int cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		cellId < worldSize;
		cellId += blockDim.x * gridDim.x)
	{
		int x = cellId & (worldWidth - 1);									// x=0
		int yAbs = cellId - x;											// yabs = 0
		int xLeft = (x + worldWidth - 1) & (worldWidth - 1);					// xleft=3
		int xRight = (x + 1) & (worldWidth - 1);								// xright=1
		int yAbsUp = (yAbs + worldSize - worldWidth) & (worldSize - 1);		// yabsup=12
		int yAbsDown = (yAbs + worldWidth) & (worldSize - 1);					// yabsdown=4
		int mult = (threadIdx.x << 1) + threadIdx.x;
		// load left neighbors to SM
		sdata[mult + 0] = lifeData[xLeft + yAbsUp];
		sdata[mult + 1] = lifeData[xLeft + yAbs];
		sdata[mult + 2] = lifeData[xLeft + yAbsDown];

		// if last thread - load 3 from current col and 3 from right
		if (threadIdx.x == blockDim.x - 1)
		{
			sdata[mult + 3] = lifeData[x + yAbsUp];
			sdata[mult + 4] = lifeData[x + yAbs];
			sdata[mult + 5] = lifeData[x + yAbsDown];

			sdata[mult + 6] = lifeData[xRight + yAbsUp];
			sdata[mult + 7] = lifeData[xRight + yAbs];
			sdata[mult + 8] = lifeData[xRight + yAbsDown];
		}

		__syncthreads();

		// now we are ready to work.

		// go to IF, and check neighbors in SM, and output to global memory.

		int currCellLocInSData = 4 + mult;
		char currCellColor = sdata[currCellLocInSData];
		//char nextColor = (currCellColor + 1) % colors;
		char nextColor = (currCellColor + 1) & (colors - 1);
		if (((sdata[currCellLocInSData - 4] ^ nextColor) == 0) ||
			((sdata[currCellLocInSData - 3] ^ nextColor) == 0) ||
			((sdata[currCellLocInSData - 2] ^ nextColor) == 0) ||
			((sdata[currCellLocInSData - 1] ^ nextColor) == 0) ||
			((sdata[currCellLocInSData + 1] ^ nextColor) == 0) ||
			((sdata[currCellLocInSData + 2] ^ nextColor) == 0) ||
			((sdata[currCellLocInSData + 3] ^ nextColor) == 0) ||
			((sdata[currCellLocInSData + 4] ^ nextColor) == 0))
		{
			resultLifeData[x + yAbs] = nextColor;
		}
		else
		{
			resultLifeData[x + yAbs] = currCellColor;
		}
		
	}
}



// no % operator
__global__ void kernel5(char* lifeData, int worldWidth, int worldHeight, char* resultLifeData)
{
	int worldSize = worldWidth * worldHeight;
	int colors = 16;

	char *sdata = SharedMemory();

	for (int cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		cellId < worldSize;
		cellId += blockDim.x * gridDim.x)
	{
		//int x = cellId % worldWidth;									// x=0
		int x = cellId & (worldWidth - 1);									// x=0
		int yAbs = cellId - x;											// yabs = 0
		//int xLeft = (x + worldWidth - 1) % worldWidth;					// xleft=3
		int xLeft = (x + worldWidth - 1) & (worldWidth - 1);					// xleft=3
		//int xRight = (x + 1) % worldWidth;								// xright=1
		int xRight = (x + 1) & (worldWidth - 1);								// xright=1
		//int yAbsUp = (yAbs + worldSize - worldWidth) % worldSize;		// yabsup=12
		int yAbsUp = (yAbs + worldSize - worldWidth) & (worldSize - 1);		// yabsup=12
		//int yAbsDown = (yAbs + worldWidth) % worldSize;					// yabsdown=4
		int yAbsDown = (yAbs + worldWidth) & (worldSize - 1);					// yabsdown=4

		// load left neighbors to SM
		sdata[threadIdx.x * 3 + 0] = lifeData[xLeft + yAbsUp];
		sdata[threadIdx.x * 3 + 1] = lifeData[xLeft + yAbs];
		sdata[threadIdx.x * 3 + 2] = lifeData[xLeft + yAbsDown];

		// if last thread - load 3 from current col and 3 from right
		if (threadIdx.x == blockDim.x - 1)
		{
			sdata[threadIdx.x * 3 + 3] = lifeData[x + yAbsUp];
			sdata[threadIdx.x * 3 + 4] = lifeData[x + yAbs];
			sdata[threadIdx.x * 3 + 5] = lifeData[x + yAbsDown];

			sdata[threadIdx.x * 3 + 6] = lifeData[xRight + yAbsUp];
			sdata[threadIdx.x * 3 + 7] = lifeData[xRight + yAbs];
			sdata[threadIdx.x * 3 + 8] = lifeData[xRight + yAbsDown];
		}

		__syncthreads();

		// now we are ready to work.

		// go to IF, and check neighbors in SM, and output to global memory.

		int currCellLocInSData = 4 + threadIdx.x * 3;
		char currCellColor = sdata[currCellLocInSData];
		//char nextColor = (currCellColor + 1) % colors;
		char nextColor = (currCellColor + 1) & (colors - 1);

		if ((sdata[currCellLocInSData - 4] == nextColor) ||
			(sdata[currCellLocInSData - 3] == nextColor) ||
			(sdata[currCellLocInSData - 2] == nextColor) ||
			(sdata[currCellLocInSData - 1] == nextColor) ||
			(sdata[currCellLocInSData + 1] == nextColor) ||
			(sdata[currCellLocInSData + 2] == nextColor) ||
			(sdata[currCellLocInSData + 3] == nextColor) ||
			(sdata[currCellLocInSData + 4] == nextColor))
		{
			resultLifeData[x + yAbs] = nextColor;
		}
		else
		{
			resultLifeData[x + yAbs] = currCellColor;
		}

	}
}


// shared memory + less registers
__global__ void kernel4(char* lifeData, int worldWidth, int worldHeight, char* resultLifeData)
{

	__shared__  char sdata[1024 * 3 + 6];

	for (int cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		cellId < __mul24(worldWidth , worldHeight);
		cellId += blockDim.x * gridDim.x)
	{
		#define WORLD_SIZE (worldWidth * worldHeight)
		#define X (cellId % worldWidth)
		#define yAbs  (cellId - X)
		#define xLeft  ((X + worldWidth - 1) % worldWidth)
		#define xRight ((X + 1) % worldWidth)
		#define yAbsUp  ((yAbs + WORLD_SIZE - worldWidth) % WORLD_SIZE)
		#define yAbsDown ( (yAbs + worldWidth) % WORLD_SIZE)
		#define currCellColor (lifeData[X + yAbs])
		#define nextColor ((currCellColor + 1) % 16)

		// load left neighbors to SM
		sdata[(threadIdx.x << 1) + threadIdx.x + 0] = lifeData[xLeft + yAbsUp];
		sdata[(threadIdx.x << 1) + threadIdx.x + 1] = lifeData[xLeft + yAbs];
		sdata[(threadIdx.x << 1) + threadIdx.x + 2] = lifeData[xLeft + yAbsDown];

		// if last thread - load 3 from current col and 3 from right
		if (threadIdx.x == blockDim.x - 1)
		{
			sdata[(threadIdx.x << 1) + threadIdx.x + 3] = lifeData[X + yAbsUp];
			sdata[(threadIdx.x << 1) + threadIdx.x + 4] = lifeData[X + yAbs];
			sdata[(threadIdx.x << 1) + threadIdx.x + 5] = lifeData[X + yAbsDown];

			sdata[(threadIdx.x << 1) + threadIdx.x + 6] = lifeData[xRight + yAbsUp];
			sdata[(threadIdx.x << 1) + threadIdx.x + 7] = lifeData[xRight + yAbs];
			sdata[(threadIdx.x << 1) + threadIdx.x + 8] = lifeData[xRight + yAbsDown];
		}

		__syncthreads();

		// now we are ready to work.

		// go to IF, and check neighbors in SM, and output to global memory.

		#define currCellLocInSData  (4 + threadIdx.x * 3)
		//char currCellColor = sdata[currCellLocInSData];

		if ((sdata[currCellLocInSData - 4] == nextColor) ||
			(sdata[currCellLocInSData - 3] == nextColor) ||
			(sdata[currCellLocInSData - 2] == nextColor) ||
			(sdata[currCellLocInSData - 1] == nextColor) ||
			(sdata[currCellLocInSData + 1] == nextColor) ||
			(sdata[currCellLocInSData + 2] == nextColor) ||
			(sdata[currCellLocInSData + 3] == nextColor) ||
			(sdata[currCellLocInSData + 4] == nextColor))
		{
			resultLifeData[X + yAbs] = nextColor;
		}
		else
		{
			resultLifeData[X + yAbs] = sdata[(4 + threadIdx.x * 3)];
		}

	}
#undef X
#undef yAbs
#undef xLeft
#undef xRight
#undef yAbsUp
#undef yAbsDown
#undef currCellColor
#undef nextColor
#undef currCellLocInSData  
#undef WORLD_SIZE
}


// shared memory - danny!
__global__ void kernel3(char* lifeData, int worldWidth, int worldHeight, char* resultLifeData)
{
	int worldSize = worldWidth * worldHeight;
	int colors = 16;

	char *sdata = SharedMemory();

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
		sdata[threadIdx.x * 3 + 0] = lifeData[xLeft + yAbsUp];
		sdata[threadIdx.x * 3 + 1] = lifeData[xLeft + yAbs];
		sdata[threadIdx.x * 3 + 2] = lifeData[xLeft + yAbsDown];

		// if last thread - load 3 from current col and 3 from right
		if (threadIdx.x == blockDim.x - 1)
		{
			sdata[threadIdx.x * 3 + 3] = lifeData[x + yAbsUp];
			sdata[threadIdx.x * 3 + 4] = lifeData[x + yAbs];
			sdata[threadIdx.x * 3 + 5] = lifeData[x + yAbsDown];

			sdata[threadIdx.x * 3 + 6] = lifeData[xRight + yAbsUp];
			sdata[threadIdx.x * 3 + 7] = lifeData[xRight + yAbs];
			sdata[threadIdx.x * 3 + 8] = lifeData[xRight + yAbsDown];
		}

		__syncthreads();

		// now we are ready to work.

		// go to IF, and check neighbors in SM, and output to global memory.

		int currCellLocInSData = 4 + threadIdx.x * 3;
		char currCellColor = sdata[currCellLocInSData];
		char nextColor = (currCellColor + 1) % colors;

		if ((sdata[currCellLocInSData - 4] == nextColor) ||
			(sdata[currCellLocInSData - 3] == nextColor) ||
			(sdata[currCellLocInSData - 2] == nextColor) ||
			(sdata[currCellLocInSData - 1] == nextColor) ||
			(sdata[currCellLocInSData + 1] == nextColor) ||
			(sdata[currCellLocInSData + 2] == nextColor) ||
			(sdata[currCellLocInSData + 3] == nextColor) ||
			(sdata[currCellLocInSData + 4] == nextColor))
		{
			resultLifeData[x + yAbs] = nextColor;
		}
		else
		{
			resultLifeData[x + yAbs] = currCellColor;
		}

	}
}



/// pointer arithmetics
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
#undef x
#undef yAbs
#undef xLeft
#undef xRight
#undef yAbsUp
#undef yAbsDown
#undef currCellColor
#undef nextColor
}




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



void  reduce(int boardHeight, int boardWidth, int numThreads, int numBlocks, char** d_idata, char** d_odata, int epochs, int kernelId)
{
	char* temp;
	switch (kernelId)
	{
	case 0:
		for (size_t i = 0; i < epochs; i++) {
			cudaDeviceSynchronize();
			kernel0 << <numBlocks, numThreads>> >(*d_idata, boardHeight, boardWidth, *d_odata);
			std::swap(*d_idata, *d_odata);
		}
		break;
	case 1:
		for (size_t i = 0; i < epochs; i++) {
			cudaDeviceSynchronize();
			kernel1 << <numBlocks, numThreads >> >(*d_idata, boardHeight, boardWidth, *d_odata);
			std::swap(*d_idata, *d_odata);
		}
		break;
	case 2:
		for (size_t i = 0; i < epochs; i++) {
			cudaDeviceSynchronize();
			kernel2 << <numBlocks, numThreads>> >(*d_idata, boardHeight, boardWidth, *d_odata);
			std::swap(*d_idata, *d_odata);
		}
		break;
	case 3:
		for (size_t i = 0; i < epochs; i++) {
			cudaDeviceSynchronize();
			kernel3 << <numBlocks, numThreads, 1024 * 3 + 6 >> >(*d_idata, boardHeight, boardWidth, *d_odata);
			std::swap(*d_idata, *d_odata);
		}
		break;
	case 4:
		for (size_t i = 0; i < epochs; i++) {
			cudaDeviceSynchronize();
			kernel4 << <numBlocks, numThreads, 1024 * 3 + 6 >> >(*d_idata, boardHeight, boardWidth, *d_odata);
			std::swap(*d_idata, *d_odata);
		}
		break;
	case 5:
		for (size_t i = 0; i < epochs; i++) {
			cudaDeviceSynchronize();
			kernel5 << <numBlocks, numThreads, 1024 * 3 + 6 >> >(*d_idata, boardHeight, boardWidth, *d_odata);
			std::swap(*d_idata, *d_odata);
		}
		break;
	case 6:
		for (size_t i = 0; i < epochs; i++) {
			cudaDeviceSynchronize();
			kernel6 << <numBlocks, numThreads, 1024 * 3 + 6 >> >(*d_idata, boardHeight, boardWidth, *d_odata);
			std::swap(*d_idata, *d_odata);
		}
		break;
	default:
		break;
	}
	
	checkCudaErrors(cudaDeviceSynchronize());

}


