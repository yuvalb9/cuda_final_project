#include "finalProject.cuh"
// CUDA Runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>
#include <helper_functions.h>
#include <algorithm>
#include <chrono>



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


int main(int argc, char **argv)
{
	printf("%s Starting...\n\n", argv[0]);

	cudaDeviceProp deviceProp;
	deviceProp.major = 1;
	deviceProp.minor = 0;
	int minimumComputeVersion = 10;


	int dev;

	dev = findCudaDevice(argc, (const char **)argv);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

	if ((deviceProp.major * 10 + deviceProp.minor) >= minimumComputeVersion)
	{
		printf("Using Device %d: %s\n\n", dev, deviceProp.name);
		checkCudaErrors(cudaSetDevice(dev));
	}
	else
	{
		printf("Error: the selected device does not support the minimum compute capability of %d.%d.\n\n",
			minimumComputeVersion / 10, minimumComputeVersion % 10);

		// cudaDeviceReset causes the driver to clean up all state. While
		// not mandatory in normal operation, it is good practice.  It is also
		// needed to ensure correct operation when the application is being
		// profiled. Calling cudaDeviceReset causes all profile data to be
		// flushed before the application exits
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	bool bResult = false;

	bResult = runTest(argc, argv);

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

	printf(bResult ? "Test passed\n" : "Test failed!\n");
}


char* calculateCPU(int boardHeight, int boardWidth, int colors, char * input_data, int epochs)
{
	char * res_data = NULL;
	char * lastEpoch = NULL;
	char * newEpoch = NULL;

	if (epochs > 0)
	{
		lastEpoch = (char *)malloc(boardHeight*boardWidth*sizeof(char));
		newEpoch = (char *)malloc(boardHeight*boardWidth*sizeof(char));

		// for the first time only - copy the input_data to lastEpoch.
		for (int i = 0; i < boardHeight*boardWidth; i++)
		{
			lastEpoch[i] = input_data[i];
		}

		for (int i = 0; i < epochs; i++)
		{
			for (int currRow = 0; currRow < boardHeight; currRow++)
			{
				for (int currCol = 0; currCol < boardWidth; currCol++)
				{
					char currCellColor = lastEpoch[(currRow * boardWidth) + currCol];
					char nextColor = (currCellColor + 1) % colors;

					int NW_ROW = ((boardHeight + currRow - 1) % boardHeight);
					int NW_COL = ((boardWidth + currCol - 1) % boardWidth);
					int NW_POS = NW_ROW*boardWidth + NW_COL;


					int N_ROW = ((boardHeight + currRow - 1) % boardHeight);
					int N_COL = ((boardWidth + currCol ) % boardWidth);
					int N_POS = N_ROW*boardWidth + N_COL;


					int NE_ROW = ((boardHeight + currRow - 1) % boardHeight);
					int NE_COL = ((boardWidth + currCol + 1) % boardWidth);
					int NE_POS = NE_ROW*boardWidth + NE_COL;


					int W_ROW = ((boardHeight + currRow ) % boardHeight);
					int W_COL = ((boardWidth + currCol -1) % boardWidth);
					int W_POS = W_ROW*boardWidth + W_COL;


					int E_ROW = ((boardHeight + currRow ) % boardHeight);
					int E_COL = ((boardWidth + currCol + 1) % boardWidth);
					int E_POS = E_ROW*boardWidth + E_COL;


					int SW_ROW = ((boardHeight + currRow + 1) % boardHeight);
					int SW_COL = ((boardWidth + currCol - 1) % boardWidth);
					int SW_POS = SW_ROW*boardWidth + SW_COL;


					int S_ROW = ((boardHeight + currRow + 1) % boardHeight);
					int S_COL = ((boardWidth + currCol) % boardWidth);
					int S_POS = S_ROW*boardWidth + S_COL;


					int SE_ROW = ((boardHeight + currRow + 1) % boardHeight);
					int SE_COL = ((boardWidth + currCol + 1) % boardWidth);
					int SE_POS = SE_ROW*boardWidth + SE_COL;


					if ((lastEpoch[NW_POS] == nextColor) || (lastEpoch[N_POS] == nextColor) || (lastEpoch[NE_POS] == nextColor) || (lastEpoch[SW_POS] == nextColor) ||
						(lastEpoch[S_POS] == nextColor) || (lastEpoch[SE_POS] == nextColor) || (lastEpoch[W_POS] == nextColor) || (lastEpoch[E_POS] == nextColor))
					{
						newEpoch[(currRow * boardWidth) + currCol] = nextColor;
					}
					else
					{
						newEpoch[(currRow * boardWidth) + currCol] = currCellColor;
					}
				}
			}
			res_data = newEpoch;
			char* temp = lastEpoch;
			lastEpoch = newEpoch;
			newEpoch = temp;
		}

		free(newEpoch);

		//res_data = lastEpoch;
	}
	

	return res_data;
}


char*  generateRandomData(int size, int colors)
{
	char * data = (char *)malloc(size * sizeof(char));
	for (int i = 0; i<size; i++)
	{
		// Keep the numbers in the range of 0 to COLORS-1
		data[i] = i%colors; //((int)(rand() & 0xFF)) % colors;
	}
	return data;
}


bool runTest(int argc, char **argv)
{
	int boardWidth = BOARD_WIDTH;
	int boardHeight = BOARD_HIEGHT;
	int colors = COLORS; 
	int epochs = EPOCHS;

	int maxThreads = 256;  // number of threads per block
	int maxBlocks = 32768;

	long size = boardWidth*boardHeight;


	printf("board size %d X %d\n", boardWidth, boardHeight);
	printf("amount of colors is %d\n", colors);
	printf("amount of EPOCHS is %d\n", epochs);
	

	// create random input data on CPU
	char *h_idata = generateRandomData(size, colors);


	// calculate the CPU result 
	
	auto t_start = std::chrono::high_resolution_clock::now();
	char *cpu_odata =  calculateCPU(boardHeight, boardWidth, colors, h_idata, epochs);
	auto t_end = std::chrono::high_resolution_clock::now();
	printf("Wall clock time passed: %f ms\n", std::chrono::duration<double, std::milli>(t_end - t_start).count());

	//outputBoardToFile(cpu_odata, boardHeight, boardWidth, colors, "C:\\Users\\yuval\\Downloads\1.ppm");
	
	
	int numBlocks = 0;
	int numThreads = 0;
	getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

	// allocate mem for the result on host side
	char *h_odata = (char *)malloc(size*sizeof(char));


	printf("num of  blocks: %d \n", numBlocks);
	printf("num of  threads: %d \n", numThreads);
	

	// allocate device memory and data
	char *d_idata = NULL;
	char *d_odata = NULL;

	checkCudaErrors(cudaMalloc((void **)&d_idata, size*sizeof(char)));
	checkCudaErrors(cudaMalloc((void **)&d_odata, size*sizeof(char)));

	// copy data directly to device memory
	checkCudaErrors(cudaMemcpy(d_idata, h_idata, size*sizeof(char), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_odata, h_idata, size*sizeof(char), cudaMemcpyHostToDevice));

	// warm-up
	//reduce(boardHeight, boardWidth, numThreads, numBlocks, d_idata, d_odata, 1);

	


	//StopWatchInterface *timer = 0;
	//sdkCreateTimer(&timer);

	t_start = std::chrono::high_resolution_clock::now();

	reduce(boardHeight, boardWidth, numThreads, numBlocks, d_idata, d_odata, epochs);

	t_end = std::chrono::high_resolution_clock::now();
	printf("Wall clock time passed: %f ms\n", std::chrono::duration<double, std::milli>(t_end - t_start).count());

	gpuErrchk(cudaMemcpy(h_odata, d_odata, size*sizeof(char), cudaMemcpyDeviceToHost));

	//double reduceTime = sdkGetAverageTimerValue(&timer) * 1e-3;
	//printf("Reduction, Throughput = %.4f GB/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %d, Workgroup = %u\n",
//		1.0e-9 * ((double)(size*sizeof(char))) / reduceTime, reduceTime, size, 1, numThreads);

	// compute reference solution
	//int cpu_result = reduceCPU(h_idata, size);
	gpuErrchk(cudaPeekAtLastError());

	bool isSame = true;
	for (int i = 0; i < size; i++)
	{
		if (cpu_odata[i] != h_odata[i])
		{
			isSame = false;
			break;

		}
	}

	// cleanup
	//sdkDeleteTimer(&timer);
	free(h_idata);
	free(h_odata);

	checkCudaErrors(cudaFree(d_idata));
	checkCudaErrors(cudaFree(d_odata));

	
	//return (gpu_result == cpu_result);
	
	return isSame;
}










////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use 
// we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

	//get device capability, to avoid block/grid size excceed the upbound
	cudaDeviceProp prop;
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));


	threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
	blocks = (n + (threads * 2 - 1)) / (threads * 2);


	if ((float)threads*blocks >(float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
	{
		printf("n is too large, please choose a smaller number!\n");
	}

	if (blocks > prop.maxGridSize[0])
	{
		printf("Grid size <%d> excceeds the device capability <%d>, set block size as %d (original %d)\n",
			blocks, prop.maxGridSize[0], threads * 2, threads);

		blocks /= 2;
		threads *= 2;
	}

	blocks = ( maxBlocks< blocks ? maxBlocks : blocks);
}