
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i<size) y[i] += scale * x[i];
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	float *x, *y, *c;
	float scale = (float)(rand() % 100);
	float *d_x, *d_y;

	// create vectors on host
	x = (float *)malloc(vectorSize*sizeof(float));
	y = (float *)malloc(vectorSize*sizeof(float));
	// copy y vector for post processing verification
	c = (float *)malloc(vectorSize*sizeof(float));

	// initialize vectors on host
	for (int idx = 0; idx < vectorSize; idx++) {
		x[idx] = (float)(rand() % 100);
		y[idx] = (float)(rand() % 100);
		c[idx] = y[idx];
	}

	// allocate memory on device for vectors
	cudaMalloc((void **)&d_x, vectorSize * sizeof(float));
	cudaMalloc((void **)&d_y, vectorSize * sizeof(float));

	// move vectors to device
	cudaMemcpy(d_x, x, vectorSize*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, vectorSize*sizeof(float), cudaMemcpyHostToDevice);

	// print vectors
	printf("\n Pre-SAXPY vectors : \n");
	printf(" scale = %f\n", scale);
	printf(" x = { ");
	for (int i = 0; i < 5; ++i) {
		printf("%3.4f, ", x[i]);
	}
	printf(" ... }\n");
	printf(" y = { ");
	for (int i = 0; i < 5; ++i) {
		printf("%3.4f, ", y[i]);
	}
	printf(" ... }\n");



	// run saxpy
	saxpy_gpu<<<ceil((vectorSize+255)/256), 256>>>(d_x, d_y, scale, vectorSize);


	cudaMemcpy(x, d_x, vectorSize*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(y, d_y, vectorSize*sizeof(float), cudaMemcpyDeviceToHost);


	// print vectors
	printf("\n Post-SAXPY vectors : \n");
	printf(" scale = %f\n", scale);
	printf(" x = { ");
	for (int i = 0; i < 5; ++i) {
		printf("%3.4f, ", x[i]);
	}
	printf(" ... }\n");
	printf(" y = { ");
	for (int i = 0; i < 5; ++i) {
		printf("%3.4f, ", y[i]);
	}
	printf(" ... }\n");



	int errorCount = 0;
	for (int idx = 0; idx < vectorSize; ++idx) {
		if (y[idx] != scale * x[idx] + c[idx]) {
			++errorCount;
			#ifndef DEBUG_PRINT_DISABLE
				std::cout << "Idx " << idx << " expected " << scale * x[idx] + c[idx] 
					<< " found " << y[idx] << " = " << x[idx] << " + " << c[idx] << "\n";
			#endif
		}
	}
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	free(x);
	free(y);
	free(c);
	cudaFree(d_x);
	cudaFree(d_y);



	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i>=pSumSize) return;
	int hits = 0;

	// Setup RNG
	curandState_t rng;
	curand_init(clock64(), i, 0, &rng);

        // Get new random values and confirm hits
	for (int idx = 0; idx < sampleSize; idx++) {
		float x = curand_uniform(&rng);
		float y = curand_uniform(&rng);
		if (int(x*x + y*y) == 0) hits++;
	}	
		
	pSums[i] = hits;

}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i>=reduceSize) return;

        uint64_t reduced_hits = 0;
	// reduce hits by block processing the threads
	// not sure if this causes some hits to be dropped by using int(), but accuracy seems to be consistent
        for (uint64_t idx = i * int(pSumSize/reduceSize); idx < (i + 1)*int(pSumSize/reduceSize); idx++) {
    	    reduced_hits += pSums[idx];
	    
    	}

	// move the total number of hits to a smaller vector
        totals[i] = reduced_hits;
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	uint64_t hitCount = 0;
	std::string str;
	uint64_t *sums = (uint64_t *)malloc(generateThreadCount*sizeof(uint64_t));
	uint64_t *totals = (uint64_t *)malloc(reduceThreadCount*sizeof(uint64_t));
	uint64_t *pSums, *rTotals;
	cudaMalloc((void **)&pSums, generateThreadCount * sizeof(uint64_t));
	cudaMalloc((void **)&rTotals, reduceThreadCount * sizeof(uint64_t));
	cudaMemcpy(pSums, sums, generateThreadCount*sizeof(uint64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(rTotals, totals, reduceThreadCount*sizeof(uint64_t), cudaMemcpyHostToDevice);

	// randomly generate values in unit square, check if they fit in unit circle, increment hit vector element for each thread
	std::cout << "Generating points...\n";
	generatePoints<<<ceil((generateThreadCount+255)/256), 256>>>(pSums, generateThreadCount, sampleSize);

	// reduce the vector size by processing threads in blocks and adding up all the hits per block
	std::cout << "Reducing counts...\n";
	reduceCounts<<<ceil((reduceThreadCount+255)/256), 256>>>(pSums, rTotals, generateThreadCount, reduceSize);

	// move smaller vector of hit values back to host
	cudaMemcpy(totals, rTotals, reduceThreadCount*sizeof(uint64_t), cudaMemcpyDeviceToHost);

	// loop on host is maybe not the most optimal way of doing things but it does work
	for (int idx = 0; idx < reduceSize; idx++) hitCount += totals[idx];

	// calculate estimate
	approxPi = (static_cast<double>(hitCount) / sampleSize) / generateThreadCount;
	approxPi *= 4.0f;

	free(sums);
	free(totals);
	cudaFree(pSums);
	cudaFree(rTotals);

	std::cout << "hitCount: " << hitCount << "\n";

	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}
