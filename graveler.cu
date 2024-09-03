/*
 * slightly adapted from https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-example
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "colors.h"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

// how many turns we need in total
constexpr unsigned int MAX_TURNS = 231;
// number of iterations in total
constexpr unsigned long long ITERATIONS = 1'000'000'000;

__global__ void setup_kernel(curandState *state, unsigned int seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // initialize PRNG generator (XORWOW)
    curand_init(seed, id, 0, &state[id]);
}

__global__ void compute_kernel(curandState *state, int n, unsigned int *result) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int maxCount = 0;
    unsigned int x, count;
    curandState localState = state[id];
    for (int i = 0; i < n; ++i) {
        count = 0;
        // iterate over our necessary turn count
        for (int j = 0; j < MAX_TURNS; ++j) {
            // generate a 32 bit random number
            x = curand(&localState);
            // only use lower 4 bits, so possible rolls are: [0, 1, 2, 3]
            // -> we define 0 as our "1-roll"
            count += (x & 0b11) == 0;
        }
        // update maximum
        maxCount = max(maxCount, count);
    }
    state[id] = localState;
    result[id] = maxCount;
}

int main(void) {
    // CUDA grid sizes (can be adjusted depending on GPU)
    constexpr unsigned int threadsPerBlock = 64;
    constexpr unsigned int blockCount = 64;
    constexpr unsigned int totalThreads = threadsPerBlock * blockCount;

    constexpr unsigned int iterationsPerKernel = ITERATIONS / totalThreads;
    std::cout << "Performing " << (iterationsPerKernel * totalThreads) << " simulations" << std::endl;

    curandState *devStates;
    unsigned int maxCount;
    unsigned int *devResults, *hostResults;
    int device;
    struct cudaDeviceProp props;

    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaGetDeviceProperties(&props, device));
    std::cout << "Running on device " << device << " (" << props.name << ')' << std::endl;

    // reserve space for results
    hostResults = new unsigned int[totalThreads];
    CUDA_CALL(cudaMalloc((void **)&devResults, totalThreads * sizeof(unsigned int)));
    CUDA_CALL(cudaMemset(devResults, 0, totalThreads * sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc((void **)&devStates, totalThreads * sizeof(curandState)));

    // initialize PRNG with nondeterministic seed
    setup_kernel<<<64, 64>>>(devStates, std::random_device{}());

    // 🚀
    auto t1 = std::chrono::high_resolution_clock::now();
    compute_kernel<<<64, 64>>>(devStates, iterationsPerKernel, devResults);
    auto t2 = std::chrono::high_resolution_clock::now();

    // gather results
    CUDA_CALL(cudaMemcpy(hostResults, devResults, totalThreads * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < totalThreads; ++i) {
        maxCount = max(maxCount, hostResults[i]);
    }

    std::cout << "Maximum number of \"1\" rolls: " KCYN << maxCount << RST << std::endl;
    std::cout << "Escaped the softlock? " << (maxCount >= 177 ? FGRN("yes") : FRED("no")) << std::endl;

    // time taken
    auto ms_int = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
    std::chrono::duration<double, std::nano> ms_double = t2 - t1;
    std::cout << "Computation (without setup) took " KYEL << ms_double.count() << RST " nanoseconds" << std::endl;

    // free memory
    CUDA_CALL(cudaFree(devStates));
    CUDA_CALL(cudaFree(devResults));
    delete[] hostResults;

    std::exit(0);
}
