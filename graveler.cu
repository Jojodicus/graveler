/*
 * slightly adapted from https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-example
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include <cpuid.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "colors.h"

// how many turns we have in total
#ifndef MAX_TURNS
#define MAX_TURNS 231
#endif
// how many turns we need to get para-hax'd
#ifndef NEEDED_TURNS
#define NEEDED_TURNS 177
#endif
// number of iterations in total
#ifndef ITERATIONS
#define ITERATIONS 1'000'000'000
#endif
// if CPU should also be used for simulation
#ifndef USE_CPU
#define USE_CPU true
#endif
// how much computation should be offloaded to the GPU
#ifndef OFFLOAD
#define OFFLOAD 0.984
#endif
// number of CPU threads (0 = all available, minus one for the main thread)
#ifndef CPU_THREADS
#define CPU_THREADS 0
#endif

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

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

class ThreadTask {
public:
    unsigned long id, start, end;
    unsigned int maxCount;
    std::mt19937 mt;

    ThreadTask(unsigned long id, unsigned long start, unsigned long end)
        : id{id}, start{start}, end{end}, maxCount{0}
    {
        mt.seed(id + std::random_device{}());
    }

    void operator()() {
        std::uniform_int_distribution<unsigned int> rng{0, 3};

        unsigned int x, count;

        // simulate, analogous to above
        for (unsigned long i = start; i < end; ++i) {
            count = 0;
            for (int j = 0; j < MAX_TURNS; ++j) {
                x = rng(mt);
                count += (x & 0b11) == 0;
            }
            maxCount = max(maxCount, count);
        }
    }
};

int main(void) {
    // CUDA grid sizes (can be adjusted depending on GPU)
    constexpr unsigned int threadsPerBlock = 64;
    constexpr unsigned int blockCount = 64;
    constexpr unsigned int totalThreads = threadsPerBlock * blockCount;

    // split work between CPU and GPU
    constexpr unsigned long iterationsPerKernel = (USE_CPU ? (ITERATIONS * OFFLOAD) : ITERATIONS) / totalThreads;
    constexpr unsigned long gpuIterations = iterationsPerKernel * totalThreads;
#if USE_CPU
    constexpr unsigned long cpuIterations = ITERATIONS - gpuIterations;
    int cpuThreads = CPU_THREADS ? CPU_THREADS : std::thread::hardware_concurrency() - 1;
#endif

    std::cout << "Performing " << ITERATIONS << " simulations: ";
    std::cout << gpuIterations << " GPU";
#if USE_CPU
    std::cout << " + " << cpuIterations << " CPU" << std::endl;
    std::string cpuName;
    cpuName.resize(49);
    uint *cpuInfo = reinterpret_cast<uint*>(cpuName.data());
    for (uint i = 0; i < 3; ++i) {
        __cpuid(0x80000002+i, cpuInfo[i*4], cpuInfo[i*4+1], cpuInfo[i*4+2], cpuInfo[i*4+3]);
    }
    cpuName.assign(cpuName.data()); // correct null terminator
    std::cout << cpuThreads << " threads on " << cpuName;
#endif
    std::cout << std::endl;

    curandState *devStates;
    unsigned int maxCount;
    unsigned int *devResults, *hostResults;
    int device;
    struct cudaDeviceProp props;

    // 🏎️ start CPU now so we have time to do CUDA in the background
    auto t0 = std::chrono::high_resolution_clock::now();
#if USE_CPU
    ThreadTask** tasks = new ThreadTask*[cpuThreads];
    std::thread** threads = new std::thread*[cpuThreads];
    for (unsigned int i = 0; i < cpuThreads; ++i) {
        unsigned long start = i * cpuIterations / cpuThreads;
        unsigned long end = (i + 1) * cpuIterations / cpuThreads;
        tasks[i] = new ThreadTask{i, start, end};
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cpuThreads; ++i) {
        threads[i] = new std::thread{*tasks[i]};
    }
#endif

    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaGetDeviceProperties(&props, device));
    std::cout << "CUDA running on device " << device << " (" << props.name << ')' << std::endl;

    // reserve space for results
    hostResults = new unsigned int[totalThreads];
    CUDA_CALL(cudaMalloc((void **)&devResults, totalThreads * sizeof(unsigned int)));
    CUDA_CALL(cudaMemset(devResults, 0, totalThreads * sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc((void **)&devStates, totalThreads * sizeof(curandState)));

    // initialize PRNG with nondeterministic seed
    setup_kernel<<<64, 64>>>(devStates, std::random_device{}());

    // 🚀
    auto t2 = std::chrono::high_resolution_clock::now();
    compute_kernel<<<64, 64>>>(devStates, iterationsPerKernel, devResults);

    // join
#if USE_CPU
    for (int i = 0; i < cpuThreads; ++i) {
        threads[i]->join();
    }
    auto t3 = std::chrono::high_resolution_clock::now();
#endif
    CUDA_CALL(cudaDeviceSynchronize());
    auto t4 = std::chrono::high_resolution_clock::now();

    // gather results
    CUDA_CALL(cudaMemcpy(hostResults, devResults, totalThreads * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < totalThreads; ++i) {
        maxCount = max(maxCount, hostResults[i]);
    }
#if USE_CPU
    for (int i = 0; i < cpuThreads; ++i) {
        maxCount = max(maxCount, tasks[i]->maxCount);
    }
#endif

    std::cout << "Maximum number of \"1\" rolls: " KCYN << maxCount << RST << std::endl;
    std::cout << "Escaped the softlock? " << (maxCount >= NEEDED_TURNS ? FGRN("yes") : FRED("no")) << std::endl;

    // time taken
    std::chrono::duration<double, std::milli> ms_double;
#if USE_CPU
    ms_double = t3 - t1;
    std::cout << "CPU computation without setup took " KYEL << ms_double.count() << RST " milliseconds" << std::endl;
#endif
    ms_double = t4 - t2;
    std::cout << "GPU computation without setup took " KYEL << ms_double.count() << RST " milliseconds" << std::endl;
    ms_double = t4 - t0;
    std::cout << "Whole interleaved computation (with PRNG setup) took " KYEL << ms_double.count() << RST " milliseconds" << std::endl;

    // free memory
    CUDA_CALL(cudaFree(devStates));
    CUDA_CALL(cudaFree(devResults));
    delete[] hostResults;

    std::exit(0);
}
