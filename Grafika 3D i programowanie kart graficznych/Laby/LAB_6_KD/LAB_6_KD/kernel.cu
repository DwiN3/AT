#include<stdio.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<iostream>
#include <algorithm>
#include <cmath>

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

using namespace std;

#define TPB 256

__global__ void reduce0(int* idata, int* odata, unsigned int n) {
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	int result = 0;
	while (i < n) {
		result += idata[i];
		i += blockDim.x * gridDim.x;
	}
	sdata[tid] = result;
	__syncthreads();
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}
	if (tid == 0) odata[blockIdx.x] = sdata[0];

}

__global__ void reduce3(int* idata, int* odata, unsigned int n) {
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	int result = 0;
	while (i < n) {
		result += idata[i];
		i += blockDim.x * gridDim.x;
	}
	sdata[tid] = result;
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) odata[blockIdx.x] = sdata[0];

}

__inline__ __device__ int warp_reduce(int val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2) {
		val += __shfl_down(val, offset);
	}
	return val;
}

__inline__ __device__ int block_reduce(int val) {
	extern __shared__ int shared[];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;
	val = warp_reduce(val);
	if (lane == 0) {
		shared[wid] = val;
	}
	__syncthreads();
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
	if (wid == 0) {
		val = block_reduce(val);
	}
	return val;
}

__global__ void reduce6(int* idata, int* odata, unsigned int n) {
	int result = 0;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	while (i < n) {
		result += idata[i];
		i += blockDim.x * gridDim.x;
	}
	result = warp_reduce(result);
	if (threadIdx.x & (warpSize - 1) == 0) atomicAdd(odata, result);
}

void random_init(int* a, int size) {
	for (int n = 0; n < size; n++) a[n] = rand() % 10 + 1;
}

int main(void) {
	//srand(time(NULL));

	int choice;
	cout << "Wpisz recznie czy skorzystaj z szablonu (1 - tak, 0 - nie)\n";
	cout << "Wybor -> ";
	cin >> choice;
	int N;
	int valuesOfN[7] = { 256,1024,4096,16384,65536,262144,423212 };
	if (choice == 1) {
		cout << "Podaj wielkosc wektora -> ";
		cin >> N;
	}
	else {
		cout << "1 - 256, 2 - 1024, 3 - 4096, 4 - 16384, 5 - 65536, 6 - 262144, 7 - 423212\n";
		cout << "Wybor -> ";
		cin >> choice;
		N = valuesOfN[choice - 1];
	}


	int* idata, * odata;
	int* d_idata, * d_odata;
	int size = N * sizeof(int);

	cudaMalloc((void**)&d_idata, size);
	cudaMalloc((void**)&d_odata, size);

	idata = (int*)malloc(size); random_init(idata, N);
	odata = (int*)malloc(size);

	int sum = 0;
	for (int n = 0; n < N; n++) sum += idata[n];

	float time0 = 0, time3 = 0, time6 = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(d_idata, idata, size, cudaMemcpyHostToDevice);

	int GRID_SIZE = N / TPB;

	cudaEventRecord(start);
	reduce0 << <1, TPB >> > (d_idata, d_odata, N);
	cudaEventRecord(stop);
	cudaMemcpy(odata, d_odata, size, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time0, start, stop);

	cudaEventRecord(start);
	reduce3 << <1, TPB >> > (d_idata, d_odata, N);
	cudaEventRecord(stop);
	cudaMemcpy(odata, d_odata, size, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time3, start, stop);

	cudaEventRecord(start);
	reduce6 << <GRID_SIZE, TPB >> > (d_idata, d_odata, N);
	cudaEventRecord(stop);
	cudaMemcpy(odata, d_odata, size, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time6, start, stop);

	cout << "\n\nTime reduce0: " << time0 << "ms";
	cout << "\nTime reduce3: " << time3 << "ms";
	cout << "\nTime reduce6: " << time6 << "ms";
	printf("\nWynik sumy w wersji sekfencyjnej = %d", sum);
	printf("\nWynik sumy po redukcji =	   %d\n", odata[0]);

	free(idata); free(odata);
	cudaFree(d_idata); cudaFree(d_odata);
	return 0;
}