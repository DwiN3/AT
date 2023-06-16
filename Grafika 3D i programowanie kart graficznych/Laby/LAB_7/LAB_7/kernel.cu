#include<stdio.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<iostream>
#include <algorithm>

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

using namespace std;

#define TPB 512

__global__ void reduce0(int* idata, int* odata, unsigned int n) {
	__shared__ int sdata[TPB];
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
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) {
		odata[blockIdx.x] = sdata[0];
	}
}

__global__ void reduce3(int* idata, int* odata, unsigned int n) {
	__shared__ int sdata[TPB];
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
	if (tid == 0) {
		odata[blockIdx.x] = sdata[0];
	}
}

void random_init(int* a, int size) {
	for (int n = 0; n < size; n++) a[n] = rand() % 10 + 1;
}

int main(void) {
	//srand(time(NULL));

	int N = 99999999;
	//cout << "Podaj wielkosc wektora -> ";
	//cin >> N;

	int* vector, * vector_results;
	int* d_vector, * d_vector_results;
	int size = N * sizeof(int);

	cudaMalloc((void**)&d_vector, size);
	cudaMalloc((void**)&d_vector_results, size);

	vector = (int*)malloc(size); random_init(vector, N);
	vector_results = (int*)malloc(size);

	int sum = 0;
	for (int n = 0; n < N; n++) sum += vector[n];
	//for (int n = 0; n < N; n++) cout << vector[n] << " ";

	float time0 = 0, time3 = 0, time6 = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(d_vector, vector, size, cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	reduce0 << <1, TPB >> > (d_vector, d_vector_results, N);
	cudaEventRecord(stop);
	cudaMemcpy(vector_results, d_vector_results, size, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time0, start, stop);

	cudaEventRecord(start);
	reduce3 << <1, TPB >> > (d_vector, d_vector_results, N);
	cudaEventRecord(stop);
	cudaMemcpy(vector_results, d_vector_results, size, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time3, start, stop);

	cout << "\n\nTime reduce0: " << time0 << "ms";
	cout << "\nTime reduce3: " << time3 << "ms";
	cout << "\nWynik sumy w wersji sekfencyjnej = " << sum;
	printf("\nWynik sumy po redukcji %d\n", vector_results[0]);

	free(vector); free(vector_results);
	cudaFree(d_vector); cudaFree(d_vector_results);
	return 0;
}