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

#define RADIUS 3
#define BLOCK_SIZE 16

__global__ void stencil_1d(int* in, int* out, int N) {
	__shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
	int lindex = threadIdx.x + RADIUS;

	temp[lindex] = in[gindex];
	if (threadIdx.x < RADIUS) {
		temp[lindex - RADIUS] = in[gindex - RADIUS < 0
			? N + gindex - RADIUS : gindex - RADIUS];
		temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE >= N
			? gindex + BLOCK_SIZE - N : gindex + BLOCK_SIZE];
	}
	__syncthreads();
	int result = 0;
	for (int offset = -RADIUS; offset <= RADIUS; offset++) result += temp[lindex + offset];
	out[gindex] = result;
}

void random_init(int* a, int size){
	for (int n = 0; n < size; n++) a[n] = rand() % 10 + 1;
}

int main(void) {
	srand(time(NULL));

	int N;
	cout << "Podaj wielkosc wektora -> ";
	cin >> N;

	int* in, * out;			
	int* d_in, * d_out;		
	int size = N * sizeof(int);

	cudaMalloc((void**)&d_in, size);
	cudaMalloc((void**)&d_out, size);

	in = (int*)malloc(size); random_init(in, N);
	out = (int*)malloc(size);

	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

	stencil_1d << <(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_in, d_out, N);

	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

	cout << endl;
	for (int n = 0, int r = 0, int central = 0; n < N; n++){
		if (r == RADIUS * 2 + 1) {
			r = 0;
			cout << "--------------------------------\n";
		}
		cout << n + 1 << ".	" << "in[" << n << "](" << in[n] << ") = " << "out[" << n << "](" << out[n] << ")\n";
		r++;
		central++;
	}

	free(in); free(out);
	cudaFree(d_in); cudaFree(d_out);
	return 0;
}