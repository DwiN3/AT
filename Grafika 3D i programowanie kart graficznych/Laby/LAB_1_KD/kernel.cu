#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <cassert>

__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) c[index] = a[index] + b[index];
}

void random_ints(int* a, int N) {
    int n;
    for (n = 0; n < N; ++n) a[n] = rand();
}

#define N 1000000
#define TPB 512 //threads per block

int main(void) {
    srand(time(NULL));
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    a = (int*)malloc(size); random_ints(a, N);
    b = (int*)malloc(size); random_ints(b, N);
    c = (int*)malloc(size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    add <<< ((N + TPB - 1) / TPB), TPB >>> (d_a, d_b, d_c, N);
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    for (int n = 0 ; n < 50; n++) printf("Wektor nr %d    %d + %d = %d\n", n+1, a[n], b[n], c[n]);
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}