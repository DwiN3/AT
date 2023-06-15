
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 50

__global__ void quicksort(int* data, size_t l, size_t r)
{
    int pivot = data[(l + r) / 2];
    int* left_ptr, * right_ptr;
    left_ptr = data + l;
    right_ptr = data + r;


    while (left_ptr <= right_ptr)
    {
        while (*left_ptr < pivot)
        {
            left_ptr++;
        }

        while (*right_ptr > pivot)
        {
            right_ptr--;
        }

        if (left_ptr <= right_ptr)
        {
            int tmp = *right_ptr;
            *right_ptr-- = *left_ptr;
            *left_ptr++ = tmp;
        }
    }

    cudaStream_t s1, s2;
    int rx = right_ptr - data;
    int lx = left_ptr - data;
    if (l < rx)
    {
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        quicksort << < 1, 1, 0, s1 >> > (data, l, rx);
        cudaStreamDestroy(s1);
    }

    if (r > lx)
    {
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
        quicksort << < 1, 1, 0, s2 >> > (data, lx, r);
        cudaStreamDestroy(s2);
    }

}

void randomInts(int* a, size_t n)
{
    size_t i;

    for (i = 0; i < n; i++)
    {
        a[i] = rand() % 100 + 1;
    }
}

int main()
{
    int* in, * out;
    int* d_in;
    size_t size = SIZE * sizeof(int);
    size_t i;

    in = (int*)malloc(size);
    out = (int*)malloc(size);

    cudaMalloc(&d_in, size);

    srand(time(0));
    randomInts(in, SIZE);
    randomInts(out, SIZE);

    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    quicksort << < 1, 1 >> > (d_in, 0, SIZE - 1);
    cudaMemcpy(out, d_in, size, cudaMemcpyDeviceToHost);

    printf("Tablica przed sortowaniem:\n");
    for (i = 0; i < SIZE; i++)
    {
        printf("%d ", in[i]);
    }

    printf("\nTablica po sortowaniu:\n");
    for (i = 0; i < SIZE; i++)
    {
        printf("%d ", out[i]);
    }

    free(in);
    free(out);

    cudaFree(d_in);

    return 0;
}