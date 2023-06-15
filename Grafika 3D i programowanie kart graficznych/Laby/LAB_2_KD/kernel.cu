#include<stdio.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<iostream>

using namespace std;

__global__ void add(int* a, int* b, int* c, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < size) c[index] = a[index] + b[index];
}

#define TPB 512 

int main() {
	srand(time(NULL));
	int N;
	int M;
	std::cout << "Podaj ilosc wierszy -> ";
	std::cin >> N;
	std::cout << "Podaj ilosc kolumn -> ";
	std::cin >> M;
	int SIZE = N * M;
	int INT_SIZE = N * M * sizeof(int);

	int** a = new int* [N];
	a[0] = new int[M * N];
	for (int i = 1; i < N; i++) a[i] = a[0] + i * N;

	int** b = new int* [N];
	b[0] = new int[M * N];
	for (int i = 1; i < N; i++) b[i] = b[0] + i * N;

	int** c = new int* [N];
	c[0] = new int[M * N];
	for (int i = 1; i < N; i++) c[i] = c[0] + i * N;

	int* d_a = *a;
	int* d_b = *b;
	int* d_c = *c;

	for (int n = 0; n < N*M; ++n) {
			a[0][n] = rand() % 10 + 1;
			b[0][n] = rand() % 10 + 1;	
	}

	cudaMalloc((void**)&d_a, INT_SIZE);
	cudaMalloc((void**)&d_b, INT_SIZE);
	cudaMalloc((void**)&d_c, INT_SIZE);
	cudaMemcpy(d_a, a[0], INT_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b[0], INT_SIZE, cudaMemcpyHostToDevice);

	add << <(SIZE + TPB - 1) / TPB, TPB >> > (d_a, d_b, d_c, SIZE);
	cudaMemcpy(c[0], d_c, INT_SIZE, cudaMemcpyDeviceToHost);

	char wybor;
	std::cout << "Wyswietlic macierze (t - tak, n - nie) -> ";
	std::cin >> wybor;
	if (wybor == 't') {
		std::cout << "\nMacierz A:\n";
		for (int y = 0, int x = 0; y < N * M; y++) {
			std::cout << a[0][y] << "  ";
			x++;
			if (x == N) {
				x = 0;
				std::cout << "\n";
			}
		}

		std::cout << "\nMacierz B:\n";
		for (int y = 0, int x = 0; y < N * M; y++) {
			std::cout << b[0][y] << "  ";
			x++;
			if (x == N) {
				x = 0;
				std::cout << "\n";
			}
		}

		std::cout << "\nMacierz C: (WYNIKOWA)\n";
		for (int y = 0, int x = 0; y < N * M; y++) {
			std::cout << c[0][y] << "  ";
			x++;
			if (x == N) {
				x = 0;
				std::cout << "\n";
			}
		}
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}