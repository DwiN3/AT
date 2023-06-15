#include<stdio.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<iostream>

using namespace std;

__global__ void mul(int* a, int* b, int* c, int n)
{
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	float w = 0.0;
	for (int i = 0; i < n; i++) {
		w += a[Row * n + i] * b[i * n + Col];
		c[Row * n + Col] = w;
	}
}

int main() {
	srand(time(NULL));

	int N;
	cout << "Podaj ilosc wierszy i kolumn -> ";
	cin >> N;

	int SIZE = N * N;
	int INT_SIZE = N * N * sizeof(int);

	int** a = new int* [N];
	a[0] = new int[SIZE];
	for (int i = 1; i < N; i++) a[i] = a[0] + i * N;

	int** b = new int* [N];
	b[0] = new int[SIZE];
	for (int i = 1; i < N; i++) b[i] = b[0] + i * N;

	int** c = new int* [N];
	c[0] = new int[SIZE];
	for (int i = 1; i < N; i++) c[i] = c[0] + i * N;

	int* d_a = *a;
	int* d_b = *b;
	int* d_c = *c;

	for (int row = 0; row < N; ++row) {
		for (int col = 0; col < N; ++col) {
			a[0][row * N + col] = rand() % 10 + 1;
			b[0][row * N + col] = rand() % 10 + 1;
		}
	}

	cudaMalloc((void**)&d_a, INT_SIZE);
	cudaMalloc((void**)&d_b, INT_SIZE);
	cudaMalloc((void**)&d_c, INT_SIZE);

	cudaMemcpy(d_a, a[0], INT_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b[0], INT_SIZE, cudaMemcpyHostToDevice);

	dim3 dimGrid(1, 1);
	dim3 dimBlock(N, N);
	mul <<<dimGrid, dimBlock >>> (d_a, d_b, d_c, N);
	cudaMemcpy(c[0], d_c, INT_SIZE, cudaMemcpyDeviceToHost);

	char wybor;
	cout << "Wyswietlic macierze (t - tak, n - nie) -> ";
	cin >> wybor;
	if (wybor == 't') {
		cout << "\nMacierz A:\n";
		for (int y = 0, int x = 0; y < SIZE; y++) {
			cout << a[0][y] << "  ";
			x++;
			if (x == N) {
				x = 0;
				cout << endl;
			}
		}

		cout << "\nMacierz B:\n";
		for (int y = 0, int x = 0; y < SIZE; y++) {
			cout << b[0][y] << "  ";
			x++;
			if (x == N) {
				x = 0;
				cout << endl;
			}
		}

		cout << "\nMacierz C: (Wynikowa)\n";
		for (int y = 0, int x = 0; y < SIZE; y++) {
			cout << c[0][y] << "  ";
			x++;
			if (x == N) {
				x = 0;
				cout << endl;
			}
		}
	}

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	free(a); free(b); free(c);
	return 0;
}