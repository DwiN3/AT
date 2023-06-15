#include<stdio.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<iostream>

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

using namespace std;

__global__ void mul_1(float* a, float* b, float* c, int n) {
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	float w = 0.0;

	for (int i = 0; i < n; i++) {
		w += a[Row * n + i] * b[i * n + Col];
		c[Row * n + Col] = w;
	}
}

#define TILE_WIDTH 8

__global__ void mul_2(float* Md, float* Nd, float* Pd, int Width) {
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float w = 0.0;

	for (int k = 0; k < Width / TILE_WIDTH; ++k) {
		Mds[ty][tx] = Md[Row * Width + (k * TILE_WIDTH + tx)];
		Nds[ty][tx] = Nd[Col + (k * TILE_WIDTH + ty) * Width];
		__syncthreads();
		for (int p = 0; p < TILE_WIDTH; ++p) {
			w += Mds[ty][p] * Nds[p][tx];
			__syncthreads();
		}
		Pd[Row * Width + Col] = w;
	}
}

int main() {
	srand(time(NULL));
	cout << "Macierz kwadratowa - 1\nMacierz dostosowana - 2\nWybor: ";
	char wybor;
	cin >> wybor;
	int x_A;
	int y_A;
	int x_B;
	int y_B;

	if (wybor == '1') {
		cout << "\nPodaj wielkosc macierzy -> ";
		cin >> x_A;
		y_A = x_A;
		x_B = x_A;
		y_B = x_A;
	}
	else {
		cout << "\nPodaj ilosc wersow macierzy A -> ";
		cin >> x_A;
		cout << "Podaj ilosc kolumn macierzy A -> ";
		cin >> y_A;
		cout << "Podaj ilosc wersow macierzy B -> ";
		cin >> x_B;
		cout << "Podaj ilosc kolumn macierzy B -> ";
		cin >> y_B;
	}

	if (x_B == y_A) {
		printf("\nA[%d][%d] x B[%d][%d] = C[%d][%d]\n", x_A, y_A, x_B, y_B, x_B, y_A);
		printf("Wartosc TILE_WIDTH = %d\n", TILE_WIDTH);
		int SIZE_A = x_A * y_A;
		int SIZE_B = x_B * y_B;
		int SIZE_C = x_B * y_A;

		int FLOAT_SIZE_A = x_A * y_A * sizeof(float);
		int FLOAT_SIZE_B = x_B * y_B * sizeof(float);
		int FLOAT_SIZE_C = x_B * y_A * sizeof(float);
		int N_ = 0;
		if (x_A > x_B) N_ = x_A;
		else N_ = x_B;

		float** a = new float* [x_A];
		a[0] = new float[x_A * y_A];
		for (int i = 1; i < x_A; i++) a[i] = a[0] + i * x_A;

		float** b = new float* [x_B];
		b[0] = new float[x_B * y_B];
		for (int i = 1; i < x_B; i++) b[i] = b[0] + i * x_B;

		float** c = new float* [x_B];
		c[0] = new float[x_B * y_A];
		for (int i = 1; i < x_B; i++) c[i] = c[0] + i * x_B;

		float* d_a = *a;
		float* d_b = *b;
		float* d_c = *c;

		for (int n = 0; n < SIZE_A; ++n) a[0][n] = rand() % 10 + 1;
		for (int n = 0; n < SIZE_B; ++n) b[0][n] = rand() % 10 + 1;

		cudaMalloc((void**)&d_a, FLOAT_SIZE_A);
		cudaMalloc((void**)&d_b, FLOAT_SIZE_B);
		cudaMalloc((void**)&d_c, FLOAT_SIZE_C);
		cudaMemcpy(d_a, a[0], FLOAT_SIZE_A, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b[0], FLOAT_SIZE_B, cudaMemcpyHostToDevice);

		int accept_to_view = false;
		cout << "\nBez optymalizacji - 1\nZ optymalizacja - 2\nWybor: ";
		char wybor;
		cin >> wybor;
		if (wybor == '1') {
			dim3 dimGrid(1, 1);
			dim3 dimBlock(x_B, y_A);
			mul_1 << <dimGrid, dimBlock >> > (d_a, d_b, d_c, x_B);
			cudaMemcpy(c[0], d_c, FLOAT_SIZE_C, cudaMemcpyDeviceToHost);
			accept_to_view = true;
		}

		if (wybor == '2') {
			if (N_ % TILE_WIDTH == 0) {
				dim3 dimGrid(N_ / TILE_WIDTH, N_ / TILE_WIDTH);
				dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
				mul_2 << <dimGrid, dimBlock >> > (d_a, d_b, d_c, x_B);
				cudaMemcpy(c[0], d_c, FLOAT_SIZE_C, cudaMemcpyDeviceToHost);
				accept_to_view = true;
			}
			else cout << "\nZmien wartosc TITLE_WIDTH\n";
		}

		cout << "\nWyswietlic macierze (t - tak, n - nie) -> ";
		cin >> wybor;
		if (wybor == 't' && accept_to_view == true) {
			cout << "\nMacierz A:\n";
			for (int y = 0, int x = 0; y < SIZE_A; y++) {
				cout << a[0][y] << "  ";
				x++;
				if (x == x_A) {
					x = 0;
					cout << endl;
				}
			}

			cout << "\nMacierz B:\n";
			for (int y = 0, int x = 0; y < SIZE_B; y++) {
				cout << b[0][y] << "  ";
				x++;
				if (x == x_B) {
					x = 0;
					cout << endl;
				}
			}

			cout << "\nMacierz C: (Wynikowa)\n";
			for (int y = 0, int x = 0; y < SIZE_C; y++) {
				cout << c[0][y] << "  ";
				x++;
				if (x == x_A) {
					x = 0;
					cout << endl;
				}
			}
		}

		cout << "\n\nPROGRAM SIE WYKONAL\n";

		cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
		free(a); free(b); free(c);
	}
	else printf("\nA[_][%d] != B[%d][_]\n", y_A, x_B);

	return 0;
}