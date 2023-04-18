// Skończone zadanie

#include <stdio.h>

int sum_fun(int a, int b, int *mult) {
  *mult = a * b;
  return a + b;
}

int main() {
  int mult;
  printf("Iloczyn = %d\nSuma    = %d\n", mult, sum_fun(5, 4, &mult));
  return 0;
}