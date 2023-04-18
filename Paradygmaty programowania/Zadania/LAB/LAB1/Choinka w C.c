// Skończone zadanie

#include <stdio.h>

int main() {
  int wysokosc;
  int space, szerokosc, wielkosc;
  char znak;

start:
  printf("Podaj wysokosc choinki -> ");
  scanf("%d", &wysokosc);
  space = 0;
  szerokosc = 0;
  wielkosc = 0;

  if(wysokosc % 2 == 0) znak = '*';
  if(wysokosc % 2 != 0) znak = '#';

counter_space:
  if(space >= wysokosc - wielkosc) goto counter_symbol;
  printf(" ");
  space++;
  goto counter_space;

counter_symbol:
  printf("%c", znak);
  szerokosc++;
  if(szerokosc < (wielkosc * 2 + 1)) goto counter_symbol;

  printf("\n");
  space = 0;
  szerokosc = 0;
  wielkosc++;

  if(wielkosc < wysokosc) goto counter_space;
  if(wysokosc != 7) goto start;

  return 0;
}