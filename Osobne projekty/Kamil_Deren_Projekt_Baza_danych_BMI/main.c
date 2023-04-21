#include <stdio.h>
#include "Baza_danych.h"

int main()
{
    struct sOsoba osoba[LIMIT_OSOB] = {0};
    Program(&osoba[1]);
    return 0;
}
