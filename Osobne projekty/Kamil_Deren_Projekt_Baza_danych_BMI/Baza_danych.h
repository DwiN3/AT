#ifndef BAZA_DANYCH_H
#define BAZA_DANYCH_H
#define LIMIT_OSOB 11
#define LIMIT_ZNAKOW 20

struct sOsoba
{
    char nazwisko[LIMIT_ZNAKOW];
    char imie[LIMIT_ZNAKOW];
    char istnieje;
    char plec;
    int wiek;
    float waga;
    float wzrost;
    float BMI;
};

void Dodaj_dane(struct sOsoba *dodaj, unsigned indeks);
void Usun_dane(struct sOsoba *usun, unsigned indeks, unsigned reset);
void Wyswietl_dane(const struct sOsoba *wyswietl, unsigned indeks, unsigned pokaz_wszystkich);
void BMI_dane(float bmi);
int Zlicz_dane(struct sOsoba *licz);
void Program(struct sOsoba *osoba);

#endif // BAZA_DANYCH_H
