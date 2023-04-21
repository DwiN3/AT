#include <stdio.h>
#include <math.h>
#include "Baza_danych.h"

void Dodaj_dane(struct sOsoba *dodaj, unsigned indeks)
{
    if(dodaj->istnieje == 1)
    {
        printf("Uzytkownik juz zajmuje numer %d\n", indeks);
        return;
    }

    dodaj->istnieje = 1;
    printf("\nOsoba nr: %d\n", indeks);
    printf("Wpisz nazwisko: ");
    scanf("%s", dodaj->nazwisko);
    printf("Wpisz imie: ");
    scanf("%s", dodaj->imie);
    printf("Wpisz plec: ");
    scanf(" %c", &dodaj->plec);
    printf("Wiek: ");
    scanf("%d", &dodaj->wiek);
    printf("Waga(kg): ");
    scanf("%f", &dodaj->waga);
    printf("Wzrost(cm): ");
    scanf("%f", &dodaj->wzrost);
    dodaj->BMI = dodaj->waga / (pow((dodaj->wzrost)/100, 2));
    printf("Uzytkownik o numerze %d zostal dodany pomyslnie\n", indeks);
}

void Wyswietl_dane(const struct sOsoba *wyswietl, unsigned indeks, unsigned pokaz_wszystkich)
{
    if(wyswietl->istnieje == 0 && pokaz_wszystkich == 0)
    {
        printf("Uzytkownik o numerze %d nie istnieje\n", indeks);
        return;
    }

    else if(wyswietl->istnieje == 1)
    {
        printf("\nDane osoby nr: %d\n", indeks);
        printf("Nazwisko: %s\n", wyswietl->nazwisko);
        printf("Imie: %s\n", wyswietl->imie);

        if(wyswietl->plec ==  'k' || wyswietl->plec ==  'K')
        {
            printf("Plec: K\n");
        }
        else if(wyswietl->plec ==  'm' || wyswietl->plec ==  'M')
        {
            printf("Plec: M\n");
        }
        else
        {
            printf("Plec: -\n");
        }

        if(wyswietl->wiek > 0)
        {
            printf("Wiek: %d lat\n", wyswietl->wiek);
        }
        else
        {
            printf("Wiek: -\n");
        }

        if(wyswietl->waga > 0)
        {
            printf("Waga: %.1fkg\n", wyswietl->waga);
        }
        else
        {
            printf("Waga: -\n");
        }

        if(wyswietl->wzrost > 0)
        {
            printf("Wzrost: %.1fcm\n", wyswietl->wzrost);
        }
        else
        {
            printf("Wzrost: -\n");
        }

        if(wyswietl->wiek > 18 && wyswietl->wzrost > 0 && wyswietl->waga > 0)
        {
            printf("Wspolczynik BMI: %.2f\n", wyswietl->BMI);
            BMI_dane(wyswietl->BMI);
        }
        else
        {
            printf("Wspolczynik BMI: -\n");
        }
    }
}


void Usun_dane(struct sOsoba *usun, unsigned indeks, unsigned reset)
{
    if(usun->istnieje == 0 && reset == 0)
    {
        printf("Uzytkownik o numerze %d nie istnieje\n", indeks);
        return;
    }

    usun->istnieje = 0;
    usun->plec = '-';
    usun->wiek = 0;
    usun->waga = 0;
    usun->wzrost = 0;
    usun->BMI = 0;

    if(reset == 0)
    {
        printf("Uzytkownik o numerze %d zostal usuniety pomyslnie\n", indeks);
    }
}

int Zlicz_dane(struct sOsoba *licz)
{
    int licznik=0;
    if(licz->istnieje == 1)
    {
        licznik++;
    }
    return licznik;
}

void BMI_dane(float bmi)
{
    if(bmi < 16)
    {
        printf("Opis: wyglodzenie\n");
    }

    else if(bmi >= 16 && bmi < 16.99)
    {
        printf("Opis: wychudzenie\n");
    }


    else if(bmi >= 17 && bmi < 18.49)
    {
        printf("Opis: niedowaga\n");
    }


    else if(bmi >= 18.50 && bmi < 24.99)
    {
        printf("Opis: wartowsc prawidlowa\n");
    }

    else if(bmi >= 25 && bmi < 29.99)
    {
        printf("Opis: nadwaga");
    }

    else if(bmi >= 30 && bmi < 34.99)
    {
        printf("Opis: 1 stopien otylosci\n");
    }

    else if(bmi >= 35 && bmi < 39.99)
    {
        printf("Opis: 2 stopien otylosci\n");
    }

    else if(bmi >= 40)
    {
        printf("Opis: otylosc skrajna\n");
    }

    else
    {
        printf("Opis: Bledny zakres\n");
    }
}

void Program(struct sOsoba *osoba)
{
    unsigned indeks;
    char komenda;
    unsigned pokaz_wszystkich = 0, reset=0, licznik;

    while(1)
    {
        licznik=0;
        for(unsigned indeks_p=1 ; indeks_p < LIMIT_OSOB ; indeks_p++)
        {
            licznik += Zlicz_dane(&osoba[indeks_p]);
        }
        printf("\nBaza danych (Ilosc zajmowanych miejsc: %d/10)\n", licznik);

        printf("\nDostepne komendy: \nDodawanie uzytkownika                      ->     d\nUsuwanie uzytkownika                       ->     u\nWyswietlenie wybranego uzytkownika         ->     w\nWyswietlanie wszystkich uzytkownikow       ->     v\nResetowanie programu                       ->     r\nZakonczenie dzialania programu             ->     0\n");
        printf("\nPodaj komende -> ");
        scanf(" %c", &komenda);

        if(komenda == 'd') // Dodawanie uzytkownikow
        {
            printf("Podaj numer uzytkownika do dodania -> ");
            scanf("%d", &indeks);
            if(indeks<LIMIT_OSOB && indeks !=0)
            {
                Dodaj_dane(&(osoba)[indeks], indeks);
            }
            else
            {
                printf("Nr poza zasiegiem!!!\n");
            }
        }

        else if(komenda == 'u') // Usuwanie uzytkownikow
        {
            printf("Podaj numer uzytkownika do usuwania -> ");
            scanf("%d", &indeks);
            if(indeks<LIMIT_OSOB && indeks !=0)
            {
                Usun_dane(&osoba[indeks], indeks, reset);
            }
            else
            {
                printf("Nr poza zasiegiem!!!\n");
            }
        }

        else if(komenda == 'w') // Wyswietlanie wybranego uzytkownika
        {
            printf("Podaj numer uzytkownika do wyswietlenia -> ");
            scanf("%d", &indeks);
            if(indeks<LIMIT_OSOB && indeks !=0)
            {
                Wyswietl_dane(&osoba[indeks], indeks, pokaz_wszystkich);
            }
            else
            {
                printf("Nr poza zasiegiem!!!\n");
            }
        }

        else if(komenda == 'v') // Wyswietlanie wszystkich uzytkownikow
        {
            if(licznik > 0)
            {
                printf("Uzytkownicy: \n");
                pokaz_wszystkich = 1;
                for(unsigned indeks_p=1 ; indeks_p < LIMIT_OSOB ; indeks_p++)
                {
                    Wyswietl_dane(&osoba[indeks_p], indeks_p, pokaz_wszystkich);
                }
                pokaz_wszystkich = 0;
            }
            else
            {
                printf("Brak uzytkownikow\n");
            }
        }

        else if(komenda == 'r') // Resetowanie programu
        {
            printf("Aby potwierdzic reset wpisz: 't' \n(aby anulowac 'f')                   -> ");
            scanf(" %c", &komenda);

            if(komenda == 't')
            {
                reset = 1;
                for(unsigned indeks_p=1 ; indeks_p < LIMIT_OSOB ; indeks_p++)
                {
                    Usun_dane(&osoba[indeks_p], indeks_p, reset);
                }
                reset = 0;
                printf("Wszyscy uzytkownicy zostali usunieci\n");
            }
            else
            {
                printf("Anulowano reset\n");
            }
        }

        else if(komenda == '0') // Wylaczanie programu
        {
            printf("Koniec dzialania programu\n");
            break;
        }

        else
        {
            printf("Bledna komenda\n");
        }
        printf("\n----------------------------------------------------\n");
    }
}
