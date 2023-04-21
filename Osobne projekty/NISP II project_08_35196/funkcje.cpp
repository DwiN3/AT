#include "funkcje.h"

void load_file(string menu[], string product[], int price[])
{
    string bufor;
    int line_txt = 1;
    int switch_line = 1;
    int menu_index = 0;
    int product_index = 0;
    int price_index = 0;

    fstream file;
    file.open("menu.txt", ios::in);

    if(file.good()==false)
    {
        cout << "Plik nieistnieje!";
        exit(0);
    }

    while(getline(file, bufor))
    {
        if(line_txt <= 30)
        {
            if(switch_line == 1)
            {
                menu[menu_index] = bufor;
                menu_index++;
                switch_line++;
            }

            else if(switch_line == 2)
            {
                product[product_index] = bufor;
                product_index++;
                switch_line = 1;
            }
        }

        else if(line_txt > 30 && line_txt <= 50)
        {
            menu[menu_index] = bufor;
            menu_index++;
        }

        else if(line_txt > 50)
        {
            price[price_index] = stoi(bufor);
            price_index++;
        }
        line_txt++;
    }
    file.close();
}

void clear_screen()
{
    getchar();
    system("cls");
}

void display_menu(vector<cProduct> *vContainer)
{
    cout << "MENU:\n" << endl;
    cout << "1. Pizza" << endl;
    cout << "2. Napoje zimne" << endl;
    cout << "3. Napoje gorace" << endl;
    cout << "4. Zestawy" << endl;
    cout << "5. Promocje" << endl;

    if((*vContainer).size() == 0)
    {
        cout << "6. Wyjscie\n" << endl;
    }

    else
    {
        cout << "6. Usun produkt" << endl;
        cout << "7. Anuluj zamowienie" << endl;
        cout << "8. Zatwierdz zamowienie\n" << endl;
    }
}

int main_menu_choice()
{
    char x;

    do
    {
        x = getch() - 48;
    } while(x < 1 || x > 8);

    return x;

}

void display_products(string menu[], string product[], int price[], int start, int end)
{
    int licznik = 1;

    for(int i = start; i < end; i++)
    {
        if(i % 2 != 0) SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 6);
        else SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 3);

        cout << licznik << ". " << menu[i] << "  -  " << price[i] << "zl" << endl;

        if(start == 0)
        {
            SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 14);
            cout << endl << product[i] << " " << endl;
        }
        cout << endl;
        licznik++;
    }

    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 15);
}

int product_choice()
{
    int x;
    cout << "Twoj wybor: "; cin >> x;
    return x;
}

void show_discount()
{
    cout << "1. Pizza 50 / 50" << endl;
    cout << "2. Przy zakupie dwoch pizz napoj gratis" << endl;
    cout << "3. Przy zakupie powyzej 100 zl zaplacisz 20% taniej" << endl;
    cout << "4. Wyjscie" << endl << endl;
    cout << "Twoj wybor: ";
}

void display_order(vector<cProduct> *vContainer)
{

    int size = (*vContainer).size();
    float roznica;

    if(size > 0)
    {

        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 14);
        cout << endl << "Twoje zamowienie: " << endl << endl;

        for(int i = 0; i < size; i++)
        {

            if((*vContainer)[i].get_id() > 0)
            {
                if((*vContainer)[i].get_id() % 2 == 0) SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 3);
                else if((*vContainer)[i].get_id() % 2 != 0) SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 2);
            }
            cout << i + 1 << ". ";
            (*vContainer)[i].show_product();
            SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 14);
        }

        roznica = normal_price(vContainer) - end_price(vContainer);

        cout << endl << "Do zaplaty:   " << end_price(vContainer) << "zl";
        if(check_minus_twenty_discount(vContainer) == true)
        {
            SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 10);
            cout << endl << "Cena -20%       Zaoszczedzone: " << roznica << "zl";
        }
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 15);
    }

    cout << endl << endl;

}

void add_product(vector<cProduct> *vContainer, string menu[], int price[], int tab_index)
{

    cProduct temporary;
    int size = (*vContainer).size();

    (*vContainer).push_back(temporary);
    (*vContainer)[size].load_data(menu[tab_index], price[tab_index], 0);

}

void delete_product(vector<cProduct> *vContainer)
{
    int tab_indeks;
    int size = (*vContainer).size();

    cout << endl << endl << "Podaj numer produktu, ktorego chcesz usunac: ";
    cin >> tab_indeks;

    if(tab_indeks <= size && tab_indeks > 0)
    {

        tab_indeks = tab_indeks - 1;

        if((*vContainer)[tab_indeks].get_id() > 0)
        {
            int sets_id;
            int size = (*vContainer).size();
            int same_id = 0;
            int start = 0;

            sets_id = (*vContainer)[tab_indeks].get_id();

            for(int i = 0; i < size; i++)
            {
                if((*vContainer)[i].get_id() == sets_id) same_id++;
            }
            for(int i = 0; i < size; i++) if((*vContainer)[i].get_id() == sets_id)
            {
                start = i;
                break;
            }

            (*vContainer).erase((*vContainer).begin() + start, (*vContainer).begin() + start + same_id);

        }
        else (*vContainer).erase((*vContainer).begin() + tab_indeks);
    }
}

void cancel_order(vector<cProduct> *vContainer)
{

    char x;

    cout << endl << endl << "Czy napewno chcesz anulowac zamowienie?" << endl;
    cout << "1. Tak" << endl;
    cout << "2. Nie" << endl;

    x = getch();

    if(x == '1') (*vContainer).erase((*vContainer).begin(), (*vContainer).end());
}

char confirm_order(vector<cProduct> *vContainer)
{

    char x;

    cout << endl << endl << "Czy napewno chcesz zatwierdzic zamowienie?" << endl;
    cout << "1. Tak" << endl;
    cout << "2. Nie" << endl;

    x = getch();

    return x;

}

void delivery_details(string *city, string *address, string *e_mail, int *phone_numer)
{

    system("cls");

    cout << "Wypelnij dane do dostawy:" << endl << endl;
    cout << "Podaj miasto: "; cin >> *city;
    cout << "Podaj ulice i numer domu: ";cin.ignore(); getline(cin, *address);
    cout << "Podaj e-mail: "; getline(cin, *e_mail);
    cout << "Podaj numer telefonu: "; cin >> *phone_numer;
    cout << endl << endl << "Dziekujemy za zlozenie zamowienia! :)";
}

bool check_discount(vector<cProduct> *vContainer)
{
    int size = (*vContainer).size();

    for(int n = 0; n < size; n++)
    {
        if((*vContainer)[n].get_price() == 0) return true;
    }
    return false;
}

void print_bill(vector<cProduct> *vContainer, const string city, const string address, const string e_mail, const int phone_numer)
{
    clear_screen();
    fstream Rachunek;

    Rachunek.open("Rachunek.txt", ios::out);
    if(Rachunek.good() != true)
    {
        cout << "Problem z plikiem" << endl;
        Rachunek.close();
    }

    int size = (*vContainer).size();
    float roznica;

    cout << "               RACHUNEK: " << endl;
    Rachunek << "               RACHUNEK: " << endl;


    cout << endl << "Twoje zamowienie: " << endl;
    Rachunek << endl << "Twoje zamowienie: " << endl;

    for(int i = 0; i < size; i++)
    {
        cout << i + 1 << ". ";
        Rachunek << i + 1 << ". ";
        cout << (*vContainer)[i].get_product() << " " << (*vContainer)[i].get_price() << "zl" << endl;
        Rachunek << (*vContainer)[i].get_product() << " " << (*vContainer)[i].get_price() << "zl" << endl;
    }

    cout << endl << "Do zaplaty:                    " << end_price(vContainer) << "zl" << endl;
    Rachunek << endl << "Do zaplaty:                    " << end_price(vContainer) << "zl" << endl;

    if(check_minus_twenty_discount(vContainer) == true)
    {
        roznica = normal_price(vContainer) - end_price(vContainer);
        cout << "Cena -20%       Zaoszczedzone: " << roznica << "zl";
        Rachunek << "Cena -20%       Zaoszczedzone: " << roznica << "zl";
    }


    cout << endl << endl << "Dane dostawy:" << endl;
    Rachunek << endl << endl << "Dane dostawy:" << endl;
    cout << "Miasto:            " << city << endl;
    Rachunek << "Miasto:            " << city << endl;
    cout << "Ulica i nr domu:   " << address << endl;
    Rachunek << "Ulica i nr domu:   " << address << endl;
    cout << "E-mail:            " << e_mail << endl;
    Rachunek << "E-mail:            " << e_mail << endl;
    cout << "Nr telefonu:       " << phone_numer << endl;
    Rachunek << "Nr telefonu:       " << phone_numer << endl;
    Rachunek.close();
}

float normal_price(vector<cProduct> *vContainer)
{
    int size = (*vContainer).size();
    float cena = 0;
    if(size > 0)
    {
        for(int i = 0; i < size; i++)
        {
            cena = cena + (*vContainer)[i].get_price();
        }
    }
    return cena;
}

bool check_minus_twenty_discount(vector<cProduct> *vContainer)
{
    if(normal_price(vContainer) <= 100) return false;
    else return true;
}

float end_price(vector<cProduct> *vContainer)
{
    int size = (*vContainer).size();
    float cena = normal_price(vContainer);

    if(size > 0)
    {
        if(check_minus_twenty_discount(vContainer) == true) cena = cena * 0.80;
    }
    return cena;
}

bool check_free_cola_discount(vector<cProduct> *vContainer)
{
    int licznik = 0;
    int support;
    int size = (*vContainer).size();

    if(size >= 2)
    {
        for(int i = 0; i < size; i++)
        {
            support = (*vContainer)[i].get_price();
            if(support >= 18) licznik++;
        }

        if(licznik >= 2) return true;
        else return false;
    }
    else return false;
}

void add_free_cola_discount(vector<cProduct> *vContainer)
{
    cProduct temporary;
    int size = (*vContainer).size();

    (*vContainer).push_back(temporary);
    (*vContainer)[size].load_data("Coca-Cola 0.5l", 0, 0);
}

void remove_cola(vector<cProduct> *vContainer)
{
    int size = (*vContainer).size();

    if(check_discount(vContainer) == true)
    {
        for(int n = 0; n < size; n++)
        {
            if((*vContainer)[n].get_price() == 0)
            {
                (*vContainer).erase(((*vContainer).begin() + n));
            }
        }
    }
}

int display_sets()
{

    int choice;

    cout << "1. Zestaw rodzinny (3 pizze kazda 5 zl taniej + 3 zimne napoje za 0zl)" << endl;
    cout << "2. Zestaw dla dwojga(2 pizze + 2 zimne napoje za 0zl)" << endl;
    cout << "3. Zestaw dla studenta(1 pizza + zimny napoj za 0zl - za okazaniem legitymacji studenckiej)" << endl;
    cout << "4. Wyjscie" << endl;
    cout << endl << "Twoj wybor: "; cin >> choice;

    return choice;

}

void fifty_fifty_pizza(vector<cProduct> *vContainer, string menu[], string products[], int price[])
{

    int choice;
    string name;
    int more_expensive_pizza;

    cout << "Wybierz pierwsza pizze" << endl << endl;
    display_products(menu, products , price, 0, 15);
    cout << "Twoj wybor: "; cin >> choice;
    name = menu[choice-1];
    more_expensive_pizza = price[choice-1];

    system("cls");
    cout << "Wybierz druga pizze" << endl << endl;
    display_products(menu, products , price, 0, 15);
    cout << "Twoj wybor: "; cin >> choice;
    name = name + "  /  " + menu[choice-1];
    if(price[choice-1] > more_expensive_pizza) more_expensive_pizza = price[choice-1];

    getchar();

    cProduct temporary;
    temporary.load_data(name, more_expensive_pizza, 0);
    (*vContainer).push_back(temporary);

}

void family_set(vector<cProduct> *vContainer, string menu[], string products[], int price[], int &id)
{

    int choice;
    id++;

    for(int i = 0; i < 6; i++)
    {

        if(i < 3)
        {

            display_products(menu, products, price, 0 ,15);
            cout << endl << "Pizza nr - " << i +1 << endl;
            cout << "Twoj wybor: "; cin >> choice;
            choice = choice - 1;
            system("cls");

            cProduct temporary;
            temporary.load_data(menu[choice], price[choice] - 5, id);
            (*vContainer).push_back(temporary);

        }

        else
        {

            display_products(menu, products, price, 15 ,25);
            cout << endl << "Napoj nr - " << i - 2 << endl;
            cout << "Twoj wybor: "; cin >> choice;
            choice = choice + 14;
            system("cls");

            cProduct temporary;
            temporary.load_data(menu[choice], 0, id);
            (*vContainer).push_back(temporary);

        }
    }
}

void double_set(vector<cProduct> *vContainer, string menu[], string products[], int price[], int &id)
{

    int choice;
    id++;

    for(int i = 0; i < 4; i++)
    {

        if(i < 2)
        {

            display_products(menu, products, price, 0 ,15);
            cout << endl << "Pizza nr - " << i + 1 << endl;
            cout << "Twoj wybor: "; cin >> choice;
            choice = choice - 1;
            system("cls");

            cProduct temporary;
            temporary.load_data(menu[choice], price[choice], id);
            (*vContainer).push_back(temporary);

        }

        else
        {

            display_products(menu, products, price, 15 ,25);
            cout << endl << "Napoj nr - " << i - 1 << endl;
            cout << "Twoj wybor: "; cin >> choice;
            choice = choice + 14;
            system("cls");

            cProduct temporary;
            temporary.load_data(menu[choice], 0, id);
            (*vContainer).push_back(temporary);

        }

    }

}

void student_set(vector<cProduct> *vContainer, string menu[], string products[], int price[], int &id)
{

    int choice;
    id++;

    display_products(menu, products, price, 0 ,15);
    cout << endl << "Pizza: " << endl;
    cout << "Twoj wybor: "; cin >> choice;
    choice = choice - 1;
    system("cls");

    cProduct temporary;
    temporary.load_data(menu[choice], price[choice], id);
    (*vContainer).push_back(temporary);

    display_products(menu, products, price, 15 ,25);
    cout << endl << "Napoj: " << endl;
    cout << "Twoj wybor: "; cin >> choice;
    choice = choice + 14;
    system("cls");

    temporary.load_data(menu[choice], 0, id);
    (*vContainer).push_back(temporary);

}
