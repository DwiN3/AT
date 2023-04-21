#ifndef FUNKCJE_H
#define FUNKCJE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <conio.h>
#include <windows.h>

using namespace std;

class cProduct
{
private:

    string name;
    int price;
    int id;

public:

    void load_data(string _name, int _price, int _id)
    {
        name = _name;
        price = _price;
        id = _id;
    }

    void show_product()
    {
        cout << name << "  -  " << price << "zl" << endl;
    }

    int get_price()
    {
        return price;
    }

    string get_product()
    {
        return name;
    }

    int get_id()
    {
        return id;
    }
};


void load_file(string menu[], string product[], int price[]);
void clear_screen();
void display_menu(vector<cProduct> *vContainer);
int main_menu_choice();
void display_products(string menu[], string product[], int price[], int start, int end);
int product_choice();
void show_discount();
void display_order(vector<cProduct> *vContainer);
void add_product(vector<cProduct> *vContainer, string menu[], int price[], int tab_index);
void delete_product(vector<cProduct> *vContainer);
void cancel_order(vector<cProduct> *vContainer);
char confirm_order(vector<cProduct> *vContainer);
void delivery_details(string *city, string *address, string *e_mail, int *phone_numer);
bool check_discount(vector<cProduct> *vContainer);
void print_bill(vector<cProduct> *vContainer, const string city, const string address, const string e_mail, const int phone_numer);
float normal_price(vector<cProduct> *vContainer);
bool check_minus_twenty_discount(vector<cProduct> *vContainer);
float end_price(vector<cProduct> *vContainer);
bool check_free_cola_discount(vector<cProduct> *vContainer);
void add_free_cola_discount(vector<cProduct> *vContainer);
void remove_cola(vector<cProduct> *vContainer);
void fifty_fifty_pizza(vector<cProduct> *vContainer, string menu[], string products[], int price[]);
int display_sets();
void family_set(vector<cProduct> *vContainer, string menu[], string products[], int price[], int &id);
void double_set(vector<cProduct> *vContainer, string menu[], string products[], int price[], int &id);
void student_set(vector<cProduct> *vContainer, string menu[], string products[], int price[], int &id);

#endif // FUNKCJE_H
