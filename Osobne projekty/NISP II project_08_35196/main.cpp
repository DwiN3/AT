#include "funkcje.h"
//Projekt nr 8 - "System do zamawiania posilkow"
//Grupa: Kamil Deren oraz Bartek Kubik

int main()
{
    // zmienne do wczytania menu
    string menu[35], products[15];
    int price[35];

    //struktura danych przechowujaca zamowienie
    vector<cProduct> vContainer;

    // do menu glownego
    int first_choice;

    // zmienne do wyboru produktow
    int second_choice;
    int discount_choice;

    // zmienne do zatwweirdzenia i podania adresu dostawy
    string city, address, e_mail;
    int phone_number;
    bool confirm = false;

    // wczytywanie danych z pliku teksowego do menu
    load_file(menu, products, price);

    // zmienna do nadania id do zestawu
    int id = 0;

    do
    {
        display_menu(&vContainer);
        display_order(&vContainer);
        first_choice = main_menu_choice();

        switch(first_choice)
        {
        case 1:

            system("cls");
            cout << "Pizza:\n" << endl;
            display_products(menu, products, price, 0, 15);
            second_choice = product_choice() - 1;
            if(second_choice >= 0 && second_choice < 15) add_product(&vContainer, menu, price, second_choice);
            clear_screen();
            break;

        case 2:

            system("cls");
            cout << "Napoje zimne:\n" << endl;
            display_products(menu, products, price, 15, 25);
            second_choice = product_choice() + 14;
            if(second_choice >= 15 && second_choice <= 25) add_product(&vContainer, menu, price, second_choice);
            clear_screen();
            break;

        case 3:

            system("cls");
            cout << "Napoje gorace:\n" << endl;
            display_products(menu, products, price, 25, 35);
            second_choice = product_choice() + 24;
            if(second_choice >= 25 && second_choice <= 35) add_product(&vContainer, menu, price, second_choice);
            clear_screen();
            break;

        case 4:

            system("cls");

            cout << "Zestawy:" << endl << endl;
            second_choice = display_sets();

            do
            {

                switch(second_choice)
                {

                case 1:

                    family_set(&vContainer, menu, products, price, id);
                    system("cls");

                    break;

                case 2:

                    double_set(&vContainer, menu, products, price, id);
                    system("cls");

                    break;

                case 3:

                    student_set(&vContainer, menu, products, price, id);
                    system("cls");

                    break;

                case 4:

                    break;

                }

                break;

            }while(second_choice > 4 || second_choice < 1);
            clear_screen();
            break;

        case 5:

            system("cls");
            cout << "Promocje:\n" << endl;
            show_discount();

            do
            {

                cin >> discount_choice;

                switch(discount_choice)
                {

                case 1:

                    fifty_fifty_pizza(&vContainer, menu, products, price);

                    break;

                case 2:

                    if(check_free_cola_discount(&vContainer) == true && check_discount(&vContainer) == false)
                    {
                        cout << endl << "Spelniasz warunki promocji dla darmowego napoju :)" << endl;
                        add_free_cola_discount(&vContainer);
                        getchar();
                    }
                    else
                    {
                        cout << endl << "Nie spelniasz warunkow promocji" << endl;
                        getchar();
                    }
                    break;

                case 3:

                    if(check_minus_twenty_discount(&vContainer) == true)
                    {
                        cout << endl << "Spelniasz warunki promocji dla -20% :)" << endl;
                        getchar();
                    }
                    else
                    {
                        cout << endl << "Nie spelniasz warunkow promocji" << endl;
                        getchar();
                    }
                    break;

                case 4:

                    break;

                default:

                    cout << "\nNie ma takiej opcji";
                    cin.get(); cin.get();
                    system("cls");
                    show_discount();

                }

            }while((discount_choice < 1 || discount_choice > 4));

            clear_screen();
            break;

        case 6:

            if(vContainer.size() > 0)
            {

                delete_product(&vContainer);
                first_choice = 1;
                system("cls");
                if(check_free_cola_discount(&vContainer) != true) remove_cola(&vContainer);
            }

            break;
            clear_screen();

            break;

        case 7:

            system("cls");
            if(vContainer.size() > 0) cancel_order(&vContainer);
            else cout << "Nie ma takiej opcji!";
            system("cls");

            break;

        case 8:

            system("cls");
            if(vContainer.size() > 0)
            {
                if(confirm_order(&vContainer) == '1')
                {
                delivery_details(&city, &address, &e_mail, &phone_number);
                confirm = true;
                print_bill(&vContainer, city, address, e_mail, phone_number);
                getchar();
                }
                system("cls");
            }

            else
            {
                cout << "Nie ma takiej opcji!";
            }

            break;

        default:

            cout << "Nie ma takiej opcji!";
        }

        if((vContainer.size() == 0 && first_choice == 6) || (confirm == true))
        {
            break;
        }
    }

    while(1);
    {
        system("cls");
        cout << "Zapraszamy ponownie!";
        clear_screen();
    }

    return 0;
}
