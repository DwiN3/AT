# Projekt nr 8 - System do zamawiania posiłków 


## Uruchomienie programu:


```
1. Otworzyć plik "Projekt_NISP.pro"

2. Nacisnąć przycisk "Configure Project"

3. Włączyć opcje "Run in Terminal"

4. Po naciśnięciu przycisku konfigurującego projekt automatycznie utworzony folder "build-Projekt_NISP-Desktop_Qt_6_3_0_MinGW_64_bit-Debug" w bliźniaczej lokalizacji do folderu repozytorium

5. Do tego folderu należy umieścić plik "menu.txt"

6. Należy uruchomić program 

7. Możliwe jest złożenie zamówienia

8. Po zatwierdzeniu zamówienia zostaje utworzony plik zawierający Rachunek w folderze w którym został umieszczony plik "menu.txt"
```


## Autorzy:
Kamil Dereń oraz Bartek Kubik


## Podział pracy w projekcie:


```
Wczytywanie do pliku (funkcja load_file)  							 -	Kamil Dereń i Bartek Kubik
Czyszczenie ekranu (funkcja clear_screen) 							 - 	Kamil Dereń
Wyświetlanie głównego menu (funkcja display_menu)   				 - 	Kamil Dereń
Wybór w głównym menu (funkcja main_menu_choice)    					 - 	Kamil Dereń
Wyświetlanie produktów (funkcja display_products)					 - 	Kamil Dereń i Bartek Kubik
Wybór produktów (funkcja product_choice) 							 - 	Kamil Dereń
Wyświetlanie promocji (funkcja show_discount) 						 - 	Kamil Dereń
Wyświetlanie zamówienia (funkcja display_order) 					 - 	Bartek Kubik i Kamil Dereń
Dodanie produktu (funkcja add_product)								 - 	Bartek Kubik
Usuwanie produktu (funkcja delete_product) 							 - 	Bartek Kubik
Anulowanie zamówienia (funkcja cancel_order) 					 	 - 	Bartek Kubik
Zatwierdzanie zamówienia (funkcja confirm_order)					 - 	Bartek Kubik
Dane do dostawy (funkcja delivery_details)							 - 	Bartek Kubik
Sprawdzenie użycia promocji (funkcja check_discount)				 - 	Kamil Dereń
Wydruk paragonu (funkcja print_bill) 								 - 	Kamil Dereń
Sprawdzanie normalnej ceny (funkcja normal_price)					 - 	Kamil Dereń
Sprawdzanie promocji -20% (funkcja check_minus_twenty_discount)		 - 	Kamil Dereń
Wyliczanie konczowej ceny cen (funkcja end_price)					 - 	Kamil Dereń
Sprawdzanie promocji darmowej coli (funkcja heck_free_cola_discount) - 	Kamil Dereń
Dodanie promocyjnego napoju (funkcja add_free_cola_discount)		 -  Kamil Dereń i Bartek Kubik
Usuniecie dodatkowego napoju (funkcja remove_cola)				     - 	Kamil Dereń
Dodanie pizzy 50/50 (funkcja fifty_fifty_pizza)					 	 -  Bartek Kubik
Dodanie wyswietlania zestawów (funkcja display_sets)		     	 -  Bartek Kubik
Dodanie zestawu rodzinnego (funkcja family_set)		     			 -  Bartek Kubik
Dodanie zestawu podwójnego (funkcja double_set)		     			 -  Bartek Kubik
Dodanie zestawu studenckiego (funkcja student_set)				     -  Bartek Kubik
```