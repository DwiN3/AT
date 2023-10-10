# Fiszki

"Fiszki" to aplikacja mobilna która umożliwia skuteczną naukę języków obcych w sposób interaktywny i przyjemny. Aplikacja polega na wyświetlaniu słów w wybranym języku, a zadaniem użytkownika jest wpisanie ich tłumaczenia lub naciśnięcie w skazane miejsce w zależności od wybranego trybu. 

Użytkownik w aplikacji "Fiszki" może wybrać tematykę słówek, z których chce się uczyć w danym momencie. Dzięki temu, nauka jest bardziej skuteczna i dostosowana do indywidualnych potrzeb i preferencji użytkownika. Aplikacja oferuje szeroki wybór tematów, takich jak podróżowanie, jedzenie, sport czy praca. 

"Fiszki" to narzędzie, które można wykorzystać w każdym miejscu i czasie. Aplikacja umożliwia codzienne powtarzanie i utrwalanie słownictwa, co pozwala na szybki postęp w nauce języka. Dodatkowo, użytkownik może monitorować swoje postępy i osiągnięcia, co motywuje do dalszej nauki. 

Aplikacja "Fiszki" jest odpowiednia dla osób w każdym wieku i o różnym poziomie zaawansowania. To narzędzie, które pozwala na skuteczną naukę języków obcych w sposób przyjemny i interaktywny.


## Ekrany aplikacji:
<div align="center">
  <table>
    <tr>
      <td style="text-align: center;">
        <img src="https://github.com/DwiN3/Fiszki/assets/104890694/a29624e0-4618-4b11-ba79-9d660e2c5544" alt="Menu główne" width="285" height="580"/>
      </td>
      <td style="text-align: center;">
        <img src="https://github.com/DwiN3/Fiszki/assets/104890694/0edf25f0-6671-47d0-8bdc-bf014fcdfa76" alt="Dodawanie fiszek" width="285" height="580"/>
      </td>
      <td style="text-align: center;">
        <img src="https://github.com/DwiN3/Fiszki/assets/104890694/3dfb5e55-9933-4f31-bf1d-81bbe244ca50" alt="Panel zestawów" width="285" height="580"/>
    </tr>
  </table>
</div>
<br>
<div align="center">
  <table>
    <tr>
      <td style="text-align: center;">
        <img src="https://github.com/DwiN3/Fiszki/assets/104890694/8fdac8f8-2e1b-4342-ace7-6d8756978847" alt="Wybór trybu gry" width="285" height="580"/>
      </td>
      <td style="text-align: center;">
        <img src="https://github.com/DwiN3/Fiszki/assets/104890694/8d8e9931-d088-4a12-bcb2-b9b5d2f6c986" alt="Tryb Gry Quiz" width="285" height="580"/>
      </td>
      <td style="text-align: center;">
        <img src="https://github.com/DwiN3/Fiszki/assets/104890694/da7e1030-4ed1-475d-8cf9-b41481de8a90" alt="Tryb Gry Nauka" width="285" height="580"/>
    </tr>
  </table>
</div>
<br>

## Opis stanu aplikacji:

Aplikacja działa na samodzielnie wykonanym api które wykorzystuje serwer: https://render.com/ a łączy się z nim poprzez rozszerzenie Retrofit.

W aplikacji można utworzyć konto, zalogować się i zresetować hasło istniejącego konta. Każde zalogowanie na konto generuje użytkownikowi nowy token który jest przechowywany w aplikacji. Gdy użytkownik się nie wylogował przed wyłączeniem aplikacji zostaje od bezpośrednie przeniesiony do menu głównego bez konieczności podawania danych logowania.

Po zalogowaniu można korzystać z zawartości aplikacji. Użytkownik ma możliwość wybrać jedną z pośród wielu kategorii która zawiera słowa o poszczególnej tematyce,  możliwe zagranie w tryb “quiz” lub “nauka”.

Aplikacja posiada dwa tryby gry:
* Tryb “quiz” pozwala użytkownikowi na zyskiwanie punktów aby zwiększać poziom użytkownika. Gracz wybiera jedną spośród czterech odpowiedzi, gra się kończy po 15 fiszkach i zostajemy przeniesieni do ekranu który zliczy nasze postępy gry, obecny lvl oraz wynik najlepszej poprawnej passy.
* Tryb “nauka” służy do powtarzaniu słówek z całego zestawu, użytkownik przechodzi przez wszystkie słówka zestawu lub kategorii, pozwala to na szybkie utrwalanie słownictwa.

Użytkownik ma również możliwość przeglądać swoje zestawy w zakładce “Twoje Fiszki”. 
Wyświetlają się tam wszystkie zestawy użytkownika, można je modyfikować lub usunąć. Po wybraniu zestawu można przeglądać słowa które się w nich znajdują, a po wyborze konkretnego słowa dowolnie je modyfikować.


## Prezentacja aplikacji:
https://www.youtube.com/watch?v=Jg4s9pfKYkw

## Autorzy:
Kamil Dereń oraz Bartek Kubik
