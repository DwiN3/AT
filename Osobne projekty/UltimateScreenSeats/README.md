# ***Technologia aplikacji webowych II***

## Wykonanie: 
- **Michał Polak**
- **Kamil Dereń**

## Temat projektu: **Aplikacja do rezerwacji biletów w kinie**

## Opis projektu:

Aplikacja do rezerwacji biletów w kinie umożliwia użytkownikom łatwe przeglądanie aktualnie prezentowanych w kinie filmów, sprawdzanie dostępnych terminów seansów oraz rezerwowanie miejsc w salach kinowych. Dzięki intuicyjnemu interfejsowi użytkownicy mogą szybko zarejestrować się, zalogować i dokonywać rezerwacji, a także zarządzać swoimi wcześniejszymi rezerwacjami. 
Administratorzy mają natomiast możliwość zarządzania filmami, terminami seansów oraz rezerwacjami. 

<br>

# Uruchomienie projektu

## Uruchomienie backendu

### Opcja 1 - z użyciem obrazu Docker

### Wymagania wstępne

- **Docker**

#### 1. **Otwórz główny folder aplikacji w terminalu**
   
np.

```
E:\Projects\UltimateScreenSeats
```

#### 2. **Zbuduj obraz Docker**

Wykonaj w konsoli polecenie:

```
docker build -t image-name .
```

-   `image-name` – nazwa obrazu.

#### 3. Uruchom kontener

Uruchomienie kontenera ze stworzonym wcześniej obrazem:

```
docker run -p 8000:8000 image-name
```

-   `-p 8000:8000` – mapowanie portów, na których będzie dostępny kontener
-   `image-name` – nazwa obrazu stworzonego w poprzednim kroku

#### 4. Po poprawnym utworzeniu kontenera

- Backend będzie dostępny pod: `http://localhost:8080/api/`.
- Dokumentacja API dostępna będzie pod: `http://localhost:8000/api/docs#/`

### Opcja 2 - klasyczne uruchomienie

### Wymagania wstępne

- **Python** w wersji `3.11.3`
- **Virtualenv** dla izolacji środowiska Python

#### 1. **Przejdź do folderu `ultimate-screen-seats-backend`:**

```
cd ultimate-screen-seats-backend
```

#### 2. **Utwórz wirtualne środowisko Python:**

```
python3 -m venv venv
```

#### 3. **Aktywuj wirtualne środowisko:**

Na systemach Unix/Mac:
    
    source venv/bin/activate
    
Na Windows:

    venv\Scripts\activate

#### 4. **Zainstaluj zależności:**

```
pip install -r requirements.txt
```

#### 5. **Wykonaj migracje bazy danych:**

```
python src/manage.py migrate
```

#### 6. **Uruchom serwer deweloperski Django:**

```
python src/manage.py runserver
```

*Backend będzie dostępny pod adresem: `http://127.0.0.1:8000`.*

## Uruchamianie Frontendu

#### 1. **Przejdź do folderu `ultimate-screen-seats-frontend`:**

```
cd ultimate-screen-seats-frontend
```

#### 2. **Zainstaluj zależności projektu:**

```
npm install
```

#### 3. **Uruchom serwer deweloperski Next.js:**

```
npm run dev
```

*Frontend będzie dostępny pod adresem: `http://localhost:3000`.*

<br>

# Technologie użyte w projekcie

### Backend:  
- 🐍 **Python** – Wszechstronny język programowania, używany do budowy logiki aplikacji.  
- 🌐 **Django** – Framework webowy oparty na Pythonie, zapewniający szybki rozwój aplikacji dzięki wbudowanym rozwiązaniom.  
- ⚡ **Django Ninja** – Nowoczesny framework typu API z obsługą FastAPI-like i automatyczną generacją OpenAPI.  
- 🎯 **Django Ninja Extra** – Rozszerzenie Django Ninja, które wprowadza dodatkowe funkcjonalności, takie jak kontrolery i lepsze zarządzanie routami.  

### Frontend:  
- ⚛️ **React.js** – Biblioteka JavaScript do budowy dynamicznych i interaktywnych interfejsów użytkownika.  
- ⚡ **Next.js** – Framework oparty na React, umożliwiający server-side rendering i optymalizację pod kątem SEO.  
- 📘 **TypeScript** – Rozszerzenie JavaScript z typowaniem, które zwiększa niezawodność i czytelność kodu.  
- 🎨 **Tailwind CSS** – Framework CSS, pozwalający na szybkie budowanie nowoczesnych i responsywnych interfejsów użytkownika.  
- 🖼️ **Hero.UI** – Biblioteka komponentów UI, dostarczająca gotowe elementy do budowy estetycznych aplikacji.  

### Baza danych:  
- 🐘 **PostgreSQL** – Relacyjna baza danych, znana z wysokiej wydajności i zaawansowanych funkcji.  

### Infrastruktura:  
- 🐳 **Docker** – Platforma do tworzenia, zarządzania i uruchamiania aplikacji w kontenerach, zapewniająca łatwą konfigurację i deployment.