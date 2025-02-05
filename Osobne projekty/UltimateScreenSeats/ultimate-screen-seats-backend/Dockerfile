# Użyj obrazu bazowego z Pythonem
FROM python:3.10-slim

# Instalacja zależności systemowych wymaganych przez psycopg2
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ustaw katalog roboczy w kontenerze
WORKDIR /app

# Skopiuj plik requirements.txt i zainstaluj zależności
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj resztę kodu aplikacji do kontenera
COPY ./src /app/

# Otwórz port używany przez serwer deweloperski Django
EXPOSE 8000

# Polecenie uruchamiające serwer aplikacji
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
