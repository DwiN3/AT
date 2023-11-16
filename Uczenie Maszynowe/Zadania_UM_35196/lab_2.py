import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ZADANIE 1

# Generowanie danych losowych z rozkładu normalnego
np.random.seed(42)
data = {
   'A': np.random.normal(5, .5, 100),
   'B': np.random.normal(0, 200, 100)
}

# Dodanie wartości odstających (outliers) do danych
outliers = [40.5, 50.0, np.nan, -4.0]
data['A'] = np.concatenate((data['A'], outliers))
data['B'] = np.concatenate((data['B'], outliers))

# Tworzenie ramki danych
df = pd.DataFrame(data)

# Wyczyszczenie danych użyj metody dropna() na DataFrame
df_clean_data = df.dropna()

# Tworzenie histogramu
plt.hist(df_clean_data['A'], bins=20)
plt.xlabel('Values A')
plt.ylabel('Numbers')
plt.show()

# Oblicz wartość średnią i odchylenie standardowe zbioru A
mean = df_clean_data['A'].mean()
std = df_clean_data['A'].std()
print(f"Average A: {mean}")
print(f"Standard deviation A: {std}")

# Analiza i usuwanie wartości odstających. Dobierz odpowiednie wartości
df_cleaned = df_clean_data[(df_clean_data['A'] >= mean - 2 * std) & (df_clean_data['A'] <= mean + 2 * std)]

# Wyświetlenie oczyszczonych danych (histogram)
plt.hist(df_cleaned['A'], bins=20)
plt.xlabel('Values A (clean data)')
plt.ylabel('Numbers')
plt.show()



# ZADANIE 2

# Generowanie danych losowych z rozkładu normalnego
np.random.seed(42)
data = {
   'A': np.random.normal(5, 0.5, 100),
   'B': np.random.normal(0, 200, 100)
}

# Stworzenie ramki danych
df = pd.DataFrame(data)

# Wizualizacja danych przed standaryzacją
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.scatter(df['A'], df['B'])
plt.title('Data before standardization')
plt.xlabel('Series A')
plt.ylabel('Series B')

# Standaryzacja danych
scaler = StandardScaler()
df[['A', 'B']] = scaler.fit_transform(df[['A', 'B']])

# Wizualizacja danych po standaryzacji
plt.subplot(122)
plt.scatter(df['A'], df['B'])
plt.title('Data after standardization')
plt.xlabel('Series A (standardized)')
plt.ylabel('Series B (standardized)')

plt.tight_layout()
plt.show()