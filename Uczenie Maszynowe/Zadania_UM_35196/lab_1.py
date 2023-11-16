import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Zadanie 1

# Wczytanie danych
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
# Wyświetlenie pierwszych wierszy
print(df.head())


# Zadanie 2

missing_data = df.isnull().sum()
print("Brakujące dane:")
print(missing_data)
# Usunięcie wierszy z brakującymi danymi
df_cleaned = df.dropna()

# Zadanie 3

# Wykres punktowy
plt.figure(figsize=(8, 6))
for species in df_cleaned['target'].unique():
 subset = df_cleaned[df_cleaned['target'] == species]
 plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'],
label=iris.target_names[species])
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.legend()
plt.show()


# Zadanie 4

class_counts = df['target'].value_counts()
print("\nLiczba próbek dla każdej z klas:")
print(class_counts)


# Zadanie 5

# Wybór cechy do analizy, na przykład "sepal length (cm)"
chosen_feature = "sepal length (cm)"
# Tworzenie histogramu
plt.figure(figsize=(8, 6))
plt.hist(df[chosen_feature], bins=20, edgecolor='k')
plt.xlabel(chosen_feature)
plt.ylabel('Liczba Próbek')
plt.title(f'Histogram {chosen_feature}')
plt.show()

# Histogramu dla klasy target
plt.figure(figsize=(8, 6))
plt.hist(df['target'], bins=3, edgecolor='k', rwidth=0.8)
plt.xlabel('Klasa (Target)')
plt.ylabel('Liczba Próbek')
plt.xticks(range(3), iris.target_names)
plt.title('Histogram Klas (Target)')
plt.show()