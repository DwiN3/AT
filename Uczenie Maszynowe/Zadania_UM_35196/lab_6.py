import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Laboratorium 6

# 1. Wczytanie danych
data_frame = pd.read_csv('home_prices.csv')

# 2. Wyświetl wszystkich kolumn
print("Wszystkie kolumny:", data_frame.columns)

# 3. Usunięcie brakujących danych
data_frame.dropna()

# 4. Wybranie kolumny "SalePrice" jako target (macierz Y)
target_column = "SalePrice"
Y = data_frame[target_column]

# 5 i 6. Wybranie cechy jako macierz X
feature_columns = ['OverallQual', 'OverallCond', 'YearBuilt', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']
X = data_frame[feature_columns]

# 7. Wyświetlenie statystyki macierzy cech
print("Statystyki macierzy cech:")
print(X.describe())

# 8. Utwórzenie zbióru treningowego i testowego
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 9. Przetasowanie przez algorytmy
algorithms = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]
for algorithm in algorithms:
    algorithm_name = algorithm.__class__.__name__
    model = algorithm.fit(X_train, Y_train)
    predictions = model.predict(X_test)

# 10. Wyświetlenie wyników testu
mse = mean_squared_error(Y_test, predictions)
mae = mean_absolute_error(Y_test, predictions)
print(f"\nWyniki dla algorytmu {algorithm_name}:")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
