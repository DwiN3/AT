
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def visualize_plot_scatter(x_plot, y_plot, x_scatter, y_scatter, title=None, xlabel="x", ylabel="y", plot_color="blue",
                           scatter_color="red"):
    """Tworzenie wykresów i zwracanie obiektu Figure."""

    # Tworzenie nowej figury
    fig = Figure(figsize=(15, 10))
    ax = fig.add_subplot(111)  # Dodanie subplots (Axes)

    # Rysowanie na wykresie liniowego, jeśli dane są dostarczone
    if x_plot is not None and y_plot is not None:
        ax.plot(x_plot, y_plot, color=plot_color, linewidth=1)

    # Dodawanie punktów na wykresie, jeśli dane są dostarczone
    if x_scatter is not None and y_scatter is not None:
        ax.scatter(x_scatter, y_scatter, color=scatter_color, marker="o")

    # Ustawienie etykiet i tytułu
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title)

    return fig  # Zwracanie utworzonej figury

def visualize(df, title, x, y, regression=False, grouping=None):
    """Tworzenie wykresów.
       Parametr df to dane w formie ramki danych."""
    colors = ["blue", "orange", "red", "green", "magenta", "grey", "yellow",
              "black", "purple", "navy", "pink", "cyan", "white"]

    # Tworzenie nowej figury
    fig = Figure(figsize=(15, 10))
    ax = fig.add_subplot(111)  # Dodanie subplots

    if regression and grouping:
        import seaborn as sns
        sns.set(style="whitegrid")
        lm = sns.lmplot(x=x, y=y, data=df, fit_reg=True, legend=True, height=10, aspect=1.55, hue=grouping,
                        palette=colors, legend_out=True)
        lm.fig.suptitle(title)
        lm.fig.tight_layout()
        return lm.fig  # Zwróć całą figurę z lmplot, który już jest Figure
    else:
        # Rysowanie punktowego wykresu danych
        scatter = ax.scatter(df[x], df[y], color=colors[0], marker="o")
        ax.set_xlabel(x, fontsize=10, horizontalalignment="center")
        ax.set_ylabel(y, fontsize=10, horizontalalignment="center")
        ax.set_title(title)

    return fig  # Zwróć utworzoną figurę


def prediction(model, x_test, y_test):
    """Predykcja z wykorzystaniem REGRESJI LINIOWEJ"""

    # Predykcja
    y_pred = model.predict(x_test)
    df_prediction = pd.DataFrame({"x_test": x_test.ravel(), "y_test": y_test.ravel(), "y_pred": y_pred.ravel()})

    # Ocena [modelu] predykcji
    # mse = mean_squared_error(y_test, y_pred)
    # r_square_predict = r2_score(y_test, y_pred)
    # r_square_test = model.score(x_test, y_test)

    values = [["Metoda", model.__class__.__name__.upper()],
              ["Błąd średniokwadratowy (MSE):", mean_squared_error(y_test, y_pred)],
              ["Współczynnik determinacji (r^2) 'P':", r2_score(y_test, y_pred)],
              ["Współczynnik determinacji (r^2) 'T'':", model.score(x_test, y_test)]]
    description = pd.DataFrame(data=values, columns=["", ""])

    return df_prediction, description


def predictor(x_train, y_train):
    """Model predykcji tj. predyktor z wykorzystaniem REGRESJI LINIOWEJ"""

    # Utworzenie modelu predykcji tj. predyktora z wykorzystaniem REGRESJI LINIOWEJ
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)

    # Parametry modelu predykcji
    # r_square = model.score(x_train, y_train)
    # slope = model.coef_
    # intercept = model.intercept_

    values = [["Metoda", model.__class__.__name__.upper()],
              ["Współczynnik determinacji (r^2):", model.score(x_train, y_train)],
              ["Współczynnik a (slope):", model.coef_],
              ["Współczynnik b (intercept):", model.intercept_]]
    description = pd.DataFrame(data=values, columns=["", ""])

    return model, description