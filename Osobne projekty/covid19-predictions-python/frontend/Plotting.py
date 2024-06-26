import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d


def configure_plot(fig, ax, title, xlabel, ylabel):
    """Konfiguruje wygląd wykresu zgodnie z ustalonym stylem."""
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid()


def create_smooth_line_chart(dates, values, title, xlabel, ylabel):
    """Tworzy wykres liniowy z interpolacją sześcienną."""
    fig, ax = plt.subplots(figsize=(8, 6))

    num_dates = [mdates.date2num(datetime.datetime.strptime(date, '%Y-%m-%d')) for date in dates]

    min_day = min(num_dates)
    normalized_days = [day - min_day for day in num_dates]

    interp_days = np.linspace(min(normalized_days), max(normalized_days), 1000)
    cubic_interp = interp1d(normalized_days, values, kind='cubic')
    smooth_values = cubic_interp(interp_days)

    ax.plot(interp_days, smooth_values, label='Smooth Line Chart', color='red')

    configure_plot(fig, ax, title, xlabel, ylabel)

    return fig


def plot_country_chart(data,country,type):
    """Tworzy i rysuje wykres danych COVID dla danego kraju."""
    cases = data[0]['cases']
    dates = list(cases.keys())
    values = [details[type] for date, details in cases.items()]

    dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dates, values, label=type)
    ax.set_title(f'{type.capitalize()} Cases for {country}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Cases')
    ax.legend()

    num_days = (dates[-1] - dates[0]).days

    if num_days > 365: # if less than year then set interval, divide the year into 4 months
        locator = mdates.MonthLocator(interval=3)
    elif num_days > 90:
        locator = mdates.MonthLocator(interval=1)
    elif num_days > 30:
        locator = mdates.WeekLocator()
    else:
        locator = mdates.DayLocator(interval=1)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    pass
