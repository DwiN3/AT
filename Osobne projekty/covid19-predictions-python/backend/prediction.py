import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import datetime
import matplotlib.dates as mdates


def prepare_data(cases, data_type='total'):
    '''Prepare data for training and prediction'''
    dates = sorted(cases.keys())
    X = np.array([datetime.datetime.strptime(date, '%Y-%m-%d').toordinal() for date in dates]).reshape(-1, 1)
    y = np.array([cases[date][data_type] for date in dates])
    return X, y


def make_prediction(filtered_data, prediction_to):
    print(prediction_to)
    '''Create predictions using linear regression and Random Forest for total and new cases'''
    train_data = filtered_data[0]['cases']

    X_train_total, y_train_total = prepare_data(train_data, 'total')
    X_train_new, y_train_new = prepare_data(train_data, 'new')

    # Train the model for total cases using Linear Regression
    model_total_lr = LinearRegression()
    model_total_lr.fit(X_train_total, y_train_total)

    # Train the model for total cases using Random Forest
    model_total_rf = RandomForestRegressor(n_estimators=100)
    model_total_rf.fit(X_train_total, y_train_total)

    # Train the model for new cases using Linear Regression
    model_new_lr = LinearRegression()
    model_new_lr.fit(X_train_new, y_train_new)

    # Train the model for new cases using Random Forest
    model_new_rf = RandomForestRegressor(n_estimators=100)
    model_new_rf.fit(X_train_new, y_train_new)

    # Define prediction range
    prediction_end_date = datetime.datetime.strptime(prediction_to, '%d-%m-%Y').toordinal()

    X_pred_total = np.arange(max(X_train_total), prediction_end_date + 1).reshape(-1, 1)
    y_pred_total_lr = model_total_lr.predict(X_pred_total)
    y_pred_total_rf = model_total_rf.predict(X_pred_total)

    X_pred_new = np.arange(max(X_train_new), prediction_end_date + 1).reshape(-1, 1)
    y_pred_new_lr = model_new_lr.predict(X_pred_new)
    y_pred_new_rf = model_new_rf.predict(X_pred_new)

    # Prepare plots for total cases
    plot_model_prediction(X_train_total, y_train_total, model_total_lr, X_pred_total, y_pred_total_lr,
                          'Prediction Model (Total Cases) - Linear Regression')
    plot_prediction(X_train_total, y_train_total, X_pred_total, y_pred_total_lr,
                    'Prediction (Total Cases) - Linear Regression')
    plot_training_data(X_train_total, y_train_total, 'Training set (Total Cases)')
    plot_training_with_regression(X_train_total, y_train_total, model_total_lr,
                                  'Training set & regression (Total Cases) - Linear Regression')

    plot_model_prediction_new(X_train_new, y_train_new, model_new_lr, X_pred_new, y_pred_new_lr,
                              'Prediction Model (New Cases) - Linear Regression')
    plot_prediction_new(X_train_new, y_train_new, X_pred_new, y_pred_new_lr,
                        'Prediction (New Cases) - Linear Regression')
    plot_training_data_new(X_train_new, y_train_new, 'Training set (New Cases)')
    plot_training_with_regression_new(X_train_new, y_train_new, model_new_lr,
                                      'Training set & regression (New Cases) - Linear Regression')

    # Prepare plots for Random Forest total cases
    plot_model_prediction_rf(X_train_total, y_train_total, model_total_rf, X_pred_total, y_pred_total_rf,
                             'Prediction Model (Total Cases) - Random Forest')
    plot_prediction_rf(X_train_total, y_train_total, X_pred_total, y_pred_total_rf,
                       'Prediction (Total Cases) - Random Forest')
    plot_training_with_regression_rf(X_train_total, y_train_total, model_total_rf,
                                     'Training set & regression (Total Cases) - Random Forest')

    # Prepare plots for Random Forest new cases
    plot_model_prediction_new_rf(X_train_new, y_train_new, model_new_rf, X_pred_new, y_pred_new_rf,
                                 'Prediction Model (New Cases) - Random Forest')
    plot_prediction_new_rf(X_train_new, y_train_new, X_pred_new, y_pred_new_rf,
                           'Prediction (New Cases) - Random Forest')
    plot_training_with_regression_new_rf(X_train_new, y_train_new, model_new_rf,
                                         'Training set & regression (New Cases) - Random Forest')

    return [model_prediction_chart, prediction_chart, training_data_chart, training_with_regression_chart,
            model_prediction_chart_new, prediction_chart_new, training_data_chart_new,
            training_with_regression_chart_new, model_prediction_chart_rf, prediction_chart_rf,
            model_prediction_chart_new_rf, prediction_chart_new_rf]


def plot_model_prediction(X_train, y_train, model, X_pred, y_pred, title):
    '''Plot model prediction for total cases'''
    global model_prediction_chart
    plt.figure(figsize=(8, 6))
    plt.scatter([datetime.date.fromordinal(int(x)) for x in X_train], y_train, color='blue', label='Training data')
    plt.plot([datetime.date.fromordinal(int(x)) for x in X_pred], y_pred, color='red', label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('Total cases')
    plt.title(title)
    plt.legend()
    model_prediction_chart = plt.gcf()
    plt.close()


def plot_prediction(X_train, y_train, X_pred, y_pred, title):
    '''Plot prediction for total cases'''
    global prediction_chart
    plt.figure(figsize=(8, 6))
    plt.plot([datetime.date.fromordinal(int(x)) for x in X_pred], y_pred, color='red', label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('Total cases')
    plt.title(title)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.gcf().autofmt_xdate()
    prediction_chart = plt.gcf()
    plt.close()


def plot_training_data(X_train, y_train, title):
    '''Plot training data for total cases'''
    global training_data_chart
    plt.figure(figsize=(8, 6))
    plt.plot([datetime.date.fromordinal(int(x)) for x in X_train], y_train, color='blue', label='Training data')
    plt.xlabel('Date')
    plt.ylabel('Total cases')
    plt.title(title)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.gcf().autofmt_xdate()
    training_data_chart = plt.gcf()
    plt.close()


def plot_training_with_regression(X_train, y_train, model, title):
    '''Plot training data with regression for total cases'''
    global training_with_regression_chart
    plt.figure(figsize=(8, 6))
    plt.scatter([datetime.date.fromordinal(int(x)) for x in X_train], y_train, color='blue', label='Training data')
    plt.plot([datetime.date.fromordinal(int(x)) for x in X_train], model.predict(X_train), color='red', label='Regression line')
    plt.xlabel('Date')
    plt.ylabel('Total cases')
    plt.title(title)
    plt.legend()
    training_with_regression_chart = plt.gcf()
    plt.close()


def plot_model_prediction_new(X_train, y_train, model, X_pred, y_pred, title):
    '''Plot model prediction for new cases'''
    global model_prediction_chart_new
    plt.figure(figsize=(8, 6))
    plt.scatter([datetime.date.fromordinal(int(x)) for x in X_train], y_train, color='blue', label='Training data')
    plt.plot([datetime.date.fromordinal(int(x)) for x in X_pred], y_pred, color='red', label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('New cases')
    plt.title(title)
    plt.legend()
    model_prediction_chart_new = plt.gcf()
    plt.close()


def plot_prediction_new(X_train, y_train, X_pred, y_pred, title):
    '''Plot prediction for new cases'''
    global prediction_chart_new
    plt.figure(figsize=(8, 6))
    plt.plot([datetime.date.fromordinal(int(x)) for x in X_pred], y_pred, color='red', label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('New cases')
    plt.title(title)
    plt.legend()
    prediction_chart_new = plt.gcf()
    plt.close()


def plot_training_data_new(X_train, y_train, title):
    '''Plot training data for new cases'''
    global training_data_chart_new
    plt.figure(figsize=(8, 6))
    plt.scatter([datetime.date.fromordinal(int(x)) for x in X_train], y_train, color='blue', label='Training data')
    plt.xlabel('Date')
    plt.ylabel('New cases')
    plt.title(title)
    plt.legend()
    training_data_chart_new = plt.gcf()
    plt.close()


def plot_training_with_regression_new(X_train, y_train, model, title):
    '''Plot training data with regression for new cases'''
    global training_with_regression_chart_new
    plt.figure(figsize=(8, 6))
    plt.scatter([datetime.date.fromordinal(int(x)) for x in X_train], y_train, color='blue', label='Training data')
    plt.plot([datetime.date.fromordinal(int(x)) for x in X_train], model.predict(X_train), color='red', label='Regression line')
    plt.xlabel('Date')
    plt.ylabel('New cases')
    plt.title(title)
    plt.legend()
    training_with_regression_chart_new = plt.gcf()
    plt.close()


def plot_model_prediction_rf(X_train, y_train, model, X_pred, y_pred, title):
    '''Plot model prediction for total cases using Random Forest'''
    global model_prediction_chart_rf
    plt.figure(figsize=(8, 6))
    plt.scatter([datetime.date.fromordinal(int(x)) for x in X_train], y_train, color='blue', label='Training data')
    plt.plot([datetime.date.fromordinal(int(x)) for x in X_pred], y_pred, color='green', label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('Total cases')
    plt.title(title)
    plt.legend()
    model_prediction_chart_rf = plt.gcf()
    plt.close()


def plot_prediction_rf(X_train, y_train, X_pred, y_pred, title):
    '''Plot prediction for total cases using Random Forest'''
    global prediction_chart_rf
    plt.figure(figsize=(8, 6))
    plt.plot([datetime.date.fromordinal(int(x)) for x in X_pred], y_pred, color='green', label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('Total cases')
    plt.title(title)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.gcf().autofmt_xdate()
    prediction_chart_rf = plt.gcf()
    plt.close()


def plot_training_with_regression_rf(X_train, y_train, model, title):
    '''Plot training data with regression for total cases using Random Forest'''
    global training_with_regression_chart_rf
    plt.figure(figsize=(8, 6))
    plt.scatter([datetime.date.fromordinal(int(x)) for x in X_train], y_train, color='blue', label='Training data')
    plt.plot([datetime.date.fromordinal(int(x)) for x in X_train], model.predict(X_train), color='green', label='Regression line')
    plt.xlabel('Date')
    plt.ylabel('Total cases')
    plt.title(title)
    plt.legend()
    training_with_regression_chart_rf = plt.gcf()
    plt.close()


def plot_model_prediction_new_rf(X_train, y_train, model, X_pred, y_pred, title):
    '''Plot model prediction for new cases using Random Forest'''
    global model_prediction_chart_new_rf
    plt.figure(figsize=(8, 6))
    plt.scatter([datetime.date.fromordinal(int(x)) for x in X_train], y_train, color='blue', label='Training data')
    plt.plot([datetime.date.fromordinal(int(x)) for x in X_pred], y_pred, color='green', label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('New cases')
    plt.title(title)
    plt.legend()
    model_prediction_chart_new_rf = plt.gcf()
    plt.close()


def plot_prediction_new_rf(X_train, y_train, X_pred, y_pred, title):
    '''Plot prediction for new cases using Random Forest'''
    global prediction_chart_new_rf
    plt.figure(figsize=(8, 6))
    plt.plot([datetime.date.fromordinal(int(x)) for x in X_pred], y_pred, color='green', label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('New cases')
    plt.title(title)
    plt.legend()
    prediction_chart_new_rf = plt.gcf()
    plt.close()


def plot_training_with_regression_new_rf(X_train, y_train, model, title):
    '''Plot training data with regression for new cases using Random Forest'''
    global training_with_regression_chart_new_rf
    plt.figure(figsize=(8, 6))
    plt.scatter([datetime.date.fromordinal(int(x)) for x in X_train], y_train, color='blue', label='Training data')
    plt.plot([datetime.date.fromordinal(int(x)) for x in X_train], model.predict(X_train), color='green', label='Regression line')
    plt.xlabel('Date')
    plt.ylabel('New cases')
    plt.title(title)
    plt.legend()
    training_with_regression_chart_new_rf = plt.gcf()
    plt.close()