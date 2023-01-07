# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
# http://dprogrammer.org/rnn-lstm-gru
import datetime

import sklearn.utils
from dateutil.relativedelta import *
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.arima import ARIMA
from sklearn.model_selection import train_test_split
from torch.nn import LSTM, Module


def read_pm(station):
    path = 'datos_2021/' + station
    data = pd.read_csv(path, sep=';', decimal=',')
    hour = data['Hour'].astype(int)
    hour = hour - 1
    data['Date'] = data['Date'].astype(str)
    data['Hour'] = hour.astype(str)
    aux = data['Date'] + ' ' + data['Hour']
    data['Time'] = pd.to_datetime(aux, format='%d/%m/%Y %H')

    data['NO'] = data['NO'].astype(float)
    data['NOX'] = data['NOX'].astype(float)
    data['NO2'] = data['NO2'].astype(float)
    data['O3'] = data['O3'].astype(float)
    data['OZONO'] = data['OZONO'].astype(float)
    data['O3 8h'] = data['O3 8h'].astype(float)
    data['PM10'] = data['PM10'].astype(float)
    data['PM2.5'] = data['PM2.5'].astype(float)
    data['SO2'] = data['SO2'].astype(float)

    data['D,vien'] = data['D,vien'].astype(float)
    data['H'] = data['H'].astype(float)
    data['P'] = data['P'].astype(float)
    data['Precipitacion'] = data['Precipitacion'].astype(float)
    data['R'] = data['R'].astype(float)
    data['T'] = data['T'].astype(float)
    data['V,vien'] = data['V,vien'].astype(float)

    return data


def preprocessing(data):
    data = data.drop('NO2 - ICA', axis=1)
    data = data.drop('O3 - ICA', axis=1)
    data = data.drop('PM10 - ICA', axis=1)
    data = data.drop('PM2.5 - ICA', axis=1)
    data = data.drop('SO2 - ICA', axis=1)
    data = data.drop('ICA Estacion', axis=1)
    data = data.drop('Date', axis=1)

    data['Hour'] = data['Hour'].fillna(data['Hour'].mean())
    data['NOX'] = data['NOX'].fillna(data['NOX'].mean())
    data['NO'] = data['NO'].fillna(data['NO'].mean())
    data['NO2'] = data['NO2'].fillna(data['NO2'].mean())
    data['O3'] = data['O3'].fillna(data['O3'].mean())
    data['OZONO'] = data['OZONO'].fillna(data['OZONO'].mean())
    data['O3 8h'] = data['O3 8h'].fillna(data['O3 8h'].mean())
    data['PM10'] = data['PM10'].fillna(data['PM10'].mean())
    # data['PM2.5'] = data['PM2.5'].fillna(data['PM2.5'].mean())
    data['SO2'] = data['SO2'].fillna(data['SO2'].mean())
    data['D,vien'] = data['D,vien'].fillna(data['D,vien'].mean())
    data['H'] = data['H'].fillna(data['H'].mean())
    data['P'] = data['P'].fillna(data['P'].mean())
    data['Precipitacion'] = data['Precipitacion'].fillna(data['Precipitacion'].mean())
    data['R'] = data['R'].fillna(data['R'].mean())
    data['T'] = data['T'].fillna(data['T'].mean())
    data['V,vien'] = data['V,vien'].fillna(data['V,vien'].mean())

    return data


class TimeBasedCV(object):
    def __init__(self, train_period=30, test_period=7, freq='days'):
        self.train_period = train_period
        self.test_period = test_period
        self.freq = freq

    def split(self, data, validation_split_date=None, date_column='record_date', gap=0):
        try:
            data['Time']
        except:
            raise KeyError('Time')

        train_indices_list = []
        test_indices_list = []

        if validation_split_date is None:
            validation_split_date = data['Time'].min().date() + eval(
                'relativedelta(' + self.freq + '=self.train_period)')

        start_train = validation_split_date - eval('relativedelta(' + self.freq + '=self.train_period)')
        end_train = start_train + eval('relativedelta(' + self.freq + '=self.train_period)')
        start_test = end_train + eval('relativedelta(' + self.freq + '=gap)')
        end_test = start_test + eval('relativedelta(' + self.freq + '=self.test_period)')

        while end_test < data['Time'].max().date():
            cur_train_indices = list(data[(data['Time'].dt.date >= start_train) &
                                              (data['Time'].dt.date < end_train)].index)
            cur_test_indices = list(data[(data['Time'].dt.date >= start_test) &
                                             (data['Time'].dt.date < end_test)].index)

            print("Train period:", start_train, "-", end_train, ", Test period", start_test, "-", end_test,
                      "# train records", len(cur_train_indices), ", # test records", len(cur_test_indices))

            train_indices_list.append(cur_train_indices)
            test_indices_list.append(cur_test_indices)

            start_train = start_train + eval('relativedelta(' + self.freq + '=self.train_period)')
            end_train = start_train + eval('relativedelta(' + self.freq + '=self.train_period)')
            start_test = end_train + eval('relativedelta(' + self.freq + '=gap)')
            end_test = start_test + eval('relativedelta(' + self.freq + '=self.test_period)')

        index_output = [(train, test) for train, test in zip(train_indices_list, test_indices_list)]
        self.n_splits = len(index_output)

        return index_output

    def get_n_splits(self):
        return self.n_splits


def model(train, target):
    tscv = TimeBasedCV(train_period=30, test_period=7, freq='days')
    # for train_index, test_index in tscv.split(train, validation_split_date=datetime.date(2021, 3, 2)):
        # print(train_index, test_index)

    # print(tscv.get_n_splits())
    x = train
    y = target
    mse_list = []
    mae_list = []
    pred = []
    test = []

    for train_index, test_index in tscv.split(x, validation_split_date=datetime.date(2021, 3, 2)):
        data_train = x.loc[train_index].drop('Time', axis=1)
        target_train = y.loc[train_index]

        data_test = x.loc[test_index].drop('Time', axis=1)
        target_test = y.loc[test_index]

        # model = RandomForestRegressor()
        # model = LinearRegression()
        model = MLPRegressor(verbose=2, max_iter=1000)
        model.fit(data_train, target_train)

        preds = model.predict(data_test)
        pred.append(preds)
        test.append(target_test)
        # mse = mean_squared_error(target_test, preds)
        # mae = mean_absolute_error(target_test, preds)
        # mse_list.append(mse)
        # mae_list.append(mae)
    pred = flatten(pred)
    test = flatten(test)
    np.random.shuffle(pred)
    avg_mse = mean_squared_error(test, pred)
    avg_mae = mean_absolute_error(test, pred)
    print('average mse: ', avg_mse, 'average mae: ', avg_mae)

    plt.title('MLP', fontsize=25)

    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    # aux = -abs(preds-target_test)

    # plt.plot(range(aux), aux, color='blue', label='difference')
    plt.plot(range(len(pred)), pred, color='green', label='prediction')
    plt.plot(range(len(test)), test, color='red', label='actual values')

    plt.legend(prop={'size': 16})
    plt.show()


def flatten(l):
    return [item for sublist in l for item in sublist]


def arima(data):
    forecaster = ARIMA(suppress_warnings=True)
    data = data.to_numpy()
    y_pred = []
    for i in range(3, len(data), 100):
        print(i)
        forecaster.fit(data[:i])
        pred = forecaster.predict(fh=np.arange(1, 100))
        y_pred.append(pred)
    y_pred = flatten(y_pred)
    plt.plot(data, color='green')
    plt.plot(y_pred, color='red')
    # plt.plot(-abs(data - y_pred), color='blue')
    plt.show()


def main():
    data = read_pm('AVDA_TOLOSA.csv')
    dataset = preprocessing(data)

    dataset = dataset.dropna(axis='index')
    """
    perm = np.arange(0, dataset.shape[0])
    np.random.shuffle(perm)
    print(perm)
    dataset = dataset.reindex(perm)
    """
    # dataset = dataset.sample(frac=1)

    # time = dataset['Time']
    # dataset = dataset.drop('Time', axis=1)

    y = dataset['PM2.5']
    x = dataset[['Time', 'Hour', 'NO', 'NOX', 'NO2', 'O3', 'OZONO', 'O3 8h',
                 'SO2', 'D,vien', 'H', 'P', 'Precipitacion', 'R', 'T', 'V,vien']]

    # model(x, y)
    # arima(dataset['PM2.5'])

    limit = int(len(dataset)*0.75)
    x_train = x[:limit]
    x_test = x[limit:]
    y_train = y[:limit]
    y_test = y[limit:]

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    """
    y = dataset['PM2.5'].to_numpy()
    x = dataset[['Hour', 'NO', 'NOX', 'NO2', 'O3', 'OZONO', 'O3 8h', 'PM10'
       'SO2', 'D,vien', 'H', 'P', 'Precipitacion', 'R', 'T', 'V,vien']].to_numpy()
    # x = dataset[['Hour', 'NO', 'NOX', 'NO2', 'O3', 'OZONO', 'O3 8h', 'PM10'
    #        'SO2', 'D,vien', 'H', 'P', 'Precipitacion', 'R', 'T', 'V,vien']].to_numpy()

    # train = dataset[['NOX', 'NO', 'NO2', 'O3', 'SO2']]
    # datasete = dataset.to_numpy()
    

    mlp = MLPRegressor(verbose=2)
    mlp.fit(x_train, y_train)
    p = mlp.predict(x_test)

    mse = np.mean(mean_squared_error(y_test, p))
    mae = np.mean(mean_absolute_error(y_test, p))
    print('mse: ', mse, 'mae: ', mae)

    plt.title('', fontsize=25)

    # plt.plot(-abs(y_test-p), color='khaki', lw=2, label='difference')
    plt.plot(y_test, color='indianred', lw=2,  label='actual value')
    plt.plot(p, color='cornflowerblue', lw=2,  label='prediction')
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    plt.legend(prop={'size': 16})
    plt.show()
    """


main()
