import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import floor, ceil


def read_data(station):
    path = '../../data_sources/euskadi/datos_2021/' + station
    data = pd.read_csv(path, sep=';', decimal=',')
    # hour = pd.to_datetime(data['Hour'], format='%H:%M')
    hour = data['Hour'].astype(int)
    hour = hour - 1
    data['Date'] = data['Date'].astype(str)
    data['Hour'] = hour.astype(str)
    aux = data['Date'] + ' ' + data['Hour']
    data['Time'] = pd.to_datetime(aux, format='%d/%m/%Y %H')
    data['PM2.5'] = data['PM2.5'].astype(float)

    return data


def line_chart_2021(data):
    x_ticks = ['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01',
               '2021-08-01', '2021-09-01', '2021-10-01', '2021-11-01', '2021-12-01']
    x_labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                'November', 'December']

    plt.plot(data['Time'], data['PM2.5'])
    plt.title('PM2.5 pollution in Tolosa Hiribidea during 2021', fontsize=25)
    plt.ylabel('PM2.5 in \u03BCm', fontsize=16)
    plt.xlabel('Date', fontsize=16)
    plt.xticks(fontsize=15, ticks=x_ticks, labels=x_labels)
    plt.yticks(fontsize=16)
    plt.show()


def measurements(data):
    x_ticks = ['2021-12-01', '2021-12-05', '2021-12-10', '2021-12-15', '2021-12-20', '2021-12-25', '2021-12-30']
    x_labels = ['12-01', '12-05', '12-10', '12-15', '12-20', '12-25', '12-30']

    plt.subplot(2, 1, 1)
    plt.ylabel('PM2.5', fontsize=16)
    plt.plot(data['Time'], data['PM2.5'], color='blue', label='PM2.5')
    plt.xticks(fontsize=16, ticks=x_ticks, labels=x_labels)
    plt.yticks(fontsize=16)

    plt.subplot(2, 1, 2)
    plt.ylabel('Precipitation', fontsize=16)
    plt.plot(data['Time'], data['Precipitacion'], color='blue', label='Euria')
    plt.xticks(fontsize=16, ticks=x_ticks, labels=x_labels)
    plt.yticks(fontsize=16)

    plt.suptitle('Measurements December 2021', fontsize=25)
    plt.show()


def pollution_per_station(data1, data2, data3):
    x_ticks = ['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01',
               '2021-08-01', '2021-09-01', '2021-10-01', '2021-11-01', '2021-12-01']
    x_labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']

    plt.plot(data1['Time'], data1['PM2.5'], color='cornflowerblue', lw=2, label='Tolosa Hiribidea, Donostia')
    plt.plot(data2['Time'], data2['PM2.5'], color='indianred', lw=2, label='Mº Diaz Haro, Bilbo')
    plt.plot(data3['Time'], data3['PM2.5'], color='khaki', lw=2, label='Martxoaren 3a, Gasteiz')
    plt.title('PM2.5 pollution in Donostia, Bilbo and Gasteiz stations during 2021', fontsize=25)
    plt.xticks(fontsize=15, ticks=x_ticks, labels=x_labels)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 16})
    plt.show()


def ranges(data):
    maxi = ceil(data['PM2.5'].max())
    mini = floor(data['PM2.5'].min())
    count = [0] * maxi
    index = [0] * maxi
    total = 0
    for i, j in enumerate(range(mini, maxi+1)):
        subset = data[(data['PM2.5'] > i) & (data['PM2.5'] < j)]
        [s, _] = subset.shape
        count[i] = s
        total += s
        index[i] = str(i) + '-' + str(j)
    bars = range(maxi)
    x_pos = np.arange(len(bars))
    plt.barh(x_pos, count)
    plt.title('Total of PM2.5 values per interval', fontsize=25)
    plt.yticks(x_pos, index, fontsize=16)
    plt.xticks(fontsize=16)
    print(total)
    plt.show()


def several_lines_location():
    x_ticks = ['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01',
               '2021-08-01', '2021-09-01', '2021-10-01', '2021-11-01', '2021-12-01']
    x_labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                'November', 'December']

    barakaldo = read_data('BARAKALDO.csv')
    durango = read_data('DURANGO.csv')
    easo = read_data('EASO.csv')

    # urban
    plt.plot(easo['Time'], easo['PM2.5'], color='cornflowerblue', lw=2, label='Easo, Traffic')  # traffic
    plt.plot(durango['Time'], durango['PM2.5'], color='indianred', lw=2, label='Durango, Industrial')  # industrial
    plt.plot(barakaldo['Time'], barakaldo['PM2.5'], color='khaki', lw=2, label='Barakaldo, Background')  # background

    plt.xticks(fontsize=15, ticks=x_ticks, labels=x_labels)
    plt.yticks(fontsize=16)
    plt.title('PM2.5 in 3 stations per pollution type', fontsize=25)
    plt.legend(prop={'size': 16})
    plt.show()


def main():
    avda_tolosa = read_data('AVDA_TOLOSA.csv')
    # diaz_haro = read_data('M_DIAZ_HARO.csv')
    # marzo = read_data('3_DE_MARZO.csv')
    ranges(avda_tolosa)

main()
