import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np
import seaborn as sns
from math import floor, ceil
from matplotlib.cm import ScalarMappable


def read_pm(station):
    path = 'datos_2021/' + station
    data = pd.read_csv(path, sep=';', decimal=',')
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
    # these lines are meant for december, change the number, ticks and labels in order to change the month
    data = data[(data['Time'].dt.month == 12)]
    x_ticks = ['2021-12-01', '2021-12-05', '2021-12-10', '2021-12-15', '2021-12-20', '2021-12-25', '2021-12-30']
    x_labels = ['12-01', '12-05', '12-10', '12-15', '12-20', '12-25', '12-30']

    data['Precipitacion'] = data['Precipitacion'].astype(float)

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


def pm_comparison(data):
    data = data[(data['Time'].dt.month == 9)]
    data['PM10'] = data['PM10'].astype(float)

    plt.plot(data['Time'], data['PM2.5'], color='cornflowerblue', lw=2, label='PM2.5')
    plt.plot(data['Time'], data['PM10'], color='indianred', lw=2, label='PM10')
    plt.title('PM measurements during September 2021', fontsize=25)

    x_ticks = ['2021-09-01', '2021-09-05', '2021-09-10', '2021-09-15', '2021-09-20', '2021-09-25', '2021-09-30']
    x_labels = ['09-01', '09-05', '09-10', '09-15', '09-20', '09-25', '09-30']
    plt.xticks(ticks=x_ticks, labels=x_labels, fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(prop={'size': 16})
    plt.show()


def pollution_per_station():
    data1 = read_pm('AVDA_TOLOSA.csv')
    data2 = read_pm('M_DIAZ_HARO.csv')
    data3 = read_pm('3_DE_MARZO.csv')

    x_ticks = ['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01',
               '2021-08-01', '2021-09-01', '2021-10-01', '2021-11-01', '2021-12-01']
    x_labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                'November', 'December']

    plt.plot(data1['Time'], data1['PM2.5'], color='cornflowerblue', lw=2, label='Tolosa Hiribidea, Donostia')
    plt.plot(data2['Time'], data2['PM2.5'], color='indianred', lw=2, label='Mº Diaz Haro, Bilbo')
    plt.plot(data3['Time'], data3['PM2.5'], color='khaki', lw=2, label='Martxoaren 3a, Gasteiz')

    plt.title('PM2.5 pollution in Donostia, Bilbo and Gasteiz stations during 2021', fontsize=25)
    plt.legend(prop={'size': 16})

    plt.xticks(fontsize=15, ticks=x_ticks, labels=x_labels)
    plt.yticks(fontsize=16)

    plt.show()


def intervals(data):
    maxi = ceil(data['PM2.5'].max())
    mini = floor(data['PM2.5'].min())
    count = [0] * maxi
    index = [0] * maxi

    for i, j in enumerate(range(mini, maxi+1)):
        subset = data[(data['PM2.5'] > i) & (data['PM2.5'] < j)]
        [s, _] = subset.shape
        count[i] = s
        index[i] = str(i) + '-' + str(j)

    bars = range(maxi)
    x_pos = np.arange(len(bars))
    plt.barh(x_pos, count)

    plt.title('Total of PM2.5 values per interval', fontsize=25)
    plt.yticks(x_pos, index, fontsize=16)
    plt.xticks(fontsize=16)

    plt.show()


def pollution_per_origin():
    barakaldo = read_pm('BARAKALDO.csv')
    durango = read_pm('DURANGO.csv')
    easo = read_pm('EASO.csv')

    x_ticks = ['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01',
               '2021-08-01', '2021-09-01', '2021-10-01', '2021-11-01', '2021-12-01']
    x_labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                'November', 'December']

    # urban
    plt.plot(easo['Time'], easo['PM2.5'], color='cornflowerblue', lw=2, label='Easo, Traffic')  # traffic
    plt.plot(durango['Time'], durango['PM2.5'], color='indianred', lw=2, label='Durango, Industrial')  # industrial
    plt.plot(barakaldo['Time'], barakaldo['PM2.5'], color='khaki', lw=2, label='Barakaldo, Background')  # background

    plt.title('PM2.5 in 3 stations per pollution type', fontsize=25)
    plt.legend(prop={'size': 16})

    plt.xticks(fontsize=15, ticks=x_ticks, labels=x_labels)
    plt.yticks(fontsize=16)
    plt.show()

# HEATMAPS


def single_plot(data, title, ax, maxi, mini):
    day = data['Time'].dt.day
    aux = data['PM2.5']

    aux = aux.values.reshape(24, len(day.unique()), order='F')
    x_grid = np.arange(len(day.unique())+1) + 1
    y_grid = np.arange(25)

    ax.pcolormesh(x_grid, y_grid, aux, cmap='magma', vmin=mini, vmax=maxi)
    ax.set_ylim(24, 0)
    ax.set_title(title, fontsize=15)

    ax.yaxis.set_ticks([6, 12, 18, 24])
    ax.xaxis.set_ticks([10, 20, 30])
    ax.yaxis.set_tick_params(labelsize=13)
    ax.xaxis.set_tick_params(labelsize=13)

    ax.set_frame_on(False)


def layout_heatmap(data):
    titles = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    fig, axes = plt.subplots(3, 4)
    minimo = data['PM2.5'].min()
    maximo = data['PM2.5'].max()

    for j, month in enumerate(range(1, 13)):
        ax1 = floor(j/4)
        ax2 = j % 4
        subset = data[(data['Time'].dt.month == month)]
        single_plot(subset, titles[j], axes[ax1, ax2], maximo, minimo)

    fig.subplots_adjust(left=0.18, right=0.8, top=0.9, hspace=0.25, wspace=0.15)
    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.025])
    norm = mc.Normalize(minimo, maximo)
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap='magma'),
        cax=cbar_ax,
        orientation='horizontal'
    )
    cb.ax.xaxis.set_tick_params(size=0)
    cb.set_label('PM2.5', size=12)
    plt.suptitle('PM2.5 in 2021 per month', fontsize=25)
    plt.show()


def clustered_heatmap(data):
    # reorder data
    data = data.iloc[::-1]
    # choose a month
    data = data[data['Time'].dt.month == 3]
    days = len(data['Day'].unique())

    data['Day'] = data['Time'].dt.day
    data['PM2.5'] = data['PM2.5'].fillna(0)
    data['Hour'] = data['Hour'].astype('int64')

    dayweek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    y_label = [''] * days

    for i in data['Time'].dt.day.unique():
        y_label[i-1] = str(i) + ' ' + dayweek[i % 7]

    data = data[['PM2.5']]
    data_numpy = data.to_numpy()
    matrix = np.resize(data_numpy, (days, 24))
    mask = np.invert(np.array(matrix, dtype=bool))

    # make row_cluster and col_cluster True to plot the clustered heatmap, False otherwise
    hm = sns.clustermap(matrix, annot=True, mask=mask, row_cluster=False, col_cluster=False, yticklabels=y_label,
                        xticklabels=range(0, 24))

    hm.fig.suptitle('Clustered PM2.5 in March', fontsize=22, x=0.73)
    plt.show()


def main():
    avda_tolosa = read_pm('AVDA_TOLOSA.csv')
    pm_comparison(avda_tolosa)


main()
