import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime
from dateutil.relativedelta import *



def read_pm(station):
    path = 'datos_2021/' + station
    data = pd.read_csv(path, sep=';', decimal=',')
    hour = data['Hour'].astype(int)
    hour = hour - 1
    data['Date'] = data['Date'].astype(str)
    data['Hour'] = hour.astype(str)
    aux = data['Date'] + ' ' + data['Hour']
    data['Time'] = pd.to_datetime(aux, format='%d/%m/%Y %H')

    data['Hour'] = data['Hour'].astype(float)
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


def missing_data(data):
    data = data.drop('NO2 - ICA', axis=1)
    data = data.drop('O3 - ICA', axis=1)
    data = data.drop('PM10 - ICA', axis=1)
    data = data.drop('PM2.5 - ICA', axis=1)
    data = data.drop('SO2 - ICA', axis=1)
    data = data.drop('ICA Estacion', axis=1)
    data = data.drop('Hour', axis=1)
    data = data.drop('Date', axis=1)
    data = data.drop('PM10', axis=1)

    # data['Hour'] = data['Hour'].fillna(data['Hour'].mean())
    data['NOX'] = data['NOX'].fillna(data['NOX'].mean())
    data['NO'] = data['NO'].fillna(data['NO'].mean())
    data['NO2'] = data['NO2'].fillna(data['NO2'].mean())
    data['O3'] = data['O3'].fillna(data['O3'].mean())
    data['OZONO'] = data['OZONO'].fillna(data['OZONO'].mean())
    data['O3 8h'] = data['O3 8h'].fillna(data['O3 8h'].mean())
    # data['PM10'] = data['PM10'].fillna(data['PM10'].mean())
    # data['PM2.5'] = data['PM2.5'].fillna(data['PM2.5'].mean())
    data['SO2'] = data['SO2'].fillna(data['SO2'].mean())
    data['D,vien'] = data['D,vien'].fillna(data['D,vien'].mean())
    data['H'] = data['H'].fillna(data['H'].mean())
    data['P'] = data['P'].fillna(data['P'].mean())
    data['Precipitacion'] = data['Precipitacion'].fillna(data['Precipitacion'].mean())
    data['R'] = data['R'].fillna(data['R'].mean())
    data['T'] = data['T'].fillna(data['T'].mean())
    data['V,vien'] = data['V,vien'].fillna(data['V,vien'].mean())

    data = data.dropna(axis='index')
    return data


df = read_pm('AVDA_TOLOSA.csv')
df = missing_data(df)
df = df.set_index('Time')
df = df.iloc[::-1]

target = 'PM2.5'
features = list(df.columns.difference([target]))

# scaler = MinMaxScaler()
# df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
# print(df.head())

# aux = '2020-04-01'
# df = df.loc[aux:]

target_mean = df[target].mean()
target_stdev = df[target].std()
"""
for c in df.columns:
    mean = df[c].mean()
    stdev = df[c].std()

    df[c] = (df[c] - mean) / stdev

"""
class TimeBasedCV(object):
    def __init__(self, train_period=30, test_period=7, freq='days'):
        self.train_period = train_period
        self.test_period = test_period
        self.freq = freq

    def split(self, data, validation_split_date=None, date_column='record_date', gap=0):

        train_indices_list = []
        test_indices_list = []

        if validation_split_date is None:
            validation_split_date = data.index.min().date() + eval(
                'relativedelta(' + self.freq + '=self.train_period)')

        start_train = validation_split_date - eval('relativedelta(' + self.freq + '=self.train_period)')
        end_train = start_train + eval('relativedelta(' + self.freq + '=self.train_period)')
        start_test = end_train + eval('relativedelta(' + self.freq + '=gap)')
        end_test = start_test + eval('relativedelta(' + self.freq + '=self.test_period)')

        while end_test < data.index.max().date():
            cur_train_indices = list(data[(data.index.date >= start_train) &
                                              (data.index.date < end_train)].index)
            cur_test_indices = list(data[(data.index.date >= start_test) &
                                             (data.index.date < end_test)].index)

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


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


class LSTMNet(nn.Module):
    def __init__(self, num_features, hidden_units):
        super().__init__()
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_layers = 2

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=0.2
        )
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.
        return out

def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print('Train loss: ', avg_loss)
    return avg_loss


def t_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print('Test loss: ', avg_loss)
    return avg_loss


def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
           y_hat = model(X)
           output = torch.cat((output, y_hat), 0)
    return output


tscv = TimeBasedCV(train_period=30, test_period=7, freq='days')
torch.manual_seed(101)

learning_rate = 5e-5
num_hidden_units = 8

model = LSTMNet(num_features=len(features), hidden_units=num_hidden_units)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for train_index, test_index in tscv.split(df, validation_split_date=datetime.date(2021, 3, 2)):
    df_train = df.loc[train_index]
    df_test = df.loc[test_index]

    batch_size = 4
    sequence_length = 30

    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        sequence_length=sequence_length
    )
    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        sequence_length=sequence_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    X, y = next(iter(train_loader))

    print("Untrained test\n--------")
    t_model(test_loader, model, loss_function)
    print()

    train_loss = list()
    test_loss = list()
    for ix_epoch in range(2):
        print(f"Epoch {ix_epoch}\n---------")
        train_loss.append(train_model(train_loader, model, loss_function, optimizer=optimizer))
        test_loss.append(t_model(test_loader, model, loss_function))
        print()

    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    prediction_col = 'Forecast'
    df_train['Forecast'] = predict(train_eval_loader, model).numpy()
    df_test['Forecast'] = predict(test_loader, model).numpy()

    df_out = pd.concat((df_train, df_test))[[target, prediction_col]]

    print(df_out[['PM2.5', 'Forecast']])
    """
    for c in df_out.columns:
        df_out[c] = df_out[c] * target_stdev + target_mean
    """

for i in range(len(train_loss)):
    plt.plot(train_loss[i], color='blue', label='train loss')
    plt.plot(test_loss[i], color='red', label='test loss')

plt.legend()
plt.show()

plt.plot(df_out['PM2.5'], color='blue', label='PM2.5')
plt.plot(df_out['Forecast'], color='red', label='forecast')

plt.legend()
plt.show()
