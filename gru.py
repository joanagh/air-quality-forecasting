import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


def read_pm(station):
    path = station
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


df = read_pm('datos_2021/AVDA_TOLOSA.csv')
data2 = read_pm('datos_2022/datos_indice/AVDA_TOLOSA.csv')

df = missing_data(df)
df = df.set_index('Time')
df = df.iloc[::-1]

target = 'PM2.5'
features = list(df.columns.difference([target]))


class GRUNet(nn.Module):
    def __init__(self, num_features, hidden_units):
        super().__init__()
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_layers = 2

        self.gru = nn.GRU(
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
        _, hn = self.gru(x, h0)
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.
        return out


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


folds = 5
current_fold = 1

learning_rate = 5e-5
num_hidden_units = 8

k_loss_train = list()
k_loss_test = list()
k_forecast = list()

mse = list()
mae = list()

torch.manual_seed(1312)

for i in range(1, 12, 2):
    while current_fold <= folds:
        df_train = df[df.index.month <= (current_fold * 2)].copy()
        df_test = df[(df.index.month > (current_fold * 2)) & (df.index.month <= (current_fold * 2 + 2))].copy()

        target_max = df_train[target].max()
        target_min = df_train[target].min()

        for c in df_train.columns:
            mini = df_train[c].min()

            df_train[c] = df_train[c] - mini
            df_test[c] = df_test[c] - mini

            if (df_train[c] != 0).any():
                maxi = df_train[c].max()

                df_train[c] = df_train[c] / maxi
                df_test[c] = df_test[c] / maxi

        batch_size = 4
        sequence_length = 30

        print('TRAIN: ')
        print(df_train)
        print('TEST:')
        print(df_test)

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

        model = GRUNet(num_features=len(features), hidden_units=num_hidden_units)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()

        X, y = next(iter(train_loader))

        # print("Features shape:", X.shape)
        # print("Target shape:", y.shape)

        print("Untrained test\n--------")
        t_model(test_loader, model, loss_function)
        print()

        train_loss = list()
        test_loss = list()
        for ix_epoch in range(6):
            print(f"Epoch {ix_epoch}\n---------")
            train_loss.append(train_model(train_loader, model, loss_function, optimizer=optimizer))
            test_loss.append(t_model(test_loader, model, loss_function))
            print()

        print(train_loss, test_loss)
        k_loss_train.append(train_loss)
        k_loss_test.append(test_loss)

        train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        prediction_col = 'Forecast'
        df_train['Forecast'] = predict(train_eval_loader, model).numpy()
        df_test['Forecast'] = predict(test_loader, model).numpy()

        df_out = pd.concat((df_train, df_test))[[target, prediction_col]]

        for c in df_out.columns:
            df_out[c] = df_out[c] * target_max + target_min


        aux = df_out[(df_out.index.month > (current_fold * 2)) & (df_out.index.month <= (current_fold * 2 + 2))].copy()

        current_mse = mean_squared_error(aux['PM2.5'], aux['Forecast'])
        current_mae = mean_absolute_error(aux['PM2.5'], aux['Forecast'])

        mse.append(current_mse)
        mae.append(current_mae)

        k_forecast.append(aux['Forecast'])

        current_fold += 1
        print('-----------------------------------------------------')

print(k_loss_train)

train_loss_mean = [0] * 6
test_loss_mean = [0] * 6


for j in range(len(k_loss_train)):
    for i in range(len(k_loss_train[j])):
        print(j, i)
        print(k_loss_train[j][i])
        train_loss_mean[i] += k_loss_train[j][i] / len(k_loss_train)
        test_loss_mean[i] += k_loss_test[j][i] / len(k_loss_train)


print(mse)
print(mae)

# plt.plot(range(6), k_loss_train[i], label='train loss'+str(i))
# plt.plot(range(6), k_loss_test[i], label='test loss'+str(i))

plt.plot(train_loss_mean, label='Train loss average')
plt.plot(test_loss_mean, label='Test loss average')

plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(prop={'size': 16})
plt.title('GRU average loss', fontsize=25)
plt.show()

plt.plot(df.index, df['PM2.5'], label='Real value of PM2.5', color='black')

for i in range(len(k_forecast)):
    plt.plot(k_forecast[i].index, k_forecast[i], label='Prediction in fold '+str(i+1))

plt.xlabel('Date', fontsize=16)
plt.ylabel('PM2.5', fontsize=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(prop={'size': 16})
plt.title('GRU forecast', fontsize=25)
# plt.show()

avg_mse = mean_squared_error(df_out['PM2.5'], df_out['Forecast'])
avg_mae = mean_absolute_error(df_out['PM2.5'], df_out['Forecast'])

print('mse: ', avg_mse)
print('mae: ', avg_mae)
