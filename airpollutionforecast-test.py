#coding=utf-8

###多变量的lstm模型来预测pm2.5值。（即空气污染状况）   1、转换成有监督数据,用历史的污染数据、天气数据预测当前时刻的污染值2、数据归一化


# 1、数据划分成训练和测试数据
# 本教程用第一年数据做训练，剩余4年数据做评估
# 2、输入=1时间步长，8个feature
# 3、第一层隐藏层节点=50，输出节点=1
# 4、用平均绝对误差MAE做损失函数、Adam的随机梯度下降做优化
# 5、epoch=50, batch_size=72
# convert series to supervised learning
#模型评估
# 1、预测后需要做逆缩放
# 2、用RMSE做评估
# from math import sqrt
# from numpy import concatenate
# from matplotlib import pyplot
# from pandas import read_csv
# from pandas import DataFrame
# from pandas import concat
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_squared_error
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
#
# #转成有监督数据
# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = DataFrame(data)
#     print df.head(5)
#     cols, names = list(), list()
#     #数据序列(也将就是input) (t-n, ... t-1)
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names+=[('var%d(t-%d)'%(j+1, i)) for j in range(n_vars)]
#     #预测数据（input对应的输出值）(t, t+1, ... t+n)
#     for i in range(0, n_out, 1):
#         cols.append(df.shift(-i))
#         if i==0:
#             names+=[('var%d(t)'%(j+1)) for j in range(n_vars)]
#         else:
#             names+=[('var%d(t+%d))'%(j+1, i)) for j in range(n_vars)]
#     #拼接
#     agg = concat(cols, axis=1)
#     agg.columns = names
#     print agg.head(5)
#     # drop rows with NaN values
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg
#
#
# #数据预处理
# #--------------------------
# dataset = read_csv('pollution.csv', header=0, index_col=0)
# values = dataset.values
#
# #标签编码
# encoder = LabelEncoder()
# values[:,4] = encoder.fit_transform(values[:,4])
# #保证为float
# values = values.astype('float32')
# #归一化
# scaler = MinMaxScaler(feature_range=(0,1))
# scaled = scaler.fit_transform(values)
# #转成有监督数据
# reframed = series_to_supervised(scaled, 1, 1)
# print 'a'
# print reframed.head(5)
# #删除不预测的列
# #reframed.drop(reframed.columns[9:16], axis=1, inplace=True)
# reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
# print 'b'
# print reframed.head()
#
# #数据准备
# #--------------------------
# values = reframed.values
# n_train_hours = 365*24 #拿一年的时间长度训练
# #划分训练数据和测试数据
# train = values[:n_train_hours, :]
# test = values[n_train_hours:, :]
# #拆分输入输出
# train_x, train_y = train[:, :-1], train[:, -1]
# test_x, test_y = test[:, :-1], test[:, -1]
# #reshape输入为LSTM的输入格式
# train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
# test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
# print 'train_x.shape, train_y.shape, test_x.shape, test_y.shape'
# print train_x.shape, train_y.shape, test_x.shape, test_y.shape
#
# #模型定义
# #-------------------------
# model = Sequential()
# model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
#
# #模型训练
# #------------------------
# history = model.fit(train_x, train_y, epochs=50, batch_size=72, validation_data=(test_x, test_y), verbose=2, shuffle=False)
#
# #输出
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.ylabel('loss')
# pyplot.xlabel('epoch')
# pyplot.legend()
# pyplot.show()
#
# #预测
# #------------------------
# yhat = model.predict(test_x)
# test_x = test_x.reshape(test_x.shape[0], test_x.shape[2])
# #预测数据逆缩放
# inv_yhat = concatenate((yhat, test_x[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:, 0]
# #真实数据逆缩放
# test_y = test_y.reshape(len(test_y), 1)
# inv_y = concatenate((test_y, test_x[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, 0]
# #计算rmse
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print 'Test RMSE:%.3f'%rmse


############以下对上述模型进行了改进，把3小时的数据作为输入，此时，输入输出有所改变，输入为3*8+8=24列作为前3小时所有特星的输入，将污染变量作为下一小时的产出

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import h5py
from keras.models import load_model

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# load dataset
dataset = read_csv('train.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag hours
n_hours = 24
n_features = 8
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)
print(reframed.shape)

# split into train and test sets
values = reframed.values
n_train_hours = 365*3 * 24
test = values
# split into input and outputs
n_obs = n_hours * n_features
test_X, test_y = test[:, :n_obs], test[:, -n_features]
# reshape input to be 3D [samples, timesteps, features]
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(test_X.shape, test_y.shape)




# make a prediction
# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')

yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

pyplot.figure(2)
#绘图
pyplot.plot(inv_yhat[0:4000],label='predictive value')
pyplot.plot(inv_y[0:4000],label='true value')
pyplot.legend()
pyplot.show()

pyplot.figure(3)
#绘图
pyplot.plot(inv_yhat[4000:], label='predictive value')
pyplot.plot(inv_y[4000:],label='true value')
pyplot.legend()
pyplot.show()


# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
