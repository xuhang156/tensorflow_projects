from keras import models
from keras import layers
from keras import regularizers

def create_model():
    model = models.Sequential()
    model.add(layers.Dense(16,activation = 'relu', input_shape =(10000,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16,activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1,activation = 'sigmoid'))
    # rmsprop优化器和binary_crossentropy损失函数
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
    return model

def create_reuters_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation = 'relu', input_shape =(10000,)))
    model.add(layers.Dense(64,activation = 'relu'))
    model.add(layers.Dense(46,activation = 'softmax'))
    # rmsprop优化器和categorical_crossentropy损失函数
    # one hot编码的标签可以使用上述损失函数，如果不是则使用sparse_categorical_crossentropy损失函数
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])
    return model


def create_mse_model(shape):
    model = models.Sequential()
    model.add(layers.Dense(64,activation = 'relu', input_shape =(shape,)))
    model.add(layers.Dense(64,activation = 'relu'))
    # 最后一层只是一个单元，没有激活，是一个线性层，这是标量回归（预测单一连续值的回归）的典型设置
    # 添加激活函数将会限制输出范围
    model.add(layers.Dense(1))
    # mse：均方误差，指预测值与目标值之差的平法，这是回归问题常用的损失函数
    # mae：平均绝对误差，指预测值与目标值之差的绝对值。
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model