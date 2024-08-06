from keras import models
from keras import layers

def create_model():
    model = models.Sequential()
    model.add(layers.Dense(16,activation = 'relu', input_shape =(10000,)))
    model.add(layers.Dense(16,activation = 'relu'))
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