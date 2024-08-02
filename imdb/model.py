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
