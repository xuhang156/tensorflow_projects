from keras import models
from keras import layers
from keras import optimizers
from keras.datasets import imdb,reuters,boston_housing

## 卷积神经网络:用于分类问题
## 150 * 150 * 3 彩色图像
def create_binary_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape =(150,150,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation = 'relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation = 'relu'))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation = 'relu'))
    model.add(layers.Dense(1,activation = 'sigmoid'))
    model.compile(optimizer = optimizers.RMSprop(lr=1e-4) , loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(model.summary())
    return model


## 全连接神经网络:用于二分类问题
## 使用Dropout层防止过拟合，随机丢弃神经元
def create_dropout_binary_model():
    model = models.Sequential()
    model.add(layers.Dense(16,activation = 'relu', input_shape =(10000,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16,activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1,activation = 'sigmoid'))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
    return model
