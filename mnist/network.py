from keras import models
from keras import layers

## 卷积神经网络用于手写字体识别
def create_conv2d_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape =(28,28,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
    print(model.summary())

    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation = 'relu'))
    model.add(layers.Dense(10,activation = 'softmax'))
    return model


## 创建了两侧网络，是一种密集型链接（全连接）
## 第二层即最后一层上一个10路的softmax层，用于输出10个数字的概率
def create_network():
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def compile_network(model):
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])