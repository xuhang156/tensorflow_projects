from keras import models
from keras import layers

## 创建了两侧网络，是一种密集型链接（全连接）
## 第二层即最后一层上一个10路的softmax层，用于输出10个数字的概率
def create_network():
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def compile_network(model):
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])