from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
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
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512,activation = 'relu'))
    model.add(layers.Dense(1,activation = 'sigmoid'))
    model.compile(optimizer = optimizers.RMSprop(lr=1e-4) , loss = 'binary_crossentropy', metrics = ['acc'])
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

## 全连接神经网络:用于多分类问题
def create_fully_connected_multiclass_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation = 'relu', input_shape =(10000,)))
    model.add(layers.Dense(64,activation = 'relu'))
    model.add(layers.Dense(46,activation = 'softmax'))
    # rmsprop优化器和categorical_crossentropy损失函数
    # one hot编码的标签可以使用上述损失函数，如果不是则使用sparse_categorical_crossentropy损失函数
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])
    return 
    
## 全连接
def create_regression_model(shape):
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

## 直接提取特征，使用预训练模型进行训练，迁移学习
def vgg16_model():
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    print(conv_base.summary())
    return conv_base

## 预训练模型，此方法是不使用数据增强，将图片数据保存成numpy数组，输入到独立的密集连接器中。这种方法适用于数据集较小的情况
## 优点：速度快，计算代价低
## 缺点：不允许使用数据增强
def create_pretraining_regression_model(shape):
    model = models.Sequential()
    model.add(layers.Dense(256,activation = 'relu', input_shape =(shape,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1,activation = 'sigmoid'))
    ## optimizer='rmsprop'，默认学习率是0.001
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),loss='binary_crossentropy',metrics=['acc'])
    return model

## 扩展预训练模型conv_base，使用数据增强
## 优点：使用数据增强，训练效果更好
## 缺点：计算代价高
## 注意点：在训练过程中，不要训练预训练模型的权重，只训练新添加的层，即冻结预训练模型的权重
## trainable参数设置为False
def create_conv_base_regression_model(conv_base):
    conv_base.trainable = False
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation = 'relu'))
    model.add(layers.Dense(1,activation = 'sigmoid'))
    ## optimizer='rmsprop'，默认学习率是0.001
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),loss='binary_crossentropy',metrics=['acc'])
    return model