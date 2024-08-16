from prepare_the_data import *
from util_codes import *
import matplotlib.pyplot as plt
from keras.utils import image_utils

## 可正常使用
# if __name__ == '__main__':
    
#     cur_file_path = os.path.dirname(os.path.abspath(__file__))
#     original_dataset_dir = os.path.join('./','dogs-vs-cats','train','train')
#     base_dir = os.path.join(cur_file_path,'cats_and_dogs_small')
#     folder = CatsAndDogsFolder(base_dir)
#     model = cm.create_binary_cnn_model()
#     train_generator, validation_generator = load_images(folder.train_dir, folder.validation_dir)
#     ut.display_images(plt,train_generator,2)

#     for data_batch, labels_batch in train_generator:
#         print('data batch shape:', data_batch.shape)
#         print('labels batch shape:', labels_batch.shape)
#         break
#     print('down')
        
#     history = model.fit_generator(train_generator, steps_per_epoch= 100,epochs= 100, validation_data = validation_generator, validation_steps = 50)
#     ut.create_and_show_plt(history)
#     model.save(os.path.join(cur_file_path,'cats_and_dogs_small_1.h5'))

## 可正常使用
## 使用预训练模型进行学习，第一种方式，将图片展平为向量，直接输入到独立密集连接器中
# if __name__ == '__main__':
    
#     cur_file_path = os.path.dirname(os.path.abspath(__file__))
#     original_dataset_dir = os.path.join('./','dogs-vs-cats','train','train')
#     base_dir = os.path.join(cur_file_path,'cats_and_dogs_small')
#     folder = CatsAndDogsFolder(base_dir)
#     model = cm.vgg16_model()

#     train_features, train_labels = ut.extract_features(model,folder.train_dir,2000)
#     validation_features, validation_labels = ut.extract_features(model,folder.validation_dir,1000)
#     test_features, test_labels = ut.extract_features(model,folder.test_dir,1000)

#     #将特征为(samples,4,4,512)的矩阵展平为(samples,8192)的矩阵
#     train_features = np.reshape(train_features,(2000,4*4*512))
#     validation_features = np.reshape(validation_features,(1000,4*4*512))
#     test_features = np.reshape(test_features,(1000,4*4*512))

#     model = cm.create_pretraining_regression_model(4*4*512)

#     history = model.fit(train_features,train_labels,epochs=30,validation_data = (validation_features,validation_labels))
#     ut.create_and_show_plt(history)
#     print('down')


## 使用预训练模型进行学习，第一种方式，将图片展平为向量，直接输入到独立密集连接器中
if __name__ == '__main__':
    cur_file_path = os.path.dirname(os.path.abspath(__file__))
    original_dataset_dir = os.path.join('./','dogs-vs-cats','train','train')
    base_dir = os.path.join(cur_file_path,'cats_and_dogs_small')
    folder = CatsAndDogsFolder(base_dir)
    conv_base = cm.vgg16_model()
    model = cm.create_conv_base_regression_model(conv_base)
    train_generator, validation_generator = load_images(folder.train_dir, folder.validation_dir)

    history = model.fit_generator(train_generator, steps_per_epoch= 100,epochs= 30, validation_data = validation_generator, validation_steps = 50)
    ut.create_and_show_plt(history)
    print('down')

