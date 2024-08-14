
import matplotlib.pyplot as plt
from keras.utils import image_utils

def create_and_show_plt(history):
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)
    
    # 颜色列表，颜色数目要至少等于你要绘制的曲线数
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 蓝, 绿, 红, 青, 洋红, 黄, 黑

    # 绘制每条曲线
    for idx, key in enumerate(history_dict.keys()):
        values = history_dict[key]
        color = colors[idx % len(colors)]  # 循环使用颜色列表
        plt.plot(epochs, values, color, label=key)
        
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.legend()
    plt.show()


def display_images(plt,generator,show_batch):
    size = 0
    for data_batch, labels_batch in generator:
        if show_batch  == size or show_batch < 0 :
            break
        size += 1
        plt.figure(figsize=(10, 10))
        for i in range(data_batch.shape[0]):
            plt.subplot(4, 8, i + 1)
            plt.imshow(image_utils.array_to_img(data_batch[i]))
            plt.axis('off')
        plt.show()
        
