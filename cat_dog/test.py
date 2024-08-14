
from util_codes import *
import matplotlib.pyplot as plt
from keras.utils import image_utils
from keras.preprocessing.image import ImageDataGenerator

## 测试数据增强功能，用于生成图片

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_cats_dir = 'D:\\projects\\mnist\\cat_dog\\cats_and_dogs_small\\train\\cats'
fnames = [ os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
image_path = fnames[3]
img = image_utils.load_img(image_path, target_size=(150, 150))
x = image_utils.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image_utils.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
print('done')
