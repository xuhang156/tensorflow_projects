import os,shutil

class CatsAndDogsFolder:
    def __init__(self,base_dir):
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir,'train')
        self.validation_dir = os.path.join(base_dir,'validation')
        self.test_dir = os.path.join(base_dir,'test')

        self.train_cats_dir = os.path.join(self.train_dir,'cats')
        self.train_dogs_dir = os.path.join(self.train_dir,'dogs')

        self.validation_cats_dir = os.path.join(self.validation_dir,'cats')
        self.validation_dogs_dir = os.path.join(self.validation_dir,'dogs')

        self.test_cats_dir = os.path.join(self.test_dir,'cats')
        self.test_dogs_dir = os.path.join(self.test_dir,'dogs')

    def create(self):
        for path in [self.base_dir,self.train_dir,self.validation_dir,self.test_dir,self.train_cats_dir,self.train_dogs_dir,self.validation_cats_dir,self.validation_dogs_dir,self.test_cats_dir,self.test_dogs_dir]:
            try:
                os.mkdir(path)
            except:
                pass

    def copy_images(self,src_dir,dst_dir,begin_number,end_number,name_prefix):
        shutil.rmtree(dst_dir)
        os.mkdir(dst_dir)
        for i in range(begin_number,end_number):
            name = '{}.{}.jpg'.format(name_prefix,i)
            src = os.path.join(src_dir,name)
            dst = os.path.join(dst_dir,name)
            shutil.copyfile(src,dst)

    def prepare_data(self,src_dir):
        self.copy_dog_images(src_dir,1000,1500,2000,'dog')
        self.copy_cat_images(src_dir,1000,1500,2000,'cat')

    def copy_dog_images(self,src_dir,train_number,validation_number,test_number,name_prefix):
        self.copy_images(src_dir,self.train_dogs_dir,0,train_number,name_prefix)
        self.copy_images(src_dir,self.validation_dogs_dir,train_number,validation_number,name_prefix)
        self.copy_images(src_dir,self.test_dogs_dir,validation_number,test_number,name_prefix)

    def copy_cat_images(self,src_dir,train_number,validation_number,test_number,name_prefix):
        self.copy_images(src_dir,self.train_cats_dir,0,train_number,name_prefix)
        self.copy_images(src_dir,self.validation_cats_dir,train_number,validation_number,name_prefix)
        self.copy_images(src_dir,self.test_cats_dir,validation_number,test_number,name_prefix)


### 数据预处理
## 1. 读取图像文件
## 2. jpeg解码为RGB像素网格
## 3. 将像素网格转换为浮点数张量
## 4. 将像素值缩放到[0,1]之间
## 要点：使用数据增强来更多的生成图片
def load_images(train_dir,validation_dir):
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    ## batch_size:一次性输出32长图片
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=32,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=32,
        class_mode='binary'
    )
    return train_generator,validation_generator


if __name__ == '__main__':
    cur_file_path = os.path.dirname(os.path.abspath(__file__))
    original_dataset_dir = os.path.join('./','dogs-vs-cats','train','train')
    base_dir = os.path.join(cur_file_path,'cats_and_dogs_small')
    folder = CatsAndDogsFolder(base_dir)
    folder.create()
    folder.prepare_data(original_dataset_dir)
    