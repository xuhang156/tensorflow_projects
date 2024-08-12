from prepare_the_data import *
import util_codes


if __name__ == '__main__':
    cur_file_path = os.path.dirname(os.path.abspath(__file__))
    original_dataset_dir = os.path.join('./','dogs-vs-cats','train','train')
    base_dir = os.path.join(cur_file_path,'cats_and_dogs_small')
    folder = CatsAndDogsFolder(base_dir)

    model = create_binary_cnn_model()
    train_generator, validation_generator = load_images(folder.train_dir, folder.validation_dir)

    for data_batch, labels_batch in train_generator:
        print('data batch shape:', data_batch.shape)
        print('labels batch shape:', labels_batch.shape)
        break
    
    history = model.fit_generator(train_generator, steps_per_epoch= 100,epochs= 30, validation_data = validation_generator, validation_steps = 50)
    util_codes.cm.create_and_show_plt(history)
    model.save(os.path.join(cur_file_path,'cats_and_dogs_small_1.h5'))