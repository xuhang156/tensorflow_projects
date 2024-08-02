import os
import pickle
import numpy as np
from keras.datasets import imdb
import matplotlib.pyplot as plt

from model import create_model


def check_and_download_file(save_path):
    if 'imdb_train_data.pkl' not in os.listdir(path=save_path):
        # num_worlds: 仅保留训练数据中前10000 个最长出现的单词，低频单词被舍弃
        (train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
        with open(os.path.join(save_path,'imdb_train_data.pkl'),'wb') as file:
            pickle.dump((train_data,train_labels),file)
        
        with open(os.path.join(save_path,'imdb_test_data.pkl'),'wb') as file:
            pickle.dump((test_data,test_labels),file)

def load_local_pkl_file(path):
    with open(path, 'rb') as train_data_file:
        data, labels = pickle.load(train_data_file)
    return (data, labels)

def load_word_index():
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])     
    return reverse_word_index

def decoded_review(reverse_word_index,train_data):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data])

# 创建一个形状为(len(sequences),dimesion)的零矩阵
def vectorize_sequences(sequences,dimension = 10000):
    results = np.zeros((len(sequences),dimension))
    for i ,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results


if __name__ == '__main__':  
    cur_file_path =os.path.dirname(os.path.abspath(__file__))
    word_index = load_word_index()
    check_and_download_file(cur_file_path)

    train_data,train_labels = load_local_pkl_file(os.path.join(cur_file_path,'imdb_train_data.pkl'))
    test_data,test_labels = load_local_pkl_file(os.path.join(cur_file_path,'imdb_test_data.pkl'))
    print(train_data[0],len(train_data[0]))
    print(train_labels[0])
    print(max([max(sequence) for sequence in train_data]))

    print(decoded_review(word_index,train_data[2]))

    x_train = vectorize_sequences(train_data)
    x_test  = vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype('float32')
    y_test =  np.asarray(test_labels).astype('float32')

    model = create_model()
    history = model.fit(x_train,y_train,epochs=4,batch_size=512)
    results = model.evaluate(x_test,y_test)
    print(results)

    model_save_path = os.path.join(cur_file_path, 'trained_imdb_model.h5')
    model.save(model_save_path)
    
    # part_test = x_test[:100]
    # test_rate = model.predict(x_test)
    # part_test_rate = model.predict(part_test)
    # print(model.predict(x_test))
    
    
    # history_dict = history.history
    # print(history_dict.keys())

    # loss_values = history_dict['loss']
    # val_loss_values = history_dict['val_loss']

    # epochs = range(1,len(loss_values) + 1)

    # plt.plot(epochs,loss_values, 'bo',label='Training loss')
    # plt.plot(epochs,val_loss_values, 'b',label='Validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()


