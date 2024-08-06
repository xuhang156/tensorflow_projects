import numpy as np
from keras.utils.np_utils import to_categorical
from load_imdb_data import *


class DMOPartId:
	def __init__(self):
		self.part_id = ""
	
	def set_part_id(self,id):
		self.part_id = id

	def to_str(self):
		return f"PN(YMD1) = '{self.part_id }' \n"

if __name__ == '__main__':  
    pn = DMOPartId()
    pn.part_id = 'qr_code'
    print(pn.to_str())

    cur_file_path =os.path.dirname(os.path.abspath(__file__))
    word_index = load_word_index()
    check_and_download_reuters_file(cur_file_path)

    train_data,train_labels = load_local_pkl_file(os.path.join(cur_file_path,'reuters_train_data.pkl'))
    test_data,test_labels = load_local_pkl_file(os.path.join(cur_file_path,'reuters_test_data.pkl'))
    print(train_data[0],len(train_data[0]))
    print(decoded_review(word_index,train_data[2]))

    x_train = vectorize_sequences(train_data)
    x_val =  x_train[:1000]
    part_x_train = x_train[1000:]

    x_test  = vectorize_sequences(test_data)

    one_hot_train_labels = vectorize_sequences(train_labels,46)
    y_val = one_hot_train_labels[:1000]
    part_y_train = one_hot_train_labels[1000:]

    one_hot_test_labels = to_categorical(test_labels)

    model = create_reuters_model()
    history = model.fit(part_x_train,part_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
    
    results = model.evaluate(x_test,one_hot_test_labels)
    print(results)
    create_and_show_plt(history)

    model_save_path = os.path.join(cur_file_path, 'trained_reuters_model.h5')
    model.save(model_save_path)