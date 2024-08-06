from keras.datasets import boston_housing

import numpy as np
from keras.utils.np_utils import to_categorical
from load_imdb_data import *

# 标准化数据
def standardize_features(data,test_data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean) / std
    test_data = (test_data - mean) / std
    return data,test_data

def perform_k_fold_cv(data,targets,k = 4):
    num_val_samples = len(data) // k
    num_epochs = 100 
    all_scores = []

    for i in range(k):
        print('processing fold #',i)
        val_data = data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = targets[i * num_val_samples: (i + 1) * num_val_samples]

        part_data = np.concatenate([data[:i * num_val_samples],data[(i + 1) * num_val_samples:]],axis=0)
        part_targets = np.concatenate([targets[:i * num_val_samples],targets[(i + 1) * num_val_samples:]],axis=0)

        model = create_mse_model(val_data.shape[1])
        model.fit(part_data,part_targets,epochs=num_epochs,batch_size=1,verbose=0)

        val_mse,val_mae = model.evaluate(val_data,val_targets,verbose=0)
        all_scores.append(val_mae)

    print(all_scores)
    print('mean:',np.mean(all_scores))

if __name__ == '__main__':  
    # 测试：打乱数据集
    data = [1,2,3,4,5,6,7,8,9]
    np.random.shuffle(data)

    cur_file_path =os.path.dirname(os.path.abspath(__file__))
    check_and_download_boston_housing_file(cur_file_path)

    train_data,train_labels = load_local_pkl_file(os.path.join(cur_file_path,'boston_housing_train_data.pkl'))
    test_data,test_labels = load_local_pkl_file(os.path.join(cur_file_path,'boston_housing_test_data.pkl'))

    train_data,test_data = standardize_features(train_data,test_data)
    print(train_data[0],len(train_data[0]))

    perform_k_fold_cv(train_data,train_labels)