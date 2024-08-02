import os
import numpy as np
from keras.models import load_model
from load_imdb_data import *


if __name__ == '__main__':  
    cur_file_path = os.path.dirname(os.path.abspath(__file__))
    test_data,test_labels = load_local_pkl_file(os.path.join(cur_file_path,'imdb_test_data.pkl'))
    x_test  = vectorize_sequences(test_data)
    model_path = os.path.join(cur_file_path,'trained_imdb_model.h5')
    model = load_model(model_path)
    predictions = model.predict(x_test)

    # 输出预测结果
    for i, prediction in enumerate(predictions[:5]):  # 这里只打印前5个预测结果
        print(f"Sample {i}: Prediction: {prediction[0]}, Predicted Label: {'Positive' if prediction[0] > 0.5 else 'Negative'}")
