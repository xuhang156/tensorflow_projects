import os
import numpy as np
from keras.models import load_model
from load_imdb_data import *


if __name__ == '__main__':  
    cur_file_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(cur_file_path,'cats_and_dogs_small_1.h5')
    model = load_model(model_path)
    # predictions = model.predict(x_test)