import numpy as np  
from keras.utils import to_categorical  # 修改此行  
import gzip  
import os  

# Define file paths  
train_images_path = './MNIST/train-images-idx3-ubyte.gz'  
train_labels_path = './MNIST/train-labels-idx1-ubyte.gz'  
test_images_path = './MNIST/t10k-images-idx3-ubyte.gz'  
test_labels_path = './MNIST/t10k-labels-idx1-ubyte.gz'  

def binaryzation_images(images):
    shape = images.shape
    width = shape[1]
    height = shape[2]
    images = images.reshape((shape[0],width * height))
    images = images.astype('float32') /255
    return images

def load_mnist_images(file_path):  
    with gzip.open(file_path, 'rb') as f:  
        # Skip the magic number and dimensions  
        f.read(16)  
        # Read the image data  
        images = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32)  
        images = images.reshape(-1, 28, 28)  
        images /= 255.0  # Normalize to [0, 1]  
    return images  

def load_mnist_labels(file_path):  
    with gzip.open(file_path, 'rb') as f:  
        # Skip the magic number  
        f.read(8)  
        # Read the label data  
        labels = np.frombuffer(f.read(), dtype=np.uint8)  
    return labels  

# Load data  
train_images = load_mnist_images(train_images_path)  
train_labels = load_mnist_labels(train_labels_path)  
test_images = load_mnist_images(test_images_path)  
test_labels = load_mnist_labels(test_labels_path)  

# Convert labels to one-hot encoding  
train_labels = to_categorical(train_labels, 10)  # 修改此行  
test_labels = to_categorical(test_labels, 10)    # 修改此行  

print(f"Training data shape: {train_images.shape}")  
print(f"Training labels shape: {train_labels.shape}")  
print(f"Training labels (example): {train_labels[:5]}")  # 只打印前5个  
print(f"Test data shape: {test_images.shape}")  
print(f"Test labels shape: {test_labels.shape}")

train_images = binaryzation_images(train_images)
print(f"Training data shape: {train_images.shape}")  
print(f"Training labels shape: {train_labels.shape}")  