import matplotlib.pyplot as plt

def show_image(images_array):
    plt.imshow(images_array,cmap=plt.cm.binary)
    plt.show()