import numpy as np

def show_np_array(array = np.array(0)):
    print(array)
    print(array.ndim)
    print(array.shape)
    print(array.dtype)

# 张量切片示例
def test_tensor_slicing():
    source = np.array([[1,2,3,4,5,6,7,8],
                      [11,12,13,14,15,16,17,18],
                      [21,22,23,24,25,26,27,28]])
    show_np_array(source[1:2])
    show_np_array(source[1:2,:])
    show_np_array(source[1:2,0:8])

# ReLU(x)=max(0,x)
# ReLU（Rectified Linear Unit，修正线性单元）是一种常用的激活函数。ReLU 的作用是在神经网络的每一层中为神经元提供非线性变换，使模型能够更好地学习复杂的模式和数据特征。
def naive_relu(x):
    assert len(x.shape) == 2

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = max(x[i,j],0)
    return x

# 2D张量的逐元素加法运算，非并发计算
def naive_add(x,y):
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] += y[i,j]

    

def tensor_broadcast_test():
    x = np.random.random((1,3,2,2))
    y = np.random.random((2,2))

    z = np.maximum(x,y)

    show_np_array(x)
    show_np_array(y)
    show_np_array(z)

if __name__ == '__main__':  
    ## array在这里可以表示一个数字，一个数字就是一个标量
    tensor_broadcast_test()
    show_np_array(np.array(12))
    show_np_array(np.array([1,23,45,3,64]))
    show_np_array(np.array([[1,23,45,3,64],[3,3,56,33,4]]))

    test_tensor_slicing()