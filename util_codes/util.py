
import matplotlib.pyplot as plt

def create_and_show_plt(history):
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)
    
    # 颜色列表，颜色数目要至少等于你要绘制的曲线数
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 蓝, 绿, 红, 青, 洋红, 黄, 黑

    # 绘制每条曲线
    for idx, key in enumerate(history_dict.keys()):
        values = history_dict[key]
        color = colors[idx % len(colors)]  # 循环使用颜色列表
        plt.plot(epochs, values, color, label=key)
        
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.legend()
    plt.show()