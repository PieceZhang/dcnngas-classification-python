"""
MAT file: 16个传感器，每个传感器产生8个数据，总共6种气体 TODO 每个传感器产生的8个数据是指什么？是时间维度上的8个数据吗？
C_data: 选取每个传感器的8个数据中的第1个数据
1.（2020）An optimized Deep Convolutional Neural Network for dendrobium classification based on electronic nose：
输入图（8,16,1）, (temporal, sensor, 1)
带状卷积核，只在temporal dimension(size=8，时间维度)上卷积，隔离相互间无关的sensor dimension(size=16，传感器维度)
但现有数据集没有时间维度，故效果有待验证
2.（2018）Gas Classification Using Deep Convolutional Neural Networks:
引入了类似残差模块通路的shortcut设计，可以避免因网络层数过多导致的梯度弥散现象
3.可能的思路:
输入图（W,H,16）, (temporal, temporal, sensor), W*H=8
可尝试不同形状卷积核，但由于输入特征图较小，效果有待验证
"""
import scipy.io as scio
import numpy as np

def loadmat_1(matdir, shuffle=True):
    """
    :param shuffle: shuffle or not
    :param matdir: directory of .mat file
    :return: [data_train, label_train, data_test, label_test] (3600,8,16,1) (3600, 6)
    """
    data = scio.loadmat(matdir)
    label = data['C_label'][0, 9].swapaxes(0, 1)   # (3600, 6)
    data = data['batch'][0, 9].reshape(3600, 16, 8, 1).swapaxes(1, 2)  # (3600, 8, 16, 1)
    if shuffle:
        state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(state)
        np.random.shuffle(label)
    return data[0:2799], label[0:2799], data[2800:], label[2800:]

def loadmat_2(matdir):
    """
    :param matdir: directory of .mat file
    :return: [data_train, label] (3600,2,4,16) (6, 3600)
    """
    data = scio.loadmat(matdir)
    label = data['C_label'][0, 9].swapaxes(0, 1)  # (6, 3600) TODO 有错误
    data = data['batch'][0, 9].reshape(3600, 16, 2, 4).swapaxes(1, 2).swapaxes(2, 3)  # (3600, 2, 4, 16)
    return data, label

def acc_calc(label, result):
    """
    :param label: (None, 6)
    :param result: (None, 6)
    :return: accuracy[7]: accuracy for 6 classes and overall accuracy
    """
    right = [0, 0, 0, 0, 0, 0]
    wrong = [0, 0, 0, 0, 0, 0]
    acc = []
    result = np.argmax(result, axis=1)  # NMS
    for index in range(result.shape[0]):
        if label[index, result[index]] == 1:
            right[result[index]] = right[result[index]] + 1
        else:
            wrong[result[index]] = wrong[result[index]] + 1  # TODO 有错误，不会影响总精度计算，但会导致各气体精度错误
    for index in range(6):
        acc.append(right[index] / (right[index] + wrong[index]))
    acc.append(sum(right) / (sum(right) + sum(wrong)))
    return acc
