"""
10 boards:
    MAT file: 16个传感器，每个传感器产生8个数据，总共6种气体
    C_data: 选取每个传感器的8个数据中的第1个数据
5 boards:
    5board_lite.mat: 修剪后的数据集
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


def loadmat_1(matdir, batch, shuffle=True, split=0.8):
    """
    10 boards
    :param shuffle: shuffle or not
    :param matdir: directory of .mat file
    :param batch: select 1~10 batch
    :param split: validation split
    :return: [sdata_train, label_train, data_test, label_test] (?,8,16,1) (?, 6)
    """
    assert batch > 0 & batch < 11, "Please input correct batch number"
    data = scio.loadmat(matdir)
    label = data['C_label'][0, batch - 1].swapaxes(0, 1)  # (?, 6)
    length = label.shape[0]
    data = data['batch'][0, batch - 1].reshape(length, 16, 8, 1).swapaxes(1, 2)  # (?, 8, 16, 1)
    if shuffle:
        state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(state)
        np.random.shuffle(label)
    if split == 1 or split == 0:
        return data[0:length], label[0:length]
    else:
        return data[0:int(length * split)], label[0:int(length * split)], data[int(length * split) + 1:], label[int(
            length * split) + 1:]


def loadmat_1_SDA(matdir, batch=None, shuffle=True, split=0.8):
    """
    load data for SDA (?, 1, 128)
    :param batch: select batch
    :param shuffle: shuffle or not
    :param matdir: directory of .mat file
    :param split: validation split
    :return: [sdata_train, label_train, data_test, label_test] (?,8,16,1) (?, 6)
    """
    if batch is None:
        batch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    label = np.empty([0, 6])
    data = np.empty([0, 1, 128])
    origindata = scio.loadmat(matdir)
    for index in batch:
        clabel = origindata['C_label'][0, index - 1].swapaxes(0, 1)  # (?, 6)
        length = clabel.shape[0]
        cdata = origindata['batch'][0, index - 1].reshape(length, 128, 1).swapaxes(1, 2)  # (?, 1, 128)
        label = np.concatenate((label, clabel))
        data = np.concatenate((data, cdata))
    if shuffle:
        state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(state)
        np.random.shuffle(label)
    length = label.shape[0]
    if split == 1 or split == 0:
        return data[0:length], label[0:length]
    else:
        return data[0:int(length * split)], label[0:int(length * split)], data[int(length * split) + 1:], label[int(
            length * split) + 1:]


def loadmat_2(matdir, shuffle=True, split=0.8):
    """
    5 boards (600s时间序列，只取后8列的前300s，30000个数据)
    :param shuffle: shuffle or not
    :param matdir: directory of .mat file
    :param split: validation split, ex. split=0.9, 90% data for training, others for testing
    :return: [sdata_train, label_train, data_test, label_test] (?,300,8,1) (?, 6)
    """
    tempt = scio.loadmat(matdir)
    tempt = np.concatenate((tempt['board1lite'][0, 0][:], tempt['board2lite'][0, 0][:], tempt['board3lite'][0, 0][:],
                            tempt['board4lite'][0, 0][:], tempt['board5lite'][0, 0][:]), axis=1).swapaxes(0, 1)
    data = np.ndarray((640, 300, 8, 1))
    for index, item in enumerate(tempt):
        data[index] = item[0].reshape(300, 8, 1)
    label = np.zeros((640, 4), dtype=int)
    label[0:40, 0] = label[160:160 + 40, 0] = label[320:320 + 40, 0] = \
        label[480:480 + 20, 0] = label[560:560 + 20, 0] = 1
    label[40:80, 1] = label[160 + 40:160 + 80, 1] = label[320 + 40:320 + 80, 1] = \
        label[480 + 20:480 + 40, 1] = label[560 + 20:560 + 40, 1] = 1
    label[80:120, 2] = label[160 + 80:160 + 120, 2] = label[320 + 80:320 + 120, 2] = \
        label[480 + 40:480 + 60, 2] = label[560 + 40:560 + 60, 2] = 1
    label[120:160, 3] = label[160 + 120:160 + 160, 3] = label[320 + 120:320 + 160, 3] = \
        label[480 + 60:480 + 80, 3] = label[560 + 60:560 + 80, 3] = 1
    if shuffle:
        state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(state)
        np.random.shuffle(label)
    if split == 1 or split == 0:
        return data[:], label[:]
    else:
        length = label.shape[0]
        return data[0:int(length * split)], label[0:int(length * split)], \
               data[int(length * split) + 1:], label[int(length * split) + 1:]


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
        if right[index] + wrong[index] != 0:
            acc.append(right[index] / (right[index] + wrong[index]))
        else:
            acc.append(0)
    acc.append(sum(right) / (sum(right) + sum(wrong)))
    return acc


if __name__ == '__main__':
    # for debugging
    d = loadmat_2('D:/A_/Enose_datasets/5board/board_lite.mat', split=1)
    print()
