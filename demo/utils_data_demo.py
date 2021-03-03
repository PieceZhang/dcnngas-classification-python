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
    d = loadmat_1('D:/A_/Enose_datasets/5board/board_lite.mat', split=1)
    print()
