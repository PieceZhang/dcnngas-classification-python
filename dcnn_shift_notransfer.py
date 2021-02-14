from tensorflow.python.keras import models as km
from utils_data import loadmat_1, acc_calc
import utils_network as net
import numpy as np
import time


def shift1_fit(matdir):
    """
    一块训练，一块测试
    :return:
    """
    for sbatch in range(1, 11):
        sdata, slabel = loadmat_1(matdir, batch=sbatch, shuffle=True, split=1)
        '''Train Network & Save Model'''
        model = net.network_2(summary=False)
        model.fit(sdata, slabel, batch_size=100, epochs=80, verbose=1,
                  callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
                  class_weight=None, sample_weight=None, initial_epoch=0)
        model.save('./dcnn_{}.h5'.format(sbatch))


def shift1_predict(matdir):
    """
    一块训练，一块测试
    :return:
    """
    accrecord = np.ndarray((10, 10))
    for sbatch in range(1, 11):
        model = km.load_model('./dcnn_{}.h5'.format(sbatch))
        for tbatch in range(1, 11):
            tdata, tlabel = loadmat_1(matdir, batch=tbatch, shuffle=True, split=0)
            result = model.predict(tdata)
            acc = acc_calc(tlabel, result)
            accrecord[sbatch - 1, tbatch - 1] = acc[6]
            print("\n=================validation=================\n")
            print("S domain: {} \nT domain: {}".format(sbatch, tbatch))
            print("Accuracy: {}".format(acc[6]))
    print("\nFinished")


def shift2_fit(matdir):
    """
    九块训练，一块测试
    :return:
    """
    for tbatch in range(1, 11):
        sdata = np.ndarray((0, 8, 16, 1))
        slabel = np.ndarray((0, 6))
        for sbatch in range(1, 11):
            if sbatch != tbatch:  # 读取除当前tbatch之外的数据作为sbatch
                data, label = loadmat_1(matdir, batch=sbatch, shuffle=True, split=1)
                sdata = np.concatenate((sdata, data), axis=0)
                slabel = np.concatenate((slabel, label), axis=0)
        '''Train Network & Save Model'''
        model = net.network_2(summary=False)
        model.fit(sdata, slabel, batch_size=70, epochs=80, verbose=1,
                  callbacks=None, validation_split=0.1, validation_data=None, shuffle=True,
                  class_weight=None, sample_weight=None, initial_epoch=0)
        model.save('./dcnn_{}.h5'.format(tbatch))


def shift2_predict(matdir):
    """
    九块训练，一块测试
    :return:
    """
    accrecord = np.ndarray((1, 10))
    for tbatch in range(1, 11):
        model = km.load_model('./dcnn_{}.h5'.format(tbatch))
        tdata, tlabel = loadmat_1(matdir, batch=tbatch, shuffle=True, split=0)
        result = model.predict(tdata)
        acc = acc_calc(tlabel, result)
        accrecord[0, tbatch - 1] = acc[6]
        print("\n=================validation=================\n")
        print("T domain: {}".format(tbatch))
        print("Accuracy: {}".format(acc[6]))
    print("\nFinished")


if __name__ == '__main__':
    filepath = 'D:/A_/Enose_datasets/10board/Batch.mat'  # directory of .mat file
    # shift2_fit(filepath)
    shift2_predict(filepath)
