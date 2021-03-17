from tensorflow.python.keras import models as km
from tensorflow.python import keras
from utils_data import loadmat_1, acc_calc, loadmat_3
from utils_arcface import create_custom_objects
import utils_network as net
import numpy as np


def shift1_fit():
    """
    一块训练，一块测试
    :return:
    """
    for sbatch in range(1, 11):
        sdata, slabel = loadmat_1(batch=sbatch, shuffle=True, split=1)
        '''Train Network & Save Model'''
        model = net.network_1b()
        model.fit(sdata, slabel, batch_size=100, epochs=80, verbose=1,
                  callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
                  class_weight=None, sample_weight=None, initial_epoch=0)
        model.save('./dcnn_{}.h5'.format(sbatch))


def shift1_predict():
    """
    一块训练，一块测试
    :return:
    """
    accrecord = np.ndarray((10, 10))
    for sbatch in range(1, 11):
        model = km.load_model('./dcnn_{}.h5'.format(sbatch))
        for tbatch in range(1, 11):
            tdata, tlabel = loadmat_1(batch=tbatch, shuffle=True, split=0)
            result = model.predict(tdata)
            acc = acc_calc(tlabel, result)
            accrecord[sbatch - 1, tbatch - 1] = acc[6]
            print("\n=================validation=================\n")
            print("S domain: {} \nT domain: {}".format(sbatch, tbatch))
            print("Accuracy: {}".format(acc[6]))
    print("\nFinished")


def shift2_fit():
    """
    九块训练，一块测试
    :return:
    """
    for tbatch in range(1, 11):
        sdata = np.ndarray((0, 8, 16, 1))
        slabel = np.ndarray((0, 6))
        for sbatch in range(1, 11):
            if sbatch != tbatch:  # 读取除当前tbatch之外的数据作为sbatch
                data, label = loadmat_1(batch=sbatch, shuffle=True, split=1)
                sdata = np.concatenate((sdata, data), axis=0)
                slabel = np.concatenate((slabel, label), axis=0)
        '''Train Network & Save Model'''
        model = net.network_2(summary=False)
        model.fit(sdata, slabel, batch_size=100, epochs=40, verbose=1,
                  callbacks=None, validation_split=0, validation_data=None, shuffle=True,
                  class_weight=None, sample_weight=None, initial_epoch=0)
        model.save('./dcnn_{}.h5'.format(tbatch))


def shift2_onlyprevious_fit():
    """
    trained the classifier with data from only the previous month and tested it on the current month.
    :return:
    """
    callback = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.005, patience=30, verbose=1, mode='auto')
    for tbatch in range(2, 11):
        sdata = np.ndarray((0, 8, 16, 1))
        slabel = np.ndarray((0, 6))
        # tdata, tlabel = loadmat_1(matdir, batch=tbatch, shuffle=True, split=1)  # 读取tdata用于验证
        for sbatch in range(1, tbatch):  # 读取当前tbatch之前的数据作为sbatch
            data, label = loadmat_1(batch=sbatch, shuffle=True, split=1)
            sdata = np.concatenate((sdata, data), axis=0)
            slabel = np.concatenate((slabel, label), axis=0)
        '''Train Network & Save Model'''
        model = net.network_1a()
        model.fit(sdata, slabel, batch_size=100, epochs=80, verbose=1,
                  callbacks=[callback], validation_split=0, validation_data=None, shuffle=True,
                  class_weight=None, sample_weight=None, initial_epoch=0)
        model.save('./dcnn_{}.h5'.format(tbatch))


def shift2_predict():
    """
    九块训练，一块测试
    :return:
    """
    accrecord = np.ndarray((1, 10))
    for tbatch in range(1, 11):
        try:
            model = km.load_model('./dcnn_{}.h5'.format(tbatch))
        except OSError:
            print('./dcnn_{}.h5 is not found! skipping...'.format(tbatch))
        else:
            tdata, tlabel = loadmat_1(batch=tbatch, shuffle=True, split=0)
            result = model.predict(tdata)
            acc = acc_calc(tlabel, result)
            accrecord[0, tbatch - 1] = acc[6]
            print("\n=================validation=================\n")
            print("T domain: {}".format(tbatch))
            print("Accuracy: {}".format(acc[6]))
    print("\nFinished")


def shift2_onlyprevious_arcface():
    """
    trained the classifier with data from only the previous month and tested it on the current month.
    :return:
    """
    callback = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.005, patience=30, verbose=1, mode='auto')
    accrecord = np.ndarray((1, 10))
    for tbatch in range(2, 11):
        sdata = np.ndarray((0, 8, 16, 1))
        slabel = np.ndarray((0, 6))
        # tdata, tlabel = loadmat_1(matdir, batch=tbatch, shuffle=True, split=1)  # 读取tdata用于验证
        for sbatch in range(1, tbatch):  # 读取当前tbatch之前的数据作为sbatch
            data, label = loadmat_1(batch=sbatch, shuffle=True, split=1)
            sdata = np.concatenate((sdata, data), axis=0)
            slabel = np.concatenate((slabel, label), axis=0)
        '''Train Network & Save Model'''
        model = net.network_1a_arcface()
        model.fit([sdata, slabel], slabel, batch_size=100, epochs=80, verbose=1,
                  callbacks=[callback], validation_split=0, validation_data=None, shuffle=True,
                  class_weight=None, sample_weight=None, initial_epoch=0)

        tdata, tlabel = loadmat_1(batch=tbatch, shuffle=True, split=0)
        result = model.predict([tdata, tlabel])
        acc = acc_calc(tlabel, result)
        accrecord[0, tbatch - 1] = acc[6]
        print("\n=================validation=================\n")
        print("T domain: {}".format(tbatch))
        print("Accuracy: {}".format(acc[6]))
    print(result)


def shift3_onlyprevious():
    """
    trained the classifier with data from only the previous month and tested it on the current month.
    :return:
    """
    callback = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.003, patience=30, verbose=1, mode='auto')
    accrecord = np.ndarray((1, 10))
    for tbatch in range(2, 11):
        sdata = np.ndarray((16, 0, 3, 6, 1))
        slabel = np.ndarray((0, 6))

        for sbatch in range(1, tbatch):  # 读取当前tbatch之前的数据作为sbatch
            data, label = loadmat_3(batch=sbatch, shuffle=True)
            sdata = np.concatenate((sdata, data), axis=1)
            slabel = np.concatenate((slabel, label), axis=0)
        '''Train Network & Save Model'''
        model = net.network_3a()
        model.fit([sdata[0], sdata[1], sdata[2], sdata[3], sdata[4], sdata[5], sdata[6], sdata[7],
                   sdata[8], sdata[9], sdata[10], sdata[11], sdata[12], sdata[13], sdata[14], sdata[15]],
                  slabel, batch_size=80, epochs=80, verbose=1, callbacks=[callback], validation_split=0,
                  validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
        # model.save('./dcnn_{}.h5'.format(tbatch))
        tdata, tlabel = loadmat_3(batch=tbatch, shuffle=False)
        result = model.predict([tdata[0], tdata[1], tdata[2], tdata[3], tdata[4], tdata[5], tdata[6], tdata[7],
                                tdata[8], tdata[9], tdata[10], tdata[11], tdata[12], tdata[13], tdata[14], tdata[15]])
        acc = acc_calc(tlabel, result)
        accrecord[0, tbatch - 1] = acc[6]
        print("\n=================validation=================\n")
        print("T domain: {}".format(tbatch))
        print("Accuracy: {}".format(acc[6]))
    print(result)


def shift3_onlyprevious_arcface():
    """
    trained the classifier with data from only the previous month and tested it on the current month.
    :return:
    """
    callback = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.003, patience=30, verbose=1, mode='auto')
    accrecord = np.ndarray((1, 10))
    for tbatch in range(2, 11):
        sdata = np.ndarray((16, 0, 3, 6, 1))
        slabel = np.ndarray((0, 6))

        for sbatch in range(1, tbatch):  # 读取当前tbatch之前的数据作为sbatch
            data, label = loadmat_3(batch=sbatch, shuffle=True)
            sdata = np.concatenate((sdata, data), axis=1)
            slabel = np.concatenate((slabel, label), axis=0)
        '''Train Network & Save Model'''
        model = net.network_3a_arcface()
        model.fit([sdata[0], sdata[1], sdata[2], sdata[3], sdata[4], sdata[5], sdata[6], sdata[7],
                   sdata[8], sdata[9], sdata[10], sdata[11], sdata[12], sdata[13], sdata[14], sdata[15], slabel],
                  slabel, batch_size=80, epochs=80, verbose=1, callbacks=[callback], validation_split=0,
                  validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
        # model.save('./dcnn_{}.h5'.format(tbatch))
        tdata, tlabel = loadmat_3(batch=tbatch, shuffle=False)
        result = model.predict([tdata[0], tdata[1], tdata[2], tdata[3], tdata[4], tdata[5], tdata[6], tdata[7],
                                tdata[8], tdata[9], tdata[10], tdata[11], tdata[12], tdata[13], tdata[14], tdata[15], tlabel])
        acc = acc_calc(tlabel, result)
        accrecord[0, tbatch - 1] = acc[6]
        print("\n=================validation=================\n")
        print("T domain: {}".format(tbatch))
        print("Accuracy: {}".format(acc[6]))
    print(result)


if __name__ == '__main__':
    # shift2_onlyprevious_fit()
    # shift2_predict()

    shift3_onlyprevious_arcface()
