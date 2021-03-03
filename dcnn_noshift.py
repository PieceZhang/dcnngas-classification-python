from tensorflow.python.keras import models as km
from utils_data import loadmat_1, acc_calc, loadmat_2
import utils_network as net
import numpy as np
import time

if __name__ == '__main__':
    # data_train, label_train, data_test, label_test = loadmat_2('D:/A_/Enose_datasets/5board/board_lite.mat',
    #                                                            shuffle=True, split=0.9)
    data_train, label_train, data_test, label_test = loadmat_1('D:/A_/Enose_datasets/10board/Batch.mat',
                                                               shuffle=True, split=0.9, batch=10)

    '''Train Network & Save Model'''
    model = net.network_1a()
    history = model.fit(data_train, label_train, batch_size=80, epochs=30, verbose=1,
                        callbacks=None, validation_split=0.0, validation_data=[data_test, label_test], shuffle=True,
                        class_weight=None, sample_weight=None, initial_epoch=0)
    model.save('./dcnn_m.h5')

    '''Load model from file & Predict'''
    # model = km.load_model('./h5bkp/dcnn_2.h5')
    # model.summary()
    t0 = time.clock()
    result = model.predict(data_test)
    t1 = time.clock() - t0
    acc = acc_calc(label_test, result)
    print("\n=================validation=================\n")
    print("总精度:{}\n气体1上的精度：{}\n气体2上的精度：{}\n气体3上的精度：{}".format(acc[6],acc[0],acc[1],acc[2]))
    print("气体4上的精度：{}\n气体5上的精度：{}\n气体6上的精度：{}".format(acc[3],acc[4],acc[5]))
    print("{}个样本共耗时：{}".format(label_test.shape[0], t1))
