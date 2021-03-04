from tensorflow.python.keras import models as km
from utils_data import loadmat_1, acc_calc, loadmat_1_SDA
import utils_network as net
import numpy as np
import time

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

if __name__ == '__main__':
    matdir = 'D:/A_/Enose_datasets/10board/Batch.mat'  # directory of .mat file

    # data_train, label_train = loadmat_1_SDA(matdir, batch=[9,10], split=1)
    # data_train = normalization(data_train)
    # SDA_train, SDA = net.SDA_1()
    # SDA_train.fit(data_train, data_train, batch_size=100, epochs=50, verbose=1,
    #               callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
    #               class_weight=None, sample_weight=None, initial_epoch=0)
    # SDA.save('./SDA1.h5')

    data_train, label_train = loadmat_1_SDA(matdir, batch=[9,10], split=1, shuffle=False)
    data_train = normalization(data_train)
    SDA = km.load_model('./SDA1.h5')
    SDA.summary()
    data_train = SDA.predict(data_train)
    data_train = data_train.reshape(data_train.shape[0], 1, 16, 1).swapaxes(1, 2)

    model = net.network_1a_SDA(summary=False)
    history = model.fit(data_train[:469], label_train[:469], batch_size=100, epochs=80, verbose=1,
                        callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
                        class_weight=None, sample_weight=None, initial_epoch=0)

    t0 = time.clock()
    result = model.predict(data_train[470:])
    t1 = time.clock() - t0
    acc = acc_calc(label_train[470:], result)
    print("\n=================validation=================\n")
    print("总精度:{}\n气体1上的精度：{}\n气体2上的精度：{}\n气体3上的精度：{}".format(acc[6],acc[0],acc[1],acc[2]))
    print("气体4上的精度：{}\n气体5上的精度：{}\n气体6上的精度：{}".format(acc[3],acc[4],acc[5]))
