from tensorflow.python.keras import models as km
from utils_data import loadmat_1, acc_calc
import utils_network as net
import numpy as np
import time

if __name__ == '__main__':
    matdir = 'D:/A_/Enose_datasets/10board/Batch.mat'  # directory of .mat file
    accrecord = np.ndarray((10, 10))
    for sbatch in range(1, 11):
        model = km.load_model('./dcnn_{}.h5'.format(sbatch))
        for tbatch in range(1, 11):
            tdata, tlabel = loadmat_1(matdir, batch=tbatch, shuffle=True, split=0)
            result = model.predict(tdata)
            acc = acc_calc(tlabel, result)
            accrecord[sbatch-1, tbatch-1] = acc[6]
            print("\n=================validation=================\n")
            print("S domain: {} \nT domain: {}".format(sbatch, tbatch))
            print("Accuracy: {}".format(acc[6]))
    print("\nFinished")



"""
    matdir = 'D:/A_/Enose_datasets/10board/Batch.mat'  # directory of .mat file
    for sbatch in range(1, 11):
        tbatch = sbatch + 1 if sbatch != 10 else 1
        sdata, slabel = loadmat_1(matdir, batch=sbatch, shuffle=True, split=1)
        tdata, tlabel = loadmat_1(matdir, batch=tbatch, shuffle=True, split=0)
        '''Train Network & Save Model'''
        # model = net.network_2(summary=False)
        # history = model.fit(sdata, slabel, batch_size=100, epochs=80, verbose=1,
        #                     callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
        #                     class_weight=None, sample_weight=None, initial_epoch=0)
        # model.save('./dcnn_{}.h5'.format(sbatch))
        '''Load model from file & Predict'''
        model = km.load_model('./dcnn_{}.h5'.format(sbatch))
        model.summary()
        t0 = time.clock()
        result = model.predict_on_batch(tdata)
        t1 = time.clock() - t0
        acc = acc_calc(tlabel, result)
        print("\n=================validation=================\n")
        print("S domain: {} \nT domain: {}".format(sbatch, tbatch))
        print("总精度:{}\n气体1上的精度：{}\n气体2上的精度：{}\n气体3上的精度：{}".format(acc[6],acc[0],acc[1],acc[2]))
        print("气体4上的精度：{}\n气体5上的精度：{}\n气体6上的精度：{}".format(acc[3],acc[4],acc[5]))
        print("{}个样本共耗时：{}".format(tlabel.shape[0], t1))
        
S domain: 1 
T domain: 2
总精度:0.4590032154340836
气体1上的精度：0.2364217252396166
气体2上的精度：0.6722689075630253
气体3上的精度：0.9090909090909091
气体4上的精度：0.0
气体5上的精度：0.0
气体6上的精度：0.2727272727272727
1244个样本共耗时：0.4645092

S domain: 2 
T domain: 3
总精度:0.8102143757881463
气体1上的精度：0.9390243902439024
气体2上的精度：1.0
气体3上的精度：0.9771689497716894
气体4上的精度：0.4519774011299435
气体5上的精度：1.0
气体6上的精度：0
1586个样本共耗时：0.6297151000000021

S domain: 3 
T domain: 4
总精度:0.7142857142857143
气体1上的精度：0.918918918918919
气体2上的精度：1.0
气体3上的精度：1.0
气体4上的精度：0.9523809523809523
气体5上的精度：0.2222222222222222
气体6上的精度：0
161个样本共耗时：0.35825160000000267

S domain: 4 
T domain: 5
总精度:0.5685279187817259
气体1上的精度：0.8
气体2上的精度：1.0
气体3上的精度：1.0
气体4上的精度：0.359375
气体5上的精度：0
气体6上的精度：0
197个样本共耗时：0.42336509999999805

S domain: 5 
T domain: 6
总精度:0.30521739130434783
气体1上的精度：1.0
气体2上的精度：0.7140902872777017
气体3上的精度：0.1737649063032368
气体4上的精度：0.02900107411385607
气体5上的精度：1.0
气体6上的精度：0
2300个样本共耗时：0.9891767999999956

S domain: 6 
T domain: 7
总精度:0.8018267367838362
气体1上的精度：0.9671150971599403
气体2上的精度：0.5453005927180355
气体3上的精度：0.8862275449101796
气体4上的精度：0.9607843137254902
气体5上的精度：0.8369272237196765
气体6上的精度：0.9977578475336323
3613个样本共耗时：1.91583940000001

S domain: 7 
T domain: 8
总精度:0.9115646258503401
气体1上的精度：1.0
气体2上的精度：1.0
气体3上的精度：1.0
气体4上的精度：0.9142857142857143
气体5上的精度：0.8606060606060606
气体6上的精度：1.0
294个样本共耗时：0.9737093000000243

S domain: 8 
T domain: 9
总精度:0.4702127659574468
气体1上的精度：0
气体2上的精度：0.3021978021978022
气体3上的精度：1.0
气体4上的精度：1.0
气体5上的精度：0.576271186440678
气体6上的精度：0.5135135135135135
470个样本共耗时：0.8273985000000152

S domain: 9 
T domain: 10
总精度:0.295
气体1上的精度：0.9736842105263158
气体2上的精度：0.2329700272479564
气体3上的精度：0.7631578947368421
气体4上的精度：0.2234891676168757
气体5上的精度：0.43159486016628873
气体6上的精度：0.0
3600个样本共耗时：1.4372544999999946

S domain: 10 
T domain: 1
总精度:0.6179775280898876
气体1上的精度：0.8461538461538461
气体2上的精度：0.9150943396226415
气体3上的精度：0.9868421052631579
气体4上的精度：0
气体5上的精度：0.31390134529147984
气体6上的精度：0.0
445个样本共耗时：0.2761529

"""
