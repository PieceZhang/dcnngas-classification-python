# DCNNGasClassification-Python

### DCNN分类电子鼻数据集，并尝试使用SDA进行漂移补偿（效果不好）
#### Ref:  
1. *An optimized Deep Convolutional Neural Network for dendrobium classification based on electronic nose* 

2. *Gas Classification Using Deep Convolutional Neural Networks* 

3. *Domain Adaptation for Large-Scale Sentiment Classification: A Deep Learning Approach*

#### Datasets:
1. 10 boards, Ref: *Chmical gas sensor drift compensation using calssifier ensembles*
2. 5 boards, Ref: *Calibration transfer and drift counteraction in chemical sensor arrays using Direct Standardization*

#### 错误（已修改）:
在network2中，BN层之前的conv层不应使用激活函数。ReLU激活函数应放置在BN层之后。