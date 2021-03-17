from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import models as km
from tensorflow.python.keras import regularizers
from utils_arcface import ArcFace, CosFace


# from tensorflow.python.keras.backend import l2_normalize


def network_1a():
    """
    reference: An optimized Deep Convolutional Neural Network for dendrobium classification based on electronic nose
    在原文基础上做了修改，加入BN层
    acc=0.992~0.995, size=6.17Mb, time=0.95s
    Total params: 534,214 (全连接层参数525312)
    :return: model
    """
    inputs = kl.Input(shape=(8, 16, 1))
    bone = kl.BatchNormalization(1)(inputs)  # modified: 加入BN层，极大加快收敛速度并提高准确率  TODO 改为layernorm可能效果更好
    bone = kl.Conv2D(filters=32, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.Flatten()(bone)
    bone = kl.Dense(units=1024, activation='relu')(bone)
    bone = kl.Dropout(0.7)(bone)
    outputs = kl.Dense(units=6, activation='softmax')(bone)
    model = km.Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def network_1a_arcface():
    """
    reference: An optimized Deep Convolutional Neural Network for dendrobium classification based on electronic nose
    add arcface
    :return: model
    """
    inputs = kl.Input(shape=(8, 16, 1))
    label = kl.Input(shape=(6,))
    bone = kl.BatchNormalization(1)(inputs)  # modified: 加入BN层，极大加快收敛速度并提高准确率  TODO 改为layernorm可能效果更好
    bone = kl.Conv2D(filters=32, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.Flatten()(bone)
    bone = kl.Dense(units=1024, activation='relu', kernel_initializer='he_normal')(bone)
    outputs = ArcFace(n_classes=6, s=1, m=0.35)([bone, label])
    model = km.Model(inputs=[inputs, label], outputs=outputs)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def network_1a_5boards():
    """
    reference: An optimized Deep Convolutional Neural Network for dendrobium classification based on electronic nose
    改用5boards数据集，使用FC512
    :return: model
    """
    inputs = kl.Input(shape=(300, 8, 1))
    bone = kl.BatchNormalization(1)(inputs)  # TODO 因为5boards数据相差不多，改为L2N层可能效果更好
    bone = kl.Conv2D(filters=32, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.Flatten()(bone)
    bone = kl.Dense(units=512, activation='relu')(bone)
    outputs = kl.Dense(units=4, activation='softmax')(bone)
    model = km.Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def network_1b():
    """
    将原文网络最后的FC层改为512个神经元
    acc=0, size=, time=0
    :return: model
    """
    inputs = kl.Input(shape=(8, 16, 1))
    bone = kl.BatchNormalization(1)(inputs)  # modified: 加入BN层，极大加快收敛速度并提高准确率
    bone = kl.Conv2D(filters=32, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.Flatten()(bone)
    bone = kl.Dense(units=512, activation='relu')(bone)
    outputs = kl.Dense(units=6, activation='softmax')(bone)
    model = km.Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def network_1c():
    """
    BN+FC(512)+FC(6)
    (实验证明FC512效果优于FC1024)
    acc=0.96~0.99, size=841kb, time=0.03s
    :return: model
    """
    inputs = kl.Input(shape=(8, 16, 1))
    bone = kl.BatchNormalization(1)(inputs)

    bone = kl.Flatten()(bone)
    bone = kl.Dense(units=512, activation='relu')(bone)
    outputs = kl.Dense(units=6, activation='softmax')(bone)
    model = km.Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def network_1d():
    """
    1层卷积层：BN+Conv(8)+Maxpooling+FC(512)+FC(6)
    acc=0.96~0.99, size=3Mb, time=0.05s
    :return: model
    """
    inputs = kl.Input(shape=(8, 16, 1))
    bone = kl.BatchNormalization(1)(inputs)
    bone = kl.Conv2D(filters=8, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(bone)
    bone = kl.Flatten()(bone)
    bone = kl.Dense(units=512, activation='relu')(bone)
    outputs = kl.Dense(units=6, activation='softmax')(bone)
    model = km.Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def network_1e():
    """
    2层卷积层：BN+Conv(8)+Conv(8)+Maxpooling+FC(512)+FC(6)
    acc=0.96~0.99, size=3Mb, time=0.05
    :return: model
    """
    inputs = kl.Input(shape=(8, 16, 1))
    bone = kl.BatchNormalization(1)(inputs)
    bone = kl.Conv2D(filters=8, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)  # 卷积核个数太少会影响效果
    bone = kl.Conv2D(filters=8, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(bone)
    bone = kl.Flatten()(bone)
    bone = kl.Dense(units=512, activation='relu')(bone)
    outputs = kl.Dense(units=6, activation='softmax')(bone)
    model = km.Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def network_2(summary=True):
    """
    reference: Gas Classification Using Deep Convolutional Neural Networks
    Total params: 685,382
    acc=0.995~0.998, size=5.38Mb, time=1.2s
    :return: model
    """
    # input
    inputs = kl.Input(shape=(8, 16, 1))
    # block 1
    block1 = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(inputs)
    block1 = kl.BatchNormalization(1)(block1)
    block1 = kl.Activation('relu')(block1)
    block1 = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(block1)
    block1 = kl.BatchNormalization(1)(block1)
    block1 = kl.Activation('relu')(block1)
    # block 2
    block2 = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(block1)
    block2 = kl.BatchNormalization(1)(block2)
    block2 = kl.Activation('relu')(block2)
    block2 = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(block2)
    block2 = kl.BatchNormalization(1)(block2)
    block2 = kl.Activation('relu')(block2)
    block2 = kl.Add()([block1, block2])  # shortcut
    # maxpooling 1
    maxp1 = kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(block2)
    # block 3
    block3 = kl.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(maxp1)
    block3 = kl.BatchNormalization(1)(block3)
    block3 = kl.Activation('relu')(block3)
    block3 = kl.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(block3)
    block3 = kl.BatchNormalization(1)(block3)
    block3 = kl.Activation('relu')(block3)
    block2 = kl.Conv2D(filters=64, kernel_size=(1, 1), padding='valid', strides=(2, 2))(
        block2)  # match dimension: 64, 1*1, 2
    block3 = kl.Add()([block2, block3])  # shortcut
    # block 4
    block4 = kl.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(block3)
    block4 = kl.BatchNormalization(1)(block4)
    block4 = kl.Activation('relu')(block4)
    block4 = kl.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(block4)
    block4 = kl.BatchNormalization(1)(block4)
    block4 = kl.Activation('relu')(block4)
    block4 = kl.Add()([block3, block4])  # shortcut
    # maxpooling 2
    maxp2 = kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(block4)
    # block 5
    block5 = kl.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(maxp2)
    block5 = kl.BatchNormalization(1)(block5)
    block5 = kl.Activation('relu')(block5)
    block5 = kl.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(block5)
    block5 = kl.BatchNormalization(1)(block5)
    block5 = kl.Activation('relu')(block5)
    block4 = kl.Conv2D(filters=128, kernel_size=(1, 1), padding='valid', activation=None, strides=(2, 2))(
        block4)  # match dimension: 128, 1*1, 2
    block5 = kl.Add()([block4, block5])  # shortcut
    # block 6
    block6 = kl.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(block5)
    block6 = kl.BatchNormalization(1)(block6)
    block6 = kl.Activation('relu')(block6)
    block6 = kl.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(block6)
    block6 = kl.BatchNormalization(1)(block6)
    block6 = kl.Activation('relu')(block6)
    block6 = kl.Add()([block5, block6])  # shortcut
    # Global Average Pooling(GAP)
    GAP = kl.GlobalAveragePooling2D(data_format='channels_last')(block6)
    # output
    outputs = kl.Dense(units=6, activation='softmax')(GAP)
    model = km.Model(inputs=inputs, outputs=outputs)
    if summary:
        model.summary()
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def network_2_1dconv(summary=True):
    """
    reference: Gas Classification Using Deep Convolutional Neural Networks
    use 1D conv kernel, and use FC in the end
    :return: model
    """
    # input
    inputs = kl.Input(shape=(8, 16, 1))
    # block 1
    block1 = kl.Conv2D(filters=32, kernel_size=(2, 1), padding='same', strides=(1, 1))(inputs)
    block1 = kl.BatchNormalization(1)(block1)
    block1 = kl.Activation('relu')(block1)
    block1 = kl.Conv2D(filters=32, kernel_size=(2, 1), padding='same', strides=(1, 1))(block1)
    block1 = kl.BatchNormalization(1)(block1)
    block1 = kl.Activation('relu')(block1)
    # block 2
    block2 = kl.Conv2D(filters=32, kernel_size=(2, 1), padding='same', strides=(1, 1))(block1)
    block2 = kl.BatchNormalization(1)(block2)
    block2 = kl.Activation('relu')(block2)
    block2 = kl.Conv2D(filters=32, kernel_size=(2, 1), padding='same', strides=(1, 1))(block2)
    block2 = kl.BatchNormalization(1)(block2)
    block2 = kl.Activation('relu')(block2)
    block2 = kl.Add()([block1, block2])  # shortcut
    # maxpooling 1
    maxp1 = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid')(block2)
    # block 3
    block3 = kl.Conv2D(filters=64, kernel_size=(2, 1), padding='same', strides=(1, 1))(maxp1)
    block3 = kl.BatchNormalization(1)(block3)
    block3 = kl.Activation('relu')(block3)
    block3 = kl.Conv2D(filters=64, kernel_size=(2, 1), padding='same', strides=(1, 1))(block3)
    block3 = kl.BatchNormalization(1)(block3)
    block3 = kl.Activation('relu')(block3)
    block2 = kl.Conv2D(filters=64, kernel_size=(1, 1), padding='valid', strides=(2, 1))(
        block2)  # match dimension: 64, 1*1, 2
    block3 = kl.Add()([block2, block3])  # shortcut
    # block 4
    block4 = kl.Conv2D(filters=64, kernel_size=(2, 1), padding='same', strides=(1, 1))(block3)
    block4 = kl.BatchNormalization(1)(block4)
    block4 = kl.Activation('relu')(block4)
    block4 = kl.Conv2D(filters=64, kernel_size=(2, 1), padding='same', strides=(1, 1))(block4)
    block4 = kl.BatchNormalization(1)(block4)
    block4 = kl.Activation('relu')(block4)
    block4 = kl.Add()([block3, block4])  # shortcut
    # maxpooling 2
    maxp2 = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid')(block4)
    # block 5
    block5 = kl.Conv2D(filters=128, kernel_size=(2, 1), padding='same', strides=(1, 1))(maxp2)
    block5 = kl.BatchNormalization(1)(block5)
    block5 = kl.Activation('relu')(block5)
    block5 = kl.Conv2D(filters=128, kernel_size=(2, 1), padding='same', strides=(1, 1))(block5)
    block5 = kl.BatchNormalization(1)(block5)
    block5 = kl.Activation('relu')(block5)
    block4 = kl.Conv2D(filters=128, kernel_size=(1, 1), padding='valid', activation=None, strides=(2, 1))(
        block4)  # match dimension: 128, 1*1, 2
    block5 = kl.Add()([block4, block5])  # shortcut
    # block 6
    block6 = kl.Conv2D(filters=128, kernel_size=(2, 1), padding='same', strides=(1, 1))(block5)
    block6 = kl.BatchNormalization(1)(block6)
    block6 = kl.Activation('relu')(block6)
    block6 = kl.Conv2D(filters=128, kernel_size=(2, 1), padding='same', strides=(1, 1))(block6)
    block6 = kl.BatchNormalization(1)(block6)
    block6 = kl.Activation('relu')(block6)
    block6 = kl.Add()([block5, block6])  # shortcut
    # Global Average Pooling(GAP)
    # GAP = kl.GlobalAveragePooling2D(data_format='channels_last')(block6)

    GAP = kl.Flatten()(block6)
    GAP = kl.Dense(units=512, activation='relu')(GAP)
    # output
    outputs = kl.Dense(units=6, activation='softmax')(GAP)
    model = km.Model(inputs=inputs, outputs=outputs)
    if summary:
        model.summary()
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def network_2_5boards(summary=True):
    """
    reference: Gas Classification Using Deep Convolutional Neural Networks
    :return: model
    """
    # input
    inputs = kl.Input(shape=(300, 8, 1))
    # block 1
    block1 = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(inputs)
    block1 = kl.BatchNormalization(1)(block1)
    block1 = kl.Activation('relu')(block1)
    block1 = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(block1)
    block1 = kl.BatchNormalization(1)(block1)
    block1 = kl.Activation('relu')(block1)
    # block 2
    block2 = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(block1)
    block2 = kl.BatchNormalization(1)(block2)
    block2 = kl.Activation('relu')(block2)
    block2 = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(block2)
    block2 = kl.BatchNormalization(1)(block2)
    block2 = kl.Activation('relu')(block2)
    block2 = kl.Add()([block1, block2])  # shortcut
    # maxpooling 1
    maxp1 = kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(block2)
    # block 3
    block3 = kl.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(maxp1)
    block3 = kl.BatchNormalization(1)(block3)
    block3 = kl.Activation('relu')(block3)
    block3 = kl.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(block3)
    block3 = kl.BatchNormalization(1)(block3)
    block3 = kl.Activation('relu')(block3)
    block2 = kl.Conv2D(filters=64, kernel_size=(1, 1), padding='valid', strides=(2, 2))(
        block2)  # match dimension: 64, 1*1, 2
    block3 = kl.Add()([block2, block3])  # shortcut
    # block 4
    block4 = kl.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(block3)
    block4 = kl.BatchNormalization(1)(block4)
    block4 = kl.Activation('relu')(block4)
    block4 = kl.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(block4)
    block4 = kl.BatchNormalization(1)(block4)
    block4 = kl.Activation('relu')(block4)
    block4 = kl.Add()([block3, block4])  # shortcut
    # maxpooling 2
    maxp2 = kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(block4)
    # block 5
    block5 = kl.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(maxp2)
    block5 = kl.BatchNormalization(1)(block5)
    block5 = kl.Activation('relu')(block5)
    block5 = kl.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(block5)
    block5 = kl.BatchNormalization(1)(block5)
    block5 = kl.Activation('relu')(block5)
    block4 = kl.Conv2D(filters=128, kernel_size=(1, 1), padding='valid', activation=None, strides=(2, 2))(
        block4)  # match dimension: 128, 1*1, 2
    block5 = kl.Add()([block4, block5])  # shortcut
    # block 6
    block6 = kl.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(block5)
    block6 = kl.BatchNormalization(1)(block6)
    block6 = kl.Activation('relu')(block6)
    block6 = kl.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(block6)
    block6 = kl.BatchNormalization(1)(block6)
    block6 = kl.Activation('relu')(block6)
    block6 = kl.Add()([block5, block6])  # shortcut
    # Global Average Pooling(GAP)
    GAP = kl.GlobalAveragePooling2D(data_format='channels_last')(block6)
    # GAP = kl.Flatten()(GAP)
    # GAP = kl.Dense(units=64, activation='relu')(GAP)
    # output
    outputs = kl.Dense(units=4, activation='softmax')(GAP)

    model = km.Model(inputs=inputs, outputs=outputs)
    if summary:
        model.summary()
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# add SDA:
def SDA_1(summary=True):
    """
    reference: Domain Adaptation for Large-Scale Sentiment Classification: A Deep Learning Approach
    :return: encoder-decoder model (for training), encoder model
    """
    inputs = kl.Input(shape=(1, 128))
    encode = kl.Dense(64, activation='relu')(inputs)
    encode = kl.Dense(32, activation='relu')(encode)
    encode_output = kl.Dense(16)(encode)  # reduce to 16 dims
    decode = kl.Dense(32, activation='relu')(encode_output)
    decode = kl.Dense(64, activation='relu')(decode)
    decode_output = kl.Dense(128, activation='relu')(decode)
    encoder_decoder = km.Model(inputs=inputs, outputs=decode_output)  # encoder-decoder model (for training)
    encoder = km.Model(inputs=inputs, outputs=encode_output)  # encoder model
    if summary:
        encoder_decoder.summary()
    encoder_decoder.compile(optimizer='adam', loss='mse')
    return encoder_decoder, encoder


def network_1a_SDA(summary=True):
    """
    reference: An optimized Deep Convolutional Neural Network for dendrobium classification based on electronic nose
    connected with SDA1
    :return: model
    """
    inputs = kl.Input(shape=(16, 1, 1))
    # bone = kl.BatchNormalization(1)(inputs)  # modified: 加入BN层，极大加快收敛速度并提高准确率
    bone = kl.Conv2D(filters=32, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(inputs)
    bone = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.Flatten()(bone)
    bone = kl.Dense(units=1024, activation='relu')(bone)
    outputs = kl.Dense(units=6, activation='softmax')(bone)
    model = km.Model(inputs=inputs, outputs=outputs)
    if summary:
        model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# add memristor net
def network_1_m():
    """
    network for memristor cnn simulink model
    Total params: 8,230
    Trainable params: 8,214
    Non-trainable params: 16
    acc: 0.983
    :return: model
    """
    inputs = kl.Input(shape=(8, 16, 1))
    bone = kl.BatchNormalization(1)(inputs)
    bone = kl.Conv2D(filters=8, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(bone)
    bone = kl.Conv2D(filters=4, kernel_size=(2, 1), padding='same',
                     activation='relu', strides=(1, 1))(bone)
    bone = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(bone)
    bone = kl.Flatten()(bone)
    bone = kl.Dense(units=60, activation='relu')(bone)
    outputs = kl.Dense(units=6, activation='softmax')(bone)
    model = km.Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# add network 3
def network_3(summary=False):
    """
    DIY network, use loadmat_3
    tricks1: 将大卷积分解成多个小卷积来减少计算量 3*3+3*3+2*2=5*5
    tricks2：compound scaling 深度逐级增加
    实验：AVGP不如MP
    :return:
    """

    def branch_create(scope):
        assert type(scope) == str, 'TypeError: scope should be a str'
        inputs = kl.Input(shape=(3, 6, 1), name=scope+'/Input')
        # bone = kl.BatchNormalization(1, name=scope + '/BN0')(inputs)
        bone = kl.Conv2D(filters=2, kernel_size=(3, 1), padding='same', strides=(1, 1), name=scope+'/Conv1')(inputs)
        # bone = kl.BatchNormalization(1, name=scope+'/BN1')(bone)
        bone = kl.Activation('relu', name=scope+'/relu1')(bone)
        bone = kl.Conv2D(filters=4, kernel_size=(3, 1), padding='same', strides=(1, 1), name=scope+'/Conv2')(bone)
        # bone = kl.BatchNormalization(1, name=scope+'/BN2')(bone)
        bone = kl.Activation('relu', name=scope+'/relu2')(bone)
        bone = kl.Conv2D(filters=8, kernel_size=(3, 1), padding='same', strides=(1, 1), name=scope+'/Conv3')(bone)
        # bone = kl.BatchNormalization(1, name=scope+'/BN3')(bone)
        bone = kl.Activation('relu', name=scope+'/relu3')(bone)
        bone = kl.MaxPool2D(pool_size=(3, 1), strides=(1, 1), padding='valid', name=scope)(bone)
        return inputs, bone

    branch = {}
    for branch_num in range(1, 17):
        branch_name = 'branch{}'.format(branch_num)
        branch[branch_name] = branch_create(branch_name)
    tree = kl.Concatenate(axis=1, name='tree')([branch['branch1'][1], branch['branch2'][1], branch['branch3'][1],
                                                branch['branch4'][1], branch['branch5'][1], branch['branch6'][1],
                                                branch['branch7'][1], branch['branch8'][1], branch['branch9'][1],
                                                branch['branch10'][1], branch['branch11'][1], branch['branch12'][1],
                                                branch['branch13'][1], branch['branch14'][1], branch['branch15'][1],
                                                branch['branch16'][1]])
    tree = kl.Conv2D(filters=8, kernel_size=(1, 3), padding='same', activation='relu', strides=(1, 1))(tree)
    tree = kl.Conv2D(filters=8, kernel_size=(1, 3), padding='same', activation='relu', strides=(1, 1))(tree)
    tree = kl.Conv2D(filters=8, kernel_size=(1, 2), padding='same', activation='relu', strides=(1, 1))(tree)
    tree = kl.Flatten()(tree)
    tree = kl.Dense(units=200, activation='relu')(tree)
    outputs = kl.Dense(units=6, activation='softmax')(tree)
    model = km.Model(inputs=[branch['branch1'][0], branch['branch2'][0], branch['branch3'][0],branch['branch4'][0],
                             branch['branch5'][0], branch['branch6'][0],branch['branch7'][0], branch['branch8'][0],
                             branch['branch9'][0],branch['branch10'][0], branch['branch11'][0], branch['branch12'][0],
                             branch['branch13'][0], branch['branch14'][0], branch['branch15'][0],branch['branch16'][0]],
                     outputs=outputs)
    if summary:
        model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def network_3a(summary=False):
    """
    DIY network, use loadmat_3
    trick1：不使用maxpooling层(10boards数据集已经经过预提取，无需池化)
    trick2：多层FC（非线性划分）
    trick3：elu(对输入变化或噪声更鲁棒)+relu激活函数组合
    :return:
    """
    def branch_create(scope):
        assert type(scope) == str, 'TypeError: scope should be a str'
        inputs = kl.Input(shape=(3, 6, 1), name=scope+'/Input')
        bone = kl.BatchNormalization(1, name=scope + '/BN0')(inputs)
        bone = kl.Conv2D(filters=2, kernel_size=(3, 3), padding='same', strides=(1, 1), name=scope+'/Conv1')(bone)
        # bone = kl.BatchNormalization(1, name=scope+'/BN1')(bone)
        bone = kl.Activation('elu', name=scope+'/relu1')(bone)
        bone = kl.Conv2D(filters=4, kernel_size=(3, 3), padding='same', strides=(1, 1), name=scope+'/Conv2')(bone)
        # bone = kl.BatchNormalization(1, name=scope+'/BN2')(bone)
        bone = kl.Activation('elu', name=scope+'/relu2')(bone)
        bone = kl.Conv2D(filters=8, kernel_size=(3, 3), padding='same', strides=(1, 1), name=scope+'/Conv3')(bone)
        # bone = kl.BatchNormalization(1, name=scope+'/BN3')(bone)
        bone = kl.Activation('elu', name=scope+'/relu3')(bone)
        # bone = kl.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='valid', name=scope)(bone)
        return inputs, bone

    branch = {}
    for branch_num in range(1, 17):
        branch_name = 'branch{}'.format(branch_num)
        branch[branch_name] = branch_create(branch_name)
    tree = kl.Concatenate(axis=1, name='tree')([branch['branch1'][1], branch['branch2'][1], branch['branch3'][1],
                                                branch['branch4'][1], branch['branch5'][1], branch['branch6'][1],
                                                branch['branch7'][1], branch['branch8'][1], branch['branch9'][1],
                                                branch['branch10'][1], branch['branch11'][1], branch['branch12'][1],
                                                branch['branch13'][1], branch['branch14'][1], branch['branch15'][1],
                                                branch['branch16'][1]])
    tree = kl.Flatten()(tree)
    tree = kl.Dense(units=200, activation='relu')(tree)
    tree = kl.Dense(units=200, activation='relu')(tree)
    outputs = kl.Dense(units=6, activation='softmax')(tree)
    model = km.Model(inputs=[branch['branch1'][0], branch['branch2'][0], branch['branch3'][0],branch['branch4'][0],
                             branch['branch5'][0], branch['branch6'][0],branch['branch7'][0], branch['branch8'][0],
                             branch['branch9'][0],branch['branch10'][0], branch['branch11'][0], branch['branch12'][0],
                             branch['branch13'][0], branch['branch14'][0], branch['branch15'][0],branch['branch16'][0]],
                     outputs=outputs)
    if summary:
        model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def network_3a_arcface(summary=False):
    """
    DIY network, use loadmat_3
    :return:
    """
    def branch_create(scope):
        assert type(scope) == str, 'TypeError: scope should be a str'
        inputs = kl.Input(shape=(3, 6, 1), name=scope+'/Input')
        bone = kl.BatchNormalization(1, name=scope + '/BN0')(inputs)
        bone = kl.Conv2D(filters=2, kernel_size=(3, 3), padding='same', strides=(1, 1), name=scope+'/Conv1')(bone)
        # bone = kl.BatchNormalization(1, name=scope+'/BN1')(bone)
        bone = kl.Activation('elu', name=scope+'/relu1')(bone)
        bone = kl.Conv2D(filters=4, kernel_size=(3, 3), padding='same', strides=(1, 1), name=scope+'/Conv2')(bone)
        # bone = kl.BatchNormalization(1, name=scope+'/BN2')(bone)
        bone = kl.Activation('elu', name=scope+'/relu2')(bone)
        bone = kl.Conv2D(filters=8, kernel_size=(3, 3), padding='same', strides=(1, 1), name=scope+'/Conv3')(bone)
        # bone = kl.BatchNormalization(1, name=scope+'/BN3')(bone)
        bone = kl.Activation('elu', name=scope+'/relu3')(bone)
        # bone = kl.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='valid', name=scope)(bone)
        return inputs, bone

    label = kl.Input(shape=(6,))
    branch = {}
    for branch_num in range(1, 17):
        branch_name = 'branch{}'.format(branch_num)
        branch[branch_name] = branch_create(branch_name)
    tree = kl.Concatenate(axis=1, name='tree')([branch['branch1'][1], branch['branch2'][1], branch['branch3'][1],
                                                branch['branch4'][1], branch['branch5'][1], branch['branch6'][1],
                                                branch['branch7'][1], branch['branch8'][1], branch['branch9'][1],
                                                branch['branch10'][1], branch['branch11'][1], branch['branch12'][1],
                                                branch['branch13'][1], branch['branch14'][1], branch['branch15'][1],
                                                branch['branch16'][1]])
    tree = kl.Flatten()(tree)
    tree = kl.Dense(units=200, activation='relu')(tree)
    tree = kl.Dense(units=200, activation='relu')(tree)
    outputs = ArcFace(n_classes=6, s=5, m=0.05)([tree, label])
    model = km.Model(inputs=[branch['branch1'][0], branch['branch2'][0], branch['branch3'][0],branch['branch4'][0],
                             branch['branch5'][0], branch['branch6'][0],branch['branch7'][0], branch['branch8'][0],
                             branch['branch9'][0],branch['branch10'][0], branch['branch11'][0], branch['branch12'][0],
                             branch['branch13'][0], branch['branch14'][0], branch['branch15'][0],branch['branch16'][0], label],
                     outputs=outputs)
    if summary:
        model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # for debugging
    network_3()
