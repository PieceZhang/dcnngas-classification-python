from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import models as km


def network_1a():
    """
    reference: An optimized Deep Convolutional Neural Network for dendrobium classification based on electronic nose
    在原文基础上做了修改，加入BN层
    acc=0.992~0.995, size=6.17Mb, time=0.95s
    Total params: 534,214 (全连接层参数525312)
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
    bone = kl.Dense(units=1024, activation='relu')(bone)
    outputs = kl.Dense(units=6, activation='softmax')(bone)
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
    block1 = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                       activation='relu', strides=(1, 1))(inputs)  # TODO 卷积层需要单独的激活函数吗？
    block1 = kl.BatchNormalization(1)(block1)
    block1 = kl.Activation('relu')(block1)
    block1 = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                       activation='relu', strides=(1, 1))(block1)
    block1 = kl.BatchNormalization(1)(block1)
    block1 = kl.Activation('relu')(block1)
    # block 2
    block2 = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                       activation='relu', strides=(1, 1))(block1)
    block2 = kl.BatchNormalization(1)(block2)
    block2 = kl.Activation('relu')(block2)
    block2 = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                       activation='relu', strides=(1, 1))(block2)
    block2 = kl.BatchNormalization(1)(block2)
    block2 = kl.Activation('relu')(block2)
    block2 = kl.Add()([block1, block2])  # shortcut
    # maxpooling 1
    maxp1 = kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(block2)
    # block 3
    block3 = kl.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                       activation='relu', strides=(1, 1))(maxp1)
    block3 = kl.BatchNormalization(1)(block3)
    block3 = kl.Activation('relu')(block3)
    block3 = kl.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                       activation='relu', strides=(1, 1))(block3)
    block3 = kl.BatchNormalization(1)(block3)
    block3 = kl.Activation('relu')(block3)
    block2 = kl.Conv2D(filters=64, kernel_size=(1, 1), padding='valid',
                       activation=None, strides=(2, 2))(block2)  # match dimension: 64, 1*1, 2
    block3 = kl.Add()([block2, block3])  # shortcut
    # block 4
    block4 = kl.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                       activation='relu', strides=(1, 1))(block3)
    block4 = kl.BatchNormalization(1)(block4)
    block4 = kl.Activation('relu')(block4)
    block4 = kl.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                       activation='relu', strides=(1, 1))(block4)
    block4 = kl.BatchNormalization(1)(block4)
    block4 = kl.Activation('relu')(block4)
    block4 = kl.Add()([block3, block4])  # shortcut
    # maxpooling 2
    maxp2 = kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(block4)
    # block 5
    block5 = kl.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                       activation='relu', strides=(1, 1))(maxp2)
    block5 = kl.BatchNormalization(1)(block5)
    block5 = kl.Activation('relu')(block5)
    block5 = kl.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                       activation='relu', strides=(1, 1))(block5)
    block5 = kl.BatchNormalization(1)(block5)
    block5 = kl.Activation('relu')(block5)
    block4 = kl.Conv2D(filters=128, kernel_size=(1, 1), padding='valid',
                       activation=None, strides=(2, 2))(block4)  # match dimension: 128, 1*1, 2
    block5 = kl.Add()([block4, block5])  # shortcut
    # block 6
    block6 = kl.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                       activation='relu', strides=(1, 1))(block5)
    block6 = kl.BatchNormalization(1)(block6)
    block6 = kl.Activation('relu')(block6)
    block6 = kl.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                       activation='relu', strides=(1, 1))(block6)
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
