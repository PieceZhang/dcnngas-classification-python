from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import models as km
from tensorflow.python.keras import regularizers
from utils_arcface import ArcFace, ArcFace2, CosFace, ArcFaceTrainable


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
        bone = kl.Conv2D(filters=2, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='elu', name=scope+'/Conv1')(bone)
        bone = kl.Conv2D(filters=4, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='elu', name=scope+'/Conv2')(bone)
        bone = kl.Conv2D(filters=8, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='elu', name=scope+'/Conv3')(bone)
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
    outputs = ArcFaceTrainable(n_classes=6)(tree)
    model = km.Model(inputs=[branch['branch1'][0], branch['branch2'][0], branch['branch3'][0],branch['branch4'][0],
                             branch['branch5'][0], branch['branch6'][0],branch['branch7'][0], branch['branch8'][0],
                             branch['branch9'][0],branch['branch10'][0], branch['branch11'][0], branch['branch12'][0],
                             branch['branch13'][0], branch['branch14'][0], branch['branch15'][0],branch['branch16'][0]],
                     outputs=outputs)
    if summary:
        model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def network_3b(summary=False):
    """
    DIY network, use loadmat_3
    :return:
    """
    def branch_create(scope):
        assert type(scope) == str, 'TypeError: scope should be a str'
        inputs = kl.Input(shape=(3, 6, 1), name=scope+'/Input')
        inputs = kl.BatchNormalization(1, name=scope + '/BN0')(inputs)
        conv1 = kl.Conv2D(filters=2, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='elu', name=scope+'/Conv1')(inputs)
        conv2 = kl.Conv2D(filters=4, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='elu', name=scope+'/Conv2')(conv1)
        # conv3 = kl.Conv2D(filters=8, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='elu', name=scope+'/Conv3')(conv2)
        concat = kl.Concatenate(axis=1)(conv1, conv2)
        return inputs, concat

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

