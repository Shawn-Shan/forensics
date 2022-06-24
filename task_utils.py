import json
import os

import gen_utils
import keras
from tensorflow.keras.utils import to_categorical

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout, BatchNormalization
from keras.models import Model
from keras.models import Sequential
from keras.regularizers import l2


class Task(object):
    def __init__(self, name, load_clean=False, test_only=False):
        self.name = name
        self.load_clean = load_clean

        if name == "cifar":
            self.num_classes = 10
            self.img_shape = (32, 32, 3)
            self.feature_layer_name = 'flatten'
        else:
            raise Exception("Not implement")

        if test_only:
            self.X_test, self.Y_test = load_dataset(name, test_only=test_only)
        else:
            self.X_train, self.Y_train, self.X_test, self.Y_test = load_dataset(name, test_only=test_only)
            self.number_train = len(self.X_train)

        if load_clean is not None:
            self.model = self.get_model(load_clean=self.load_clean)
            self.bottleneck_model = self.build_bottleneck_model()

    def build_bottleneck_model(self):
        self.bottleneck_model = build_bottleneck_model(self.model, self.feature_layer_name)
        return self.bottleneck_model

    def get_model(self, load_clean=False):
        model = get_model(self.name, load_clean=load_clean)
        return model

    def get_student_model(self):
        assert self.load_clean or self.name == 'cifar'
        student_model = get_student_model(self.bottleneck_model, self.num_classes)
        return student_model


def get_model(dataset, load_clean=False):
    assert dataset == 'cifar'
    if load_clean:
        # model = keras.models.load_model("models/cifar_cifar2_model.h5")
        model = keras.models.load_model(
            "/home/shansixioing/forensic/wide_resnets_keras-master/models/cifar_clean.h5")
    else:
        # model = create_model()
        model = get_cifar_model()
    return model


def get_cifar_model(softmax=True):
    layers = [
        Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),  # 0
        Activation('relu'),  # 1
        BatchNormalization(),  # 2
        Conv2D(32, (3, 3), padding='same'),  # 3
        Activation('relu'),  # 4
        BatchNormalization(),  # 5
        MaxPooling2D(pool_size=(2, 2)),  # 6

        Conv2D(64, (3, 3), padding='same'),  # 7
        Activation('relu'),  # 8
        BatchNormalization(),  # 9
        Conv2D(64, (3, 3), padding='same'),  # 10
        Activation('relu'),  # 11
        BatchNormalization(),  # 12
        MaxPooling2D(pool_size=(2, 2)),  # 13

        Conv2D(128, (3, 3), padding='same'),  # 14
        Activation('relu'),  # 15
        BatchNormalization(),  # 16
        Conv2D(128, (3, 3), padding='same'),  # 17
        Activation('relu'),  # 18
        BatchNormalization(),  # 19
        MaxPooling2D(pool_size=(2, 2)),  # 20

        Flatten(),  # 21
        Dropout(0.2),  # 22

        Dense(2048, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),  # 23
        Activation('relu'),  # 24
        BatchNormalization(),  # 25

        Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),  # 27
        Activation('relu'),  # 28
        BatchNormalization(),  # 29
        Dense(10),  # 31
    ]

    model = Sequential()

    for layer in layers:
        model.add(layer)
    if softmax:
        model.add(Activation('softmax'))
    return model


def load_dataset(dataset, test_only=False):
    if dataset == "cifar":
        from keras.datasets import cifar10
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        if not test_only:
            X_train = X_train / 255.0
            Y_train = to_categorical(Y_train, 10)

        X_test = X_test / 255.0
        Y_test = to_categorical(Y_test, 10)
    if test_only:
        return X_test, Y_test
    return X_train, Y_train, X_test, Y_test


def load_attack(config_name, load_clean=False):
    config = json.load(open(os.path.join("configs/", config_name + ".json"), "r"))
    dataset = config['dataset']
    task = Task(dataset, load_clean=load_clean, test_only=True)

    res_file = gen_utils.pickle_read(f"results/{dataset}_{config_name}_res.p")
    target_ls = res_file['target_ls']

    injected_X = res_file['injected_X']
    injected_Y = res_file['injected_Y']
    X_test = res_file['X_test']
    Y_test = res_file['Y_test']

    is_backdoor_ls = res_file['injected']

    if "inject_ratio" not in res_file and "number_poison" in res_file:  # clean label case
        injected_X_test = res_file['target_image']
        injected_Y_test = res_file['target_test_Y']
    else:
        injected_X_test = res_file['injected_X_test']
        injected_Y_test = res_file['injected_Y_test']

    target_label = target_ls[0]
    task.X_test = X_test
    task.Y_test = Y_test

    number_train = len(injected_X)
    return dataset, task, injected_X, injected_Y, X_test, Y_test, injected_X_test, injected_Y_test, is_backdoor_ls, number_train, task.feature_layer_name, task.num_classes, target_label


def get_student_model(bottleneck_model, num_classes):
    for l in bottleneck_model.layers:
        l.trainable = False
    x = bottleneck_model.layers[-1].output
    x = Dense(num_classes, name='logit')(x)
    x = Activation('softmax', name='act')(x)
    model = Model(bottleneck_model.input, x)

    opt = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def build_bottleneck_model(model, layer_name):
    bottleneck_model = keras.models.Model(model.input, model.get_layer(layer_name).output)
    bottleneck_model.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])

    return bottleneck_model
