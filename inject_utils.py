import random

import keras
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


def sample_from_generator(gen, nb_sample, preprocessing=None):
    cur_x, cur_y = next(gen)
    input_shape = list(cur_x.shape)[1:]
    num_classes = cur_y.shape[1]
    batch_size = len(cur_x)

    X_sample = np.zeros([nb_sample] + list(input_shape))
    Y_sample = np.zeros((nb_sample, num_classes))

    for i in range(0, nb_sample, batch_size):
        cur_x, cur_y = next(gen)
        if len(X_sample[i:i + batch_size]) < len(cur_x):
            cur_x = cur_x[:len(X_sample[i:i + batch_size])]
            cur_y = cur_y[:len(Y_sample[i:i + batch_size])]

        X_sample[i:i + batch_size] = cur_x
        Y_sample[i:i + batch_size] = cur_y

    if preprocessing is not None:
        X_sample = preprocessing(X_sample)
    return X_sample, Y_sample


def eval_attack(model, injected_X_test, injected_Y_test):
    test_pred = model.predict(injected_X_test)
    attack_succ = np.argmax(test_pred, axis=1) == np.argmax(injected_Y_test, axis=1)
    loss = tf.keras.losses.categorical_crossentropy(injected_Y_test, test_pred)
    return np.mean(loss), np.mean(attack_succ)


def eval_incident(model, incident_X, incident_Y, avg=True):
    ypred = model.predict(incident_X)
    attack_prob = ypred[incident_Y == 1]
    loss = tf.keras.losses.categorical_crossentropy(incident_Y, ypred).numpy()
    if avg:
        attack_prob = np.mean(attack_prob)
        loss = np.mean(loss)

    return loss, attack_prob


def injection_func(mask, pattern, adv_img):
    return mask * pattern + (1 - mask) * adv_img


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 50:
        lr *= 0.5e-1
    elif epoch > 40:
        lr *= 1e-1
    elif epoch > 15:
        lr *= 1e-1
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def mask_pattern_func(pattern_dict, y_target):
    mask, pattern = random.choice(pattern_dict[y_target])
    return mask, pattern


def infect_X(img, tgt, pattern_dict, num_classes):
    mask, pattern = mask_pattern_func(pattern_dict, tgt)
    adv_img = injection_func(mask, pattern, img)
    return adv_img, to_categorical(tgt, num_classes=num_classes)


def inject_data(X, y, target_ls, inject_ratio, pattern_dict, num_classes):
    injected_X, injected_Y = [], []
    injected = []
    for cur_x, cur_y in zip(X, y):
        inject_ptr = random.uniform(0, 1)
        if inject_ptr <= inject_ratio:
            tgt = random.choice(target_ls)
            cur_x, cur_y = infect_X(cur_x, tgt, pattern_dict, num_classes)
            injected.append(1)
        else:
            injected.append(0)
        injected_X.append(cur_x)
        injected_Y.append(cur_y)

    injected_X = np.array(injected_X)
    injected_Y = np.array(injected_Y)
    return injected_X, injected_Y, injected


class DataGenerator(object):
    def __init__(self, target_ls, pattern_dict, num_classes, preprocess_input=None):
        self.target_ls = target_ls
        self.pattern_dict = pattern_dict
        self.num_classes = num_classes
        self.preprocess_input = preprocess_input

    def mask_pattern_func(self, y_target):
        mask, pattern = random.choice(self.pattern_dict[y_target])
        return mask, pattern

    def infect_X(self, img, tgt):
        mask, pattern = self.mask_pattern_func(tgt)
        adv_img = img
        adv_img = injection_func(mask, pattern, adv_img)
        return adv_img, keras.utils.to_categorical(tgt, num_classes=self.num_classes)

    def infect_X_batch(self, imgs, tgt):
        mask, pattern = self.mask_pattern_func(tgt)
        adv_imgs = imgs
        adv_imgs = injection_func(mask, pattern, adv_imgs)
        return adv_imgs, keras.utils.to_categorical([tgt] * len(imgs), num_classes=self.num_classes)

    def generate_data(self, gen, inject_ratio):
        while 1:
            batch_X, batch_Y = next(gen)
            batch_size = len(batch_X)
            num_select = int(batch_size * inject_ratio)

            inject_index = random.sample(list(range(batch_size)), num_select)
            batch_X[inject_index], batch_Y[inject_index] = self.infect_X_batch(batch_X[inject_index], self.target_ls[0])

            yield self.preprocess_input(batch_X), batch_Y


def cal_acc(model, x, y):
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis=1)
    y = np.argmax(y, axis=1)
    acc = np.mean(y_pred == y)
    return acc


def eval_clean_and_attack(model, X_test, Y_test, injected_X_test, injected_Y_test):
    _, clean_acc = model.evaluate(X_test, Y_test, verbose=0)
    _, attack_acc = model.evaluate(injected_X_test, injected_Y_test, verbose=0)
    return clean_acc, attack_acc


class CallbackGenerator(keras.callbacks.Callback):
    def __init__(self, X_test, Y_test, injected_X_test, injected_Y_test):
        test_subsample = random.sample(range(len(X_test)), 100)
        attack_subsample = random.sample(range(len(injected_X_test)), 100)
        self.X_test = X_test[test_subsample]
        self.Y_test = Y_test[test_subsample]
        self.injected_X_test = injected_X_test[attack_subsample]
        self.injected_Y_test = injected_Y_test[attack_subsample]

    def on_epoch_end(self, epoch, logs=None):
        clean_acc, attack_acc = eval_clean_and_attack(self.model, self.X_test, self.Y_test, self.injected_X_test,
                                                      self.injected_Y_test)
        print("Epoch: {} - Clean Acc {:.4f} - Backdoor Acc {:.4f}".format(epoch, clean_acc, attack_acc))
        if clean_acc > 0.90 and attack_acc == 1.0:
            self.model.stop_training = True
        return clean_acc, attack_acc


def construct_mask_random_location(image_row=32, image_col=32, channel_num=3, pattern_size=4,
                                   color=[255.0, 255.0, 255.0]):
    c_col = random.choice(range(0, image_col - pattern_size + 1))
    c_row = random.choice(range(0, image_row - pattern_size + 1))

    mask = np.zeros((image_row, image_col, channel_num))
    pattern = np.zeros((image_row, image_col, channel_num))

    mask[c_row:c_row + pattern_size, c_col:c_col + pattern_size, :] = 1
    if channel_num == 1:
        pattern[c_row:c_row + pattern_size, c_col:c_col + pattern_size, :] = [1]
    else:
        pattern[c_row:c_row + pattern_size, c_col:c_col + pattern_size, :] = color

    return mask, pattern


def construct_mask_random_location_mnist(image_row=28, image_col=28, channel_num=1, pattern_size=4,
                                         color=[1.]):
    c_col = random.choice(range(0, image_col - pattern_size + 1))
    c_row = random.choice(range(0, image_row - pattern_size + 1))

    mask = np.zeros((image_row, image_col, channel_num))
    pattern = np.zeros((image_row, image_col, channel_num))

    mask[c_row:c_row + pattern_size, c_col:c_col + pattern_size, :] = 1
    if channel_num == 1:
        pattern[c_row:c_row + pattern_size, c_col:c_col + pattern_size, :] = [1]
    else:
        pattern[c_row:c_row + pattern_size, c_col:c_col + pattern_size, :] = color

    return mask, pattern


def iter_pattern_base_per_mnist(target_ls, image_shape, num_clusters, pattern_per_label=1, pattern_size=3,
                                mask_ratio=0.1):
    total_ls = {}

    for y_target in target_ls:

        cur_pattern_ls = []

        for p in range(pattern_per_label):
            tot_mask = np.zeros(image_shape)
            tot_pattern = np.zeros(image_shape)
            for p in range(num_clusters):
                mask, _ = construct_mask_random_location_mnist(image_row=image_shape[0],
                                                               image_col=image_shape[1],
                                                               channel_num=image_shape[2],
                                                               pattern_size=pattern_size)
                tot_mask += mask

                m1 = random.uniform(0, 1)

                s1 = random.uniform(0, 1)

                r = np.random.normal(m1, s1, image_shape[:-1])
                cur_pattern = np.stack([r], axis=2)
                cur_pattern = cur_pattern * (mask != 0)
                cur_pattern = np.clip(cur_pattern, 0, 1.0)
                tot_pattern += cur_pattern

            tot_mask = (tot_mask > 0) * mask_ratio
            tot_pattern = np.clip(tot_pattern, 0, 1.0)
            cur_pattern_ls.append([tot_mask, tot_pattern])

        total_ls[y_target] = cur_pattern_ls
    return total_ls


def construct_mask_mnist(image_row=28, image_col=28, pattern_size=6, margin=1, channel_num=1, randomize=False,
                         color=[1.]):
    mask = np.zeros((image_row, image_col, channel_num))
    pattern = np.zeros((image_row, image_col, channel_num))

    mask[image_row - margin - pattern_size:image_row - margin, image_col - margin - pattern_size:image_col - margin,
    :] = 1
    rdm = np.random.uniform(0, 1, [pattern_size, pattern_size, 1])
    if randomize:
        pattern[image_row - margin - pattern_size:image_row - margin,
        image_col - margin - pattern_size:image_col - margin, :] = rdm
    else:
        pattern[image_row - margin - pattern_size:image_row - margin,
        image_col - margin - pattern_size:image_col - margin, :] = color

    return mask, pattern


def construct_mask(image_row=32, image_col=32, pattern_size=6, margin=1, channel_num=3, randomize=False,
                   color=[255., 255., 255.]):
    mask = np.zeros((image_row, image_col, channel_num))
    pattern = np.zeros((image_row, image_col, channel_num))

    mask[image_row - margin - pattern_size:image_row - margin, image_col - margin - pattern_size:image_col - margin,
    :] = 1
    rdm = np.random.uniform(0, 255, [pattern_size, pattern_size, 3])
    if randomize:
        pattern[image_row - margin - pattern_size:image_row - margin,
        image_col - margin - pattern_size:image_col - margin, :] = rdm
    else:
        pattern[image_row - margin - pattern_size:image_row - margin,
        image_col - margin - pattern_size:image_col - margin, :] = color

    return mask, pattern


def iter_pattern_base_fixed(target_ls, image_shape, pattern_size=3, mask_ratio=0.1):
    total_ls = {}
    for y_target in target_ls:
        cur_pattern_ls = []
        if image_shape[2] == 1:
            mask, _ = construct_mask_mnist(image_row=image_shape[0],
                                           image_col=image_shape[1],
                                           channel_num=image_shape[2],
                                           pattern_size=pattern_size, margin=3)
            cur_pattern = np.ones(image_shape) * 1.0
        else:
            mask, _ = construct_mask(image_row=image_shape[0],
                                     image_col=image_shape[1],
                                     channel_num=image_shape[2],
                                     pattern_size=pattern_size, margin=3)
            cur_pattern = np.ones(image_shape) * 1.0

        mask = mask * mask_ratio
        cur_pattern = cur_pattern * np.array([1, 1, 0])
        cur_pattern_ls.append([mask, cur_pattern])
        total_ls[y_target] = cur_pattern_ls
    return total_ls


def craft_trapdoors(target_ls, image_shape, pattern_size=4, mask_ratio=1.0):
    return iter_pattern_base_fixed(target_ls, image_shape, pattern_size=pattern_size, mask_ratio=mask_ratio)


def get_other_label_data(X, Y, target):
    X_filter = np.array(X)
    Y_filter = np.array(Y)
    remain_idx = np.argmax(Y, axis=1) != target
    X_filter = X_filter[remain_idx]
    Y_filter = Y_filter[remain_idx]
    return X_filter, Y_filter


def preprocess(X, method):
    assert method in {'raw', 'imagenet', 'inception', 'mnist'}

    if method is 'raw':
        pass
    elif method is 'imagenet':
        X = imagenet_preprocessing(X)
    else:
        raise Exception('unknown method %s' % method)

    return X


def reverse_preprocess(X, method):
    assert method in {'raw', 'imagenet', 'inception', 'mnist'}

    if method is 'raw':
        pass
    elif method is 'imagenet':
        X = imagenet_reverse_preprocessing(X)
    else:
        raise Exception('unknown method %s' % method)

    return X
