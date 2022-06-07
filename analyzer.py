import sys

import keras.models
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans

import random

from collections import Counter
from inject_utils import eval_incident

import numpy as np

from inject_utils import eval_attack
from keras.optimizers import SGD

GLOBAL_CACHE = {}


def get_hash(data):
    data = np.sort(data)
    data = [str(i) for i in data]
    data = "-".join(data)
    return data


class Analyzer(object):
    def __init__(self, embs, task, original_model, num_clusters=2, verbose=0,
                 pass_unlearning=False, unlearning_lr=0.008):
        self.task = task
        self.X_train = task.injected_X
        self.Y_train = task.injected_Y
        self.verbose = verbose
        self.X_test = task.X_test
        self.Y_test = task.Y_test
        self.num_train = len(self.X_train)
        self.is_backdoor_ls = task.is_backdoor_ls
        self.backdoor_index = set(np.where(np.array(self.is_backdoor_ls) == 1)[0])
        self.embs = embs
        self.num_clusters = num_clusters
        self.original_model = original_model
        self.data_loader = CleanDataLoader(self.X_test, self.Y_test)
        self.unlearning_lr = unlearning_lr

        if pass_unlearning:  # debugging parameter to bypass unlearning setup and check clustering results
            self.unlearner = None
        else:
            self.unlearner = Unlearner(self, original_model, data_loader=self.data_loader)
        self.problem_set = []
        self.pass_unlearning = pass_unlearning

    def _cluster(self, cur_embs, mapping, num_clusters=5):
        kmean = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
        cluster_results = kmean.fit_predict(cur_embs)
        res = []
        for i in range(0, num_clusters):
            idx = np.where(cluster_results == i)[0]
            global_idx = [mapping[i] for i in idx]
            res.append(global_idx)
        return res

    def _is_bad(self, cur_res):

        is_bad_ls = []
        for r in cur_res:
            print("*" * 80)

            backdoor_ratio, total_backdoor_ratio = self._backdoor_ratio(r)
            if self.verbose:
                print("Current Backdoor Ratio: {:.4f}, consist {:.4f} of all backdoor".format(backdoor_ratio,
                                                                                              total_backdoor_ratio))

            if not self.pass_unlearning:
                results = self.unlearner.is_bad(r, verbose=self.verbose)

                ground_truth_label = (
                            total_backdoor_ratio > 0.5 or backdoor_ratio > 0.1)  # ground truth label of this cluster

                if results and ground_truth_label:
                    print("correct BAD")
                elif not results and not ground_truth_label:
                    print("correct GOOD")
                else:
                    print("\x1b[31m\"WRONG\"\x1b[0m")

            if results:
                is_bad_ls.append(True)
            else:
                is_bad_ls.append(False)

        return is_bad_ls

    def _backdoor_ratio(self, index):
        backdoor_ratio = len(set(index).intersection(self.backdoor_index)) / len(index)
        total_backdoor_ratio = len(set(index).intersection(self.backdoor_index)) / len(self.backdoor_index)
        return backdoor_ratio, total_backdoor_ratio

    def run_clustering(self):
        final_bad_indices = []
        root_index = list(range(0, len(self.X_train)))
        queue = [[root_index, 0, self._backdoor_ratio(root_index)]]
        while queue:
            cur_indices, level, backdoor_ratio = queue.pop()
            cur_embs = self.embs[cur_indices]
            cur_mapping = dict((k, v) for k, v in enumerate(cur_indices))

            cur_num_clusters = self.num_clusters

            cur_res = self._cluster(cur_embs, cur_mapping, num_clusters=cur_num_clusters)
            bad_clusters = self._is_bad(cur_res)

            if all(bad_clusters):
                num_added = 0
                for idx in cur_res:
                    final_bad_indices += idx
                    num_added += len(idx)
                print("DONE, added: {}".format(num_added))
                continue

            if not any(bad_clusters):
                continue

            for cluster_index, is_bad in zip(cur_res, bad_clusters):
                if not is_bad:
                    continue
                queue.append([cluster_index, level + 1, self._backdoor_ratio(cluster_index)])

        return final_bad_indices

    def report(self, final_bad_indices):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        y_pred_set = set(final_bad_indices)
        y_true_set = set(np.where(np.array(self.is_backdoor_ls) == 1)[0])
        negative_set = set(np.where(np.array(self.is_backdoor_ls) == 0)[0])
        for ytrue in y_true_set:
            if ytrue in y_pred_set:
                tp += 1
            else:
                fn += 1

        for yneg in negative_set:
            if yneg in y_pred_set:
                fp += 1
            else:
                tn += 1

        if (tp + fp) == 0:
            percision = 0
        else:
            percision = tp / (tp + fp)

        recall = tp / (tp + fn)
        return {
            "precision": percision,
            "recall": recall,
        }


class Unlearner(object):
    def __init__(self, analyzer, original_model, data_loader, verbose=0):
        self.analyzer = analyzer
        self.unlearning_lr = analyzer.unlearning_lr

        self.original_weights = original_model.get_weights()
        self.model = original_model
        for l in self.model.layers[:-1]:
            l.trainable = False

        self.injected_X_test = analyzer.task.injected_X_test
        self.injected_Y_test = analyzer.task.injected_Y_test
        self.verbose = verbose

        incident_idx = 100
        self.cur_incident_X = self.injected_X_test[incident_idx:incident_idx + 1]
        self.cur_incident_Y = self.injected_Y_test[incident_idx:incident_idx + 1]
        self.cur_incident_correct_Y = self.analyzer.Y_test[incident_idx:incident_idx + 1]

        self.clean_test_X = self.analyzer.X_test
        self.clean_test_Y = self.analyzer.Y_test

        self.loader = data_loader
        self.augmentation = tf.keras.Sequential([
            keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        ])

    def reset_model(self):
        self.model.set_weights(self.original_weights)
        return self.model

    def load_data(self, index):
        index_hash = get_hash(index)
        if index_hash in GLOBAL_CACHE:
            cur_X, cur_Y = GLOBAL_CACHE[index_hash]
        else:
            cur_X = self.analyzer.X_train[index]
            cur_Y = self.analyzer.Y_train[index]
            GLOBAL_CACHE[index_hash] = cur_X, cur_Y
        return cur_X, cur_Y

    def filter_data(self, unlearn_index):
        set_unlearn_index = set(unlearn_index)
        rest_index = [i for i in range(0, self.analyzer.num_train) if i not in set_unlearn_index]
        random.shuffle(rest_index)
        random.shuffle(unlearn_index)

        backdoor_ratio, total_backdoor_ratio = self.analyzer._backdoor_ratio(unlearn_index)
        print("Unlearn Backdoor Ratio: {:.4f}, consist {:.4f} of all backdoor".format(backdoor_ratio,
                                                                                      total_backdoor_ratio))
        self.ratio_unlearn = backdoor_ratio

        backdoor_ratio, total_backdoor_ratio = self.analyzer._backdoor_ratio(rest_index)
        print(
            "Rest Backdoor Ratio: {:.4f}, consist {:.4f} of all backdoor".format(backdoor_ratio, total_backdoor_ratio))
        self.ratio_rest = backdoor_ratio

        unlearn_train_X, unlearn_train_Y = self.load_data(unlearn_index)
        rest_train_X, rest_train_Y = self.load_data(rest_index)

        self.sub_X, self.sub_Y = self.loader.sample(unlearn_train_Y)
        self.c1 = Counter(np.argmax(unlearn_train_Y, axis=1))
        self.c2 = Counter(np.argmax(self.sub_Y, axis=1))
        if self.verbose:
            print("Unlearning Set Distribution over Labels: ", self.c1)

        self.original_normal_loss, self.original_normal_acc = eval_attack(self.model, self.sub_X, self.sub_Y)
        return unlearn_train_X, unlearn_train_Y, rest_train_X, rest_train_Y

    def eval_new_model(self, verbose=1):
        attack_loss, attack_prob = eval_incident(self.model, self.injected_X_test[0:1],
                                                 self.injected_Y_test[0:1],
                                                 avg=False)

        normal_loss, normal_acc = eval_attack(self.model, self.sub_X, self.sub_Y)

        if verbose:
            print("Attack loss: {:.2f}".format(attack_loss))
            print("Normal loss: {:.2f}".format(normal_loss))
            print("attack prob: {:.2f}".format(attack_prob))
        return attack_loss, attack_prob, normal_loss, normal_acc

    def acc(self, ypred, labels):
        acc = tf.reduce_mean(
            tf.cast(tf.math.argmax(ypred, axis=1) == tf.math.argmax(labels, axis=1), tf.float32)).numpy()
        return acc

    def collect_gradient(self, unlearn_train_X, unlearn_train_Y, rest_train_X, rest_train_Y, unlearn_i, rest_i):
        unlearn_inputs = unlearn_train_X[unlearn_i:unlearn_i + self.batch_size]
        unlearn_labels = unlearn_train_Y[unlearn_i:unlearn_i + self.batch_size]
        rest_inputs = rest_train_X[rest_i:rest_i + self.batch_size]
        rest_labels = rest_train_Y[rest_i:rest_i + self.batch_size]

        unlearn_inputs = self.augmentation(unlearn_inputs)
        rest_inputs = self.augmentation(rest_inputs)

        is_training = True

        with tf.GradientTape(persistent=True) as tape:
            ypred = self.model(unlearn_inputs, training=is_training)
            unlearn_acc = self.acc(ypred, unlearn_labels)

            correct_list = tf.math.argmax(ypred, axis=1) == tf.math.argmax(unlearn_labels, axis=1)
            unlearn_labels = tf.ones(unlearn_labels.shape) * 1 / self.analyzer.task.num_classes

            loss_unlearn = tf.keras.losses.categorical_crossentropy(unlearn_labels, ypred)
            loss_unlearn = loss_unlearn[correct_list]
            loss_unlearn = tf.reduce_sum(loss_unlearn) / len(loss_unlearn)

            ypred = self.model(rest_inputs, training=is_training)
            loss_rest = tf.keras.losses.categorical_crossentropy(rest_labels, ypred)

            rest_acc = self.acc(ypred, rest_labels)
            loss_rest = tf.reduce_sum(loss_rest) / len(loss_rest)

            if unlearn_acc < 0.01:  # Set the unlearning loss to 0 when accuracy on unlearning set is very low => this entire the entire model does not collapse.
                loss = loss_rest * self.batch_size
            else:
                loss = (loss_unlearn + loss_rest) * self.batch_size

        g = tape.gradient(loss, self.model.trainable_variables)

        unlearn_i = unlearn_i + self.batch_size
        if unlearn_i > len(unlearn_train_X):
            unlearn_i = 0

        rest_i = rest_i + self.batch_size
        if rest_i > len(rest_train_X):
            rest_i = 0

        loss_ewc = 0
        return g, unlearn_i, rest_i, np.mean(loss_unlearn), np.mean(loss_rest), np.mean(loss_ewc), np.mean(
            unlearn_acc), np.mean(rest_acc)

    def is_bad(self, unlearn_index, verbose=1):
        self.model = self.reset_model()

        unlearn_train_X, unlearn_train_Y, rest_train_X, rest_train_Y = self.filter_data(unlearn_index)

        self.original_loss, self.original_accuracy = eval_attack(self.model, self.sub_X, self.sub_Y)

        original_attack_loss, _, _, _ = self.eval_new_model(verbose=0)

        original_attack_loss_mean = np.percentile(original_attack_loss, 50)

        assert self.original_accuracy > 0.7
        self.batch_size = 128

        unlearn_i = 0
        rest_i = 0

        update_unlearn, _, _, loss_unlearn, loss_rest, loss_ewc, unlearn_acc, rest_acc = self.collect_gradient(
            unlearn_train_X, unlearn_train_Y, rest_train_X, rest_train_Y, unlearn_i, rest_i)

        optimizer = SGD(lr=self.unlearning_lr, momentum=0.9, nesterov=True, clipvalue=0.05)

        n_steps = len(unlearn_train_X) // self.batch_size * 3
        if n_steps < 40:
            n_steps = 40

        for i in range(0, n_steps):
            update_unlearn, unlearn_i, rest_i, loss_unlearn, loss_rest, loss_ewc, unlearn_acc, rest_acc = self.collect_gradient(
                unlearn_train_X, unlearn_train_Y, rest_train_X, rest_train_Y, unlearn_i, rest_i)
            optimizer.apply_gradients(zip(update_unlearn, self.model.trainable_variables))

            if (i % (n_steps // 10) == 0 and (self.verbose or verbose)) or i + 1 == n_steps:
                attack_loss, attack_prob, normal_loss, normal_acc = self.eval_new_model(verbose=0)
                attack_prob_mean = np.percentile(attack_prob, 50)

                attack_loss_mean = np.percentile(attack_loss, 50)

                if self.verbose or verbose:
                    print("Unlearn ACC: {:.2f} - Rest ACC: {:.2f}".format(unlearn_acc, rest_acc))
                    print("{} Unlearn L: {:.4f} - Rest L: {:.4f}".format(i, loss_unlearn, loss_rest, loss_ewc))
                    print("Attack Confidence: {:.2f} - Normal Acc: {:.2f} Attack Loss: {:.2f}} - Normal Loss: {:.2f}\n".format(
                        attack_prob_mean, normal_acc, attack_loss_mean, normal_loss))

                    if unlearn_acc < 0.02 and rest_acc > 0.97:  # Early stop condition
                        break

        self.model = self.reset_model()
        delta = attack_loss_mean - original_attack_loss_mean
        if delta > 1e-2:  # Account for approximation error
            return True
        else:
            return False


class CleanDataLoader(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.num_classes = Y.shape[1]
        self.dict = {}

    def sample(self, target_Y_array, number_images=256):
        number_images = min([number_images, len(self.Y)])
        r = random.sample(range(len(self.Y)), number_images)

        c_X = self.X[r]
        c_Y = self.Y[r]

        res_X = c_X
        res_Y = c_Y

        return res_X, res_Y
