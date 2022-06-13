import argparse
import os
import random
import shutil
import sys

import keras.models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import gen_utils
from inject_utils import eval_attack
from task_utils import load_attack

BASE_DIR = 'results'


def main():
    cur_results_dir = os.path.join(BASE_DIR, "{}".format(args.config))

    gen_utils.init_gpu(args.gpu)

    dataset, task, injected_X, injected_Y, X_test, Y_test, injected_X_test, injected_Y_test, is_backdoor_ls, number_train, feature_layer_name, num_classes, target_label = load_attack(
        args.config, load_clean=None)
    task.injected_X = injected_X
    task.injected_Y = injected_Y
    task.injected_X_test = injected_X_test
    task.injected_Y_test = injected_Y_test
    task.is_backdoor_ls = is_backdoor_ls
    task.target_label = target_label

    if dataset == 'physical' or dataset == 'imagenet':
        BATCH_SIZE = 30
    else:
        BATCH_SIZE = 256

    model = keras.models.load_model(f"models/{dataset}_{args.config}_model.h5")

    succ_attack_idx = model.predict(injected_X_test)
    succ_attack_idx = np.argmax(succ_attack_idx, axis=1) == np.argmax(injected_Y_test, axis=1)
    injected_X_test = injected_X_test[succ_attack_idx]
    injected_Y_test = injected_Y_test[succ_attack_idx]

    attack_accuracy = eval_attack(model, injected_X_test, injected_Y_test)[1]
    assert attack_accuracy > 0.6
    classification_accuracy = eval_attack(model, X_test, Y_test)[1]
    print("classification_accuracy: {:.2f}".format(classification_accuracy))

    task.model = model
    # model.summary()
    for l in model.layers:
        l.trainable = True

    layer_of_interest = None
    for idx, variable in enumerate(model.trainable_variables[::-1][:10]):
        if "kernel" in variable.name.lower():
            layer_of_interest = variable
            break

    if (layer_of_interest is None) or len(layer_of_interest.shape) != 2:
        raise Exception("Selected Layer is problematic. Please check you have a dense layer at the end of the model. ")
    print("Selected Layer: ", layer_of_interest.name)

    full_size_of_embedding = layer_of_interest.shape[0] * layer_of_interest.shape[1]

    print("Full embedding size: {}".format(full_size_of_embedding))
    size_kept = int(full_size_of_embedding * args.ratio)
    if size_kept > 5000:
        size_kept = 5000

    kept_mask = random.sample(list(range(full_size_of_embedding)), size_kept)
    gradient_list = np.zeros((number_train, size_kept))

    print("Shape: ", gradient_list.shape)
    for i, batch_i in enumerate(range(0, number_train, BATCH_SIZE)):
        print("Batch: ", i)
        inputs = injected_X[batch_i:batch_i + BATCH_SIZE]
        labels = injected_Y[batch_i:batch_i + BATCH_SIZE]

        with tf.GradientTape() as tape:
            ypred = model(inputs, training=False)
            labels = tf.ones(labels.shape) * 1 / num_classes
            loss = tf.keras.losses.categorical_crossentropy(labels, ypred)
        jacobian = tape.jacobian(loss, layer_of_interest)
        jacobian = np.array(jacobian)
        jacobian = np.reshape(jacobian, (len(inputs), -1))
        reduced_jacobian = jacobian[:, kept_mask]
        gradient_list[batch_i:batch_i + BATCH_SIZE] = reduced_jacobian
        # np.sum(np.abs(jacobian), axis=1)
        # import pdb
        # pdb.set_trace()

    gradients = np.array(gradient_list.reshape(number_train, -1))
    embedding = gradients

    print("Embedding Shape: {}, pid: {}".format(embedding.shape, os.getpid()))

    if os.path.exists(cur_results_dir):
        shutil.rmtree(cur_results_dir)

    normalized_embedding = normalize(embedding)
    os.mkdir(cur_results_dir)

    np.save(os.path.join(cur_results_dir, "embedding_norm.p"), normalized_embedding)
    np.save(os.path.join(cur_results_dir, "embedding.p"), embedding)

    if args.pca:
        plot_pca(embedding, is_backdoor_ls, injected_Y, task.target_label, os.path.join(cur_results_dir, "pca.png"))
        plot_pca(normalized_embedding, is_backdoor_ls, injected_Y, task.target_label,
                 os.path.join(cur_results_dir, "pca_norm.png"))


def plot_pca(cur_embs, is_backdoor_ls, injected_Y, target_y, output_file):
    fig = plt.figure()
    pca = PCA(n_components=2).fit(cur_embs)
    pca_results = pca.fit_transform(cur_embs)
    backdoor_pca = np.array([p for i, p in enumerate(pca_results) if is_backdoor_ls[i]])
    clean_pca = np.array([p for i, p in enumerate(pca_results) if not is_backdoor_ls[i]])
    target_clean_pca = np.array(
        [p for i, p in enumerate(pca_results) if injected_Y[i][target_y] == 1 and not is_backdoor_ls[i]])

    plt.scatter(clean_pca[:, 0], clean_pca[:, 1], label="Clean X", marker=".", color="red", alpha=0.05)
    plt.scatter(target_clean_pca[:, 0], target_clean_pca[:, 1], label="Target clean", marker=".", color='g', alpha=0.05)
    plt.scatter(backdoor_pca[:, 0], backdoor_pca[:, 1], label="Backdoor X", marker=".", color='b', alpha=0.05)
    plt.legend()
    plt.savefig(output_file)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=str,
                        help='GPU id', default='0')
    parser.add_argument('--config', '-c', type=str,
                        help='name of dataset', default='cifar1')
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--ratio', '-r', type=float,
                        help='name of dataset', default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
