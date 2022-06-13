import os
import sys

import keras.models

import numpy as np

from inject_utils import eval_attack
from task_utils import load_attack
import gen_utils
import argparse
from analyzer import Analyzer

# np.random.seed(1234)

def main():
    learning_rate = 0.004  # Please set this carefully depends on your training configuration. Use the last learning rate at the end of training, if you have build in adaptive learning rate. If notice the unlearning is highly unstable, reduce the learning rate to a smaller value
    gen_utils.init_gpu(args.gpu)
    config = args.config
    CLUSTER_DIR = 'results'

    dataset, task, injected_X, injected_Y, X_test, Y_test, injected_X_test, injected_Y_test, is_backdoor_ls, number_train, feature_layer_name, num_classes, target_label = load_attack(
        config, load_clean=None)
    print("Successfully Load Dataset and Poisoned Model")
    model = keras.models.load_model(f"models/{dataset}_{config}_model.h5")
    task.model = model

    succ_attack_idx = model.predict(injected_X_test)
    succ_attack_idx = np.argmax(succ_attack_idx, axis=1) == np.argmax(injected_Y_test, axis=1)
    injected_X_test = injected_X_test[succ_attack_idx]
    injected_Y_test = injected_Y_test[succ_attack_idx]

    attack_accuracy = eval_attack(model, injected_X_test, injected_Y_test)[1]
    assert attack_accuracy >= 0.8
    classification_accuracy = eval_attack(model, X_test, Y_test)[1]
    print("model clean classification_accuracy: {:.2f}".format(classification_accuracy))

    task.injected_X = injected_X
    task.injected_Y = injected_Y
    task.injected_X_test = injected_X_test
    task.injected_Y_test = injected_Y_test
    task.is_backdoor_ls = is_backdoor_ls
    task.target_label = target_label

    embedding = np.load(os.path.join(CLUSTER_DIR, "{}/embedding_norm.p.npy".format(config)))

    print("Embedding Shape: {}".format(embedding.shape))

    model = keras.models.load_model(f"models/{dataset}_{config}_model.h5")
    analyzer = Analyzer(embedding, task, model, num_clusters=2, verbose=1,
                        pass_unlearning=False, unlearning_lr=learning_rate)

    final_results = analyzer.run_clustering()
    cur_res = analyzer.report(final_results)
    print("*" * 80)
    print("Final Traceback Results: Precision = {:.4f} | Recall = {:.4f}".format(cur_res['precision'], cur_res["recall"]))
    print("DONE")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='config file name', default='cifar1')
    parser.add_argument('--gpu', '-g', type=str, help='GPU id', default='0')

    args = parser.parse_args(argv)

    return args


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
