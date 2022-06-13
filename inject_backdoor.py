import argparse
import os
import pickle
import random
import sys

import gen_utils
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler

import numpy as np
from inject_utils import craft_trapdoors, inject_data, CallbackGenerator, eval_clean_and_attack
from task_utils import Task
import json
from keras.preprocessing.image import ImageDataGenerator

MODEL_PREFIX = "models/"
DIRECTORY = 'results/'

ANALYSIS = True
CHECK_RUN = False
BATCH_SIZE = 128


def schedule(epoch_idx):
    lr_schedule = [15, 25, 35]  # epoch_step
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.1
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.02  # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.004
    return 0.0008


def main():
    random.seed(args["seed"])
    np.random.seed(args["seed"])

    gen_utils.init_gpu(args["gpu"])
    print("Successfully Init GPU")

    print("Start preparing dataset... ")

    task = Task(args["dataset"], load_clean=args["pretrain"])

    target_ls = [int(args['target'])]
    INJECT_RATIO = args["inject_ratio"]
    print("Injection Ratio: ", INJECT_RATIO)
    f_name = "{}_{}".format(args["dataset"], args["config"])

    os.makedirs(DIRECTORY, exist_ok=True)
    file_prefix = os.path.join(DIRECTORY, f_name)

    pattern_dict = craft_trapdoors(target_ls, task.img_shape, pattern_size=args["pattern_size"], mask_ratio=1.0)

    RES = {}
    RES['target_ls'] = target_ls
    RES['dataset'] = args['dataset']
    RES['inject_ratio'] = args['inject_ratio']
    RES['pattern_dict'] = pattern_dict

    print(args["dataset"])

    X_train, Y_train, X_test, Y_test = task.X_train, task.Y_train, task.X_test, task.Y_test

    injected_X, injected_Y, injected = inject_data(X_train, Y_train, target_ls, args["inject_ratio"], pattern_dict,
                                                   task.num_classes)
    injected_X_test, injected_Y_test, _ = inject_data(X_test, Y_test, target_ls, 1.0, pattern_dict,
                                                      task.num_classes)
    print("Finished preparing dataset. ")

    RES['injected_X'] = injected_X
    RES['injected_Y'] = injected_Y
    RES['injected_X_test'] = injected_X_test
    RES['injected_Y_test'] = injected_Y_test
    RES['X_test'] = X_test
    RES['Y_test'] = Y_test
    RES['injected'] = injected

    model_file = MODEL_PREFIX + f_name + "_model.h5"

    number_images = len(injected_X)
    eval_callback = CallbackGenerator(X_test, Y_test, injected_X_test, injected_Y_test)

    if args["pretrain"]:
        callbacks = [eval_callback]
        start_lr = 0.004
    else:
        callbacks = [eval_callback, LearningRateScheduler(schedule=schedule)]
        start_lr = 0.1

    new_model = task.model

    sgd = SGD(lr=start_lr, momentum=0.9, nesterov=True)

    new_model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])

    if args["pretrain"]:
        clean_acc, attack_acc = eval_clean_and_attack(new_model, X_test, Y_test, injected_X_test, injected_Y_test)
        print("Pretrain model acc_clean={:.4f} | acc_backdoor={:.4f}".format(clean_acc, attack_acc))

    train_datagen = ImageDataGenerator(
        horizontal_flip=True
    )

    final_gen = train_datagen.flow(injected_X, injected_Y, batch_size=BATCH_SIZE)

    num_train = number_images // BATCH_SIZE

    try:
        new_model.fit(final_gen, callbacks=callbacks, epochs=args['epochs'],
                      steps_per_epoch=num_train, verbose=1)
    except KeyboardInterrupt:
        print("Early stopping, saving current model...")
        pass

    new_model.save(model_file)

    clean_acc, attack_acc = eval_clean_and_attack(new_model, X_test, Y_test, injected_X_test, injected_Y_test)
    RES['clean_acc'] = clean_acc
    RES['attack_acc'] = attack_acc

    os.makedirs(MODEL_PREFIX, exist_ok=True)
    os.makedirs(DIRECTORY, exist_ok=True)

    file_save_path = file_prefix + "_res.p"
    pickle.dump(RES, open(file_save_path, 'wb'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='config id', default='')
    parser.add_argument('--gpu', '-g', type=str, help='GPU id', default='0')

    args = parser.parse_args(argv)
    config_json = json.load(
        open(os.path.join("configs/", args.config + ".json"), "r"))
    config_json['gpu'] = args.gpu
    config_json['config'] = args.config

    return config_json


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
