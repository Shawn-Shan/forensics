import datetime
import json
import logging
import os
import pickle

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('tensorflow').disabled = True
import h5py
import numpy as np
import tensorflow as tf
from keras.preprocessing import image


def now():
    return datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")


def show(data):
    if type(data) is np.ndarray:
        if len(data.shape) == 3:
            return image.array_to_img(data)


def dump_dictionary_as_json(dict, outfile):
    j = json.dumps(dict)
    with open(outfile, "wb") as f:
        f.write(j.encode())


def pickle_write(data, outfile):
    return pickle.dump(data, open(outfile, "wb"))


def pickle_read(infile):
    return pickle.load(open(infile, "rb"))


def load_json(file):
    return json.load(open(file))


def init_gpu(gpu_index, force=False):
    assert gpu_index is not None
    assert tf.__version__ > '2'

    if isinstance(gpu_index, list):
        gpu_num = ','.join([str(i) for i in gpu_index])
    else:
        gpu_num = str(gpu_index)

    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] and not force:
        print('GPU already initiated')
        return

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)


def mkdir(path):
    full_path = path.split("/")
    for i in range(2, len(full_path) + 1):
        cur_path = "/".join(full_path[:i])
        if not os.path.exists(cur_path):
            print(cur_path)
            os.mkdir(cur_path)


def load_pattern(file, target_size):
    im = image.load_img(file, target_size=target_size)
    im = image.img_to_array(im)
    return im


def dump_image(x, filename, format="png", scale=False):
    img = image.array_to_img(x, scale=scale)
    img.save(filename, format)
    return


def save_h5py(data, name, outfile):
    hf = h5py.File(outfile, 'w')
    for d, n in zip(data, name):
        hf.create_dataset(n, data=d)
    hf.close()


def load_h5py(name, outfile):
    f = h5py.File(outfile, 'r')
    res = []
    for n in name:
        res.append(np.array((f[n])))
    f.close()
    return res


def write_file(text, file, mode="w+"):
    with open(file, mode) as f:
        f.write(text)


def read_file(file, mode="r"):
    with open(file, mode) as f:
        data = f.read()
    return data
