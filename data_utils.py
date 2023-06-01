import pickle
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.datasets import cifar10, cifar100, mnist
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import scipy.io as sio
import random
import json


def get_index(wv_file):
    embeddings_index = {}
    word_index = {}
    count = 0
    with open(wv_file, encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
            word_index[word] = count
            count += 1
    return word_index, embeddings_index


def load_YELP_data(file, word_index, max_sequence_length):
    texts = []
    labels = []
    labels_index = {1.0: 0, 5.0: 1}
    with open(file, 'r', encoding='utf-8') as fw:
        for line in fw.readlines():
            r = json.loads(line)
            texts.append(r['text'])
            labels.append(labels_index[r['stars']])

    tokenizer = Tokenizer()
    tokenizer.word_index = word_index
    sequences = tokenizer.texts_to_sequences(texts)

    data = pad_sequences(sequences, maxlen=max_sequence_length)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


def load_MNIST_data(standarized=False, verbose=False):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if standarized:
        X_train = X_train / 255
        X_test = X_test / 255
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image

    if verbose is True:
        print("MNIST dataset ... ")
        print("X_train shape :", X_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_train shape :", y_train.shape)
        print("y_test shape :", y_test.shape)

    return X_train, y_train, X_test, y_test


def load_EMNIST_data(file, verbose=False, standarized=False):
    """
    file should be the downloaded EMNIST file in .mat format.
    """
    mat = sio.loadmat(file)
    data = mat["dataset"]

    writer_ids_train = data['train'][0, 0]['writers'][0, 0]
    writer_ids_train = np.squeeze(writer_ids_train)
    X_train = data['train'][0, 0]['images'][0, 0]
    X_train = X_train.reshape((X_train.shape[0], 28, 28), order="F")
    y_train = data['train'][0, 0]['labels'][0, 0]
    y_train = np.squeeze(y_train)
    y_train -= 1  # y_train is zero-based

    writer_ids_test = data['test'][0, 0]['writers'][0, 0]
    writer_ids_test = np.squeeze(writer_ids_test)
    X_test = data['test'][0, 0]['images'][0, 0]
    X_test = X_test.reshape((X_test.shape[0], 28, 28), order="F")
    y_test = data['test'][0, 0]['labels'][0, 0]
    y_test = np.squeeze(y_test)
    y_test -= 1  # y_test is zero-based

    if standarized:
        X_train = X_train / 255
        X_test = X_test / 255
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image

    if verbose is True:
        print("EMNIST-letter dataset ... ")
        print("X_train shape :", X_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_train shape :", y_train.shape)
        print("y_test shape :", y_test.shape)

    return X_train, y_train, X_test, y_test, writer_ids_train, writer_ids_test


def load_CIFAR_data(data_type="CIFAR10", label_mode="fine",
                    standarized=False, verbose=False):
    if data_type == "CIFAR10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    elif data_type == "CIFAR100":
        (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode=label_mode)
    else:
        print("Unknown Data type. Stopped!")
        return None

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    # substract mean and normalized to [-1/2,1/2]
    if standarized:
        X_train = X_train / 255
        X_test = X_test / 255
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image

    if verbose is True:
        print("X_train shape :", X_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_train shape :", y_train.shape)
        print("y_test shape :", y_test.shape)

    return X_train, y_train, X_test, y_test


def load_CIFAR_from_local(local_dir, data_type="CIFAR10", with_coarse_label=False,
                          standarized=False, verbose=False):
    # dir_name = os.path.abspath(local_dir)
    if data_type == "CIFAR10":
        X_train, y_train = [], []
        for i in range(1, 6, 1):
            file_name = None
            file_name = os.path.join(local_dir + "data_batch_{0}".format(i))
            X_tmp, y_tmp = None, None
            with open(file_name, 'rb') as fo:
                datadict = pickle.load(fo, encoding='bytes')

            X_tmp = datadict[b'data']
            y_tmp = datadict[b'labels']
            X_tmp = X_tmp.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            y_tmp = np.array(y_tmp)

            X_train.append(X_tmp)
            y_train.append(y_tmp)
            del X_tmp, y_tmp
        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)

        file_name = None
        file_name = os.path.join(local_dir + "test_batch")
        with open(file_name, 'rb') as fo:
            datadict = pickle.load(fo, encoding='bytes')

            X_test = datadict[b'data']
            y_test = datadict[b'labels']
            X_test = X_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            y_test = np.array(y_test)

    elif data_type == "CIFAR100":
        file_name = None
        file_name = os.path.abspath(local_dir + "train")
        with open(file_name, 'rb') as fo:
            datadict = pickle.load(fo, encoding='bytes')
            X_train = datadict[b'data']
            if with_coarse_label:
                y_train = datadict[b'coarse_labels']
            else:
                y_train = datadict[b'fine_labels']
            X_train = X_train.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            y_train = np.array(y_train)

        file_name = os.path.join(local_dir + "test")
        with open(file_name, 'rb') as fo:
            datadict = pickle.load(fo, encoding='bytes')
            X_test = datadict[b'data']
            if with_coarse_label:
                y_test = datadict[b'coarse_labels']
            else:
                y_test = datadict[b'fine_labels']
            X_test = X_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            y_test = np.array(y_test)

    else:
        print("Unknown Data type. Stopped!")
        return None

    if standarized:
        X_train = X_train / 255
        X_test = X_test / 255
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image

    if verbose is True:
        print("X_train shape :", X_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_train shape :", y_train.shape)
        print("y_test shape :", y_test.shape)

    return X_train, y_train, X_test, y_test


def generate_partial_data(X, y, class_in_use=None, verbose=False):
    if class_in_use is None:  # private classes
        idx = np.ones_like(y, dtype=bool)
    else:
        idx = [y == i for i in class_in_use]
        idx = np.any(idx, axis=0)
    X_incomplete, y_incomplete = X[idx], y[idx]
    if verbose is True:
        print("X shape :", X_incomplete.shape)
        print("y shape :", y_incomplete.shape)
    return X_incomplete, y_incomplete


def generate_bal_private_data(X, y, N_parties=10, classes_in_use=range(11),
                              N_samples_per_class=20, data_overlap=False, read_saved=False, dataset='yelp'):
    """
    Input: 
    -- N_parties : int, number of collaboraters in this activity;
    -- classes_in_use: array or generator, the classes of EMNIST-letters dataset 
    (0 <= y <= 25) to be used as private data; 
    -- N_sample_per_class: int, the number of private data points of each class for each party
    
    return: 
    
    """
    priv_data = [None] * N_parties
    combined_idx = np.array([], dtype=np.int16)
    print('cls_in_use', classes_in_use)
    print('y', y)
    if read_saved:
        for i in range(N_parties):
            idxs_per_party = np.load('dataset/' + dataset + '/iid/train_party' + str(i) + '.npy')
            print('i =', i, 'len(idxs_per_party) =', idxs_per_party.shape)
            priv_data[i] = {}
            priv_data[i]['idx'] = idxs_per_party
            priv_data[i]["X"] = X[idxs_per_party]
            priv_data[i]["y"] = y[idxs_per_party]
    else:
        for cls in classes_in_use:  # private_classes
            idx = np.where(y == cls)[0]
            print('cls,idx', cls, idx)
            idx = np.random.choice(idx, N_samples_per_class * N_parties,
                                   replace=data_overlap)
            combined_idx = np.r_[combined_idx, idx]
            for i in range(N_parties):
                idx_tmp = idx[i * N_samples_per_class: (i + 1) * N_samples_per_class]
                if priv_data[i] is None:
                    tmp = {}
                    tmp["X"] = X[idx_tmp]
                    tmp["y"] = y[idx_tmp]
                    tmp["idx"] = idx_tmp
                    priv_data[i] = tmp  # {'x':...,'y':...,'idx':...}
                else:
                    priv_data[i]['idx'] = np.r_[priv_data[i]["idx"], idx_tmp]
                    priv_data[i]["X"] = np.vstack([priv_data[i]["X"], X[idx_tmp]])
                    priv_data[i]["y"] = np.r_[priv_data[i]["y"], y[idx_tmp]]
                    if cls == classes_in_use[-1]:
                        np.save('dataset/' + dataset + '/iid/train_party' + str(i), priv_data[i]['idx'])

    total_priv_data = {}  # {'x':...,'y':...,'idx':...}
    total_priv_data["idx"] = combined_idx
    total_priv_data["X"] = X[combined_idx]
    total_priv_data["y"] = y[combined_idx]
    return priv_data, total_priv_data


def generate_alignment_data(X, y, N_alignment=3000):
    split = StratifiedShuffleSplit(n_splits=1, train_size=N_alignment)
    if N_alignment == "all":
        alignment_data = {}
        alignment_data["idx"] = np.arange(y.shape[0])
        alignment_data["X"] = X
        alignment_data["y"] = y
        return alignment_data
    for train_index, _ in split.split(X, y):
        X_alignment = X[train_index]
        y_alignment = y[train_index]
    alignment_data = {}
    alignment_data["idx"] = train_index
    alignment_data["X"] = X_alignment
    alignment_data["y"] = y_alignment

    return alignment_data


def generate_YELP_business_based_data(fpath, N_parties, word_index, max_sequence_length,
                                      read_saved=False):
    priv_data = [None] * N_parties
    priv_data_test = [None] * N_parties
    total_priv_data = {}

    tokenizer = Tokenizer()
    tokenizer.word_index = word_index
    labels_index = {1.0: 0, 5.0: 1}

    i = 0
    save_path_test = r'dataset/yelp/noniid_test/'
    file = os.listdir(fpath)
    for f in file:
        real_path = os.path.join(fpath, f)
        reviews = []
        stars = []
        print(real_path)
        with open(real_path, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                r = json.loads(line)
                reviews.append(r['text'])
                stars.append(labels_index[r['stars']])

        sequences = tokenizer.texts_to_sequences(reviews)
        data = pad_sequences(sequences, maxlen=max_sequence_length)

        if read_saved is False:
            test_idx = random.sample(list(range(len(stars))), int(0.5 * len(stars)))
            train_idx = list(set(range(len(stars))) - set(test_idx))
            np.save(save_path_test + os.path.splitext(f)[0], np.array(test_idx))

        else:
            test_idx = np.load(save_path_test + os.path.splitext(f)[0] + '.npy')
            train_idx = list(set(range(len(stars))) - set(test_idx))

        reviews_train = [data[idx] for idx in train_idx]
        reviews_test = [data[idx] for idx in test_idx]
        stars_train = [stars[idx] for idx in train_idx]
        stars_test = [stars[idx] for idx in test_idx]

        priv_data[i] = {"X": np.array(reviews_train), "y": np.array(stars_train)}
        priv_data_test[i] = {"X": np.array(reviews_test), "y": np.array(stars_test)}
        if i == 0:
            total_priv_data = {"X": np.array(reviews_train), "y": np.array(stars_train)}
        else:
            total_priv_data["X"] = np.r_[total_priv_data["X"], reviews_train]
            total_priv_data["y"] = np.r_[total_priv_data["y"], stars_train]
        i += 1
        if i == N_parties:
            break
    return priv_data, priv_data_test, total_priv_data


def generate_EMNIST_writer_based_data(X, y, writer_info, N_priv_data_min=30,
                                      N_parties=5, classes_in_use=range(6), read_saved=False):
    # mask is a boolean array of the same shape as y
    # mask[i] = True if y[i] in classes_in_use
    mask = None
    mask = [y == i for i in classes_in_use]
    mask = np.any(mask, axis=0)

    df_tmp = None
    df_tmp = pd.DataFrame({"writer_ids": writer_info, "is_in_use": mask})
    # print(df_tmp.head())
    groupped = df_tmp[df_tmp["is_in_use"]].groupby("writer_ids")

    # organize the input the data (X,y) by writer_ids.
    # That is, 
    # data_by_writer is a dictionary where the keys are writer_ids,
    # and the contents are the correcponding data. 
    # Notice that only data with labels in class_in_use are included.
    data_by_writer = {}
    writer_ids = []
    for wt_id, idx in groupped.groups.items():
        if len(idx) >= N_priv_data_min:
            # print('wt_idx',wt_id,idx)
            writer_ids.append(wt_id)
            data_by_writer[wt_id] = {"X": X[idx], "y": y[idx],
                                     "idx": idx, "writer_id": wt_id}
    print(len(writer_ids))

    # each participant in the collaborative group is assigned data 
    # from a single writer.
    Writers_per_party = 5
    combined_idx = np.array([], dtype=np.int64)
    private_data = []
    private_data_test = []

    if read_saved is False:
        ids_to_use = np.random.choice(writer_ids, size=N_parties * Writers_per_party, replace=False)
        np.save('dataset/emnist/noniid/wids.npy', ids_to_use)
    else:
        ids_to_use = np.load('dataset/emnist/noniid/wids.npy')

    for i in range(N_parties):
        test_idxs = []
        train_idxs = []

        if read_saved is False:
            for j in range(Writers_per_party):
                id_tmp = ids_to_use[i * Writers_per_party + j]
                test_idx = random.sample(list(data_by_writer[id_tmp]["idx"]),
                                         int(0.6 * len(data_by_writer[id_tmp]["idx"])))
                train_idx = list(set(data_by_writer[id_tmp]["idx"]) - set(test_idx))

                test_idxs.extend(test_idx)
                train_idxs.extend(train_idx)
            np.save('dataset/emnist/noniid/' + 'train_party' + str(i) + '.npy', np.array(train_idxs))
            np.save('dataset/emnist/noniid/' + 'test_party' + str(i) + '.npy', np.array(test_idxs))
        else:
            test_idxs = np.load('dataset/emnist/noniid/' + 'test_party' + str(i) + '.npy')
            train_idxs = np.load('dataset/emnist/noniid/' + 'train_party' + str(i) + '.npy')

        combined_idx = np.r_[combined_idx, train_idxs]

        private_data.append({"X": X[train_idxs], "y": y[train_idxs],
                             "idx": train_idxs})
        private_data_test.append({"X": X[test_idxs], "y": y[test_idxs],
                                  "idx": test_idxs})
        print('party ' + str(i) + ' 的数据集总量为 ' + str(len(train_idxs) + len(test_idxs)) + ',训练集总量为 ' \
              + str(len(train_idxs)) + ',测试集总量为 ' + str(len(test_idxs)))

    total_priv_data = {}
    total_priv_data["idx"] = combined_idx
    total_priv_data["X"] = X[combined_idx]
    total_priv_data["y"] = y[combined_idx]
    return private_data, total_priv_data, private_data_test


def generate_imbal_CIFAR_private_data(X, y, y_super, classes_per_party, N_parties,
                                      samples_per_class=7, read_saved=False):
    priv_data = [None] * N_parties
    priv_test_data = [None] * N_parties
    combined_idxs = []

    print('classes per party:', classes_per_party)
    count = 0
    for subcls_list in classes_per_party:
        idxs_per_party = []
        idxs_test_per_party = []

        if read_saved:
            idxs_per_party = np.load('dataset/cifar/noniid/train_party' + str(count) + '.npy')
            idxs_test_per_party = np.load('dataset/cifar/noniid/test_party' + str(count) + '.npy')
        else:
            for c in subcls_list:
                idxs = np.flatnonzero(y == c)
                idxs = np.random.choice(idxs, samples_per_class, replace=False)
                test_idx = np.random.choice(idxs, int(0.7 * len(idxs)), replace=False)
                train_idx = np.array(list(set(idxs) - set(test_idx)))

                idxs_per_party.append(train_idx)
                idxs_test_per_party.append(test_idx)

            idxs_per_party = np.hstack(idxs_per_party)
            idxs_test_per_party = np.hstack(idxs_test_per_party)
            np.save('dataset/cifar/noniid/train_party' + str(count), idxs_per_party)
            np.save('dataset/cifar/noniid/test_party' + str(count), idxs_test_per_party)

        combined_idxs.append(idxs_per_party)
        dict_to_add = {}
        dict_to_add["idx"] = idxs_per_party
        dict_to_add["X"] = X[idxs_per_party]

        dict_to_add["y"] = y_super[idxs_per_party]
        dict_to_add["y_temporary"] = y_super[idxs_per_party]
        dict_to_add["y_fine"] = y[idxs_per_party]

        priv_data[count] = dict_to_add
        priv_test_data[count] = {"X": X[idxs_test_per_party], "y": y_super[idxs_test_per_party],
                                 "y_temporary": y_super[idxs_test_per_party], "idx": idxs_test_per_party,
                                 "y_fine": y[idxs_test_per_party]}
        count += 1

    combined_idxs = np.hstack(combined_idxs)
    total_priv_data = {}
    total_priv_data["idx"] = combined_idxs
    total_priv_data["X"] = X[combined_idxs]

    total_priv_data["y_fine"] = y[combined_idxs]
    total_priv_data["y"] = y_super[combined_idxs]
    total_priv_data["y_temporary"] = y_super[combined_idxs]
    return priv_data, total_priv_data, priv_test_data
