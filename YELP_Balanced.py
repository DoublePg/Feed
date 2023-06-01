import os
import errno
import argparse
import sys
import pickle

import numpy as np
from tensorflow.keras.models import load_model

from data_utils import load_YELP_data, generate_bal_private_data, get_index
from Neural_Networks import train_models, cnn1d_model, mmoe_cnn1d_model
from methods.FedMD import FedMD
from methods.Feed import Feed
from methods.FedAvg import FedAvg
from methods.FedRep import FedRep
from methods.FedPer import FedPer
from methods.FedPhp import FedPhp
from methods.FedProx import FedProx
from methods.Ditto import Ditto
from methods.MOON import MOON
from methods.FedProto import FedProto
from methods.FedBabu import FedBabu
from methods.FedProc import FedProc
from methods.LGFedAvg import LGFedAvg


def parseArg():
    parser = argparse.ArgumentParser(description='FedMD, a federated learning framework. \
    Participants are training collaboratively. ')
    parser.add_argument('--conf', type=str, default="conf/YELP_balanced_conf.json",
                        help='the config file.'
                        )

    parser.add_argument('--method', type=str, default='Feed',
                        help='the method used.'
                        )

    args = parser.parse_args()

    method = args.method
    conf_file = args.conf
    return conf_file, method


CANDIDATE_MODELS = {"CNN1d": cnn1d_model,
                    "mmoe_CNN1d": mmoe_cnn1d_model}

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    conf_file, method = parseArg()
    with open(conf_file, "r") as f:
        conf_dict = eval(f.read())

        public_dataset = conf_dict["public_dataset"]
        private_dataset_train = conf_dict["private_dataset_train"]
        private_dataset_test = conf_dict["private_dataset_test"]
        word_vector_file = conf_dict["word_vector_file"]
        max_sequence_length = conf_dict["max_sequence_length"]
        max_words_num = conf_dict["max_words_num"]
        embedding_dim = conf_dict["embedding_dim"]

        model_config = conf_dict["models"]
        mmoe_model_config = conf_dict["mmoe_models"]
        n_experts = conf_dict["N_experts"]
        pre_train_params = conf_dict["pre_train_params"]
        model_saved_dir = conf_dict["model_saved_dir"]
        model_saved_names = conf_dict["model_saved_names"]
        is_early_stopping = conf_dict["early_stopping"]
        public_classes = conf_dict["public_classes"]
        private_classes = conf_dict["private_classes"]
        n_classes = len(public_classes)

        emnist_data_dir = conf_dict["EMNIST_dir"]
        N_parties = conf_dict["N_parties"]
        N_samples_per_class = conf_dict["N_samples_per_class"]

        N_rounds = conf_dict["N_rounds"]
        N_alignment = conf_dict["N_alignment"]
        N_private_training_round = conf_dict["N_private_training_round"]
        private_training_batchsize = conf_dict["private_training_batchsize"]
        N_logits_matching_round = conf_dict["N_logits_matching_round"]
        logits_matching_batchsize = conf_dict["logits_matching_batchsize"]

        result_save_dir = conf_dict["result_save_dir"]

    del conf_dict, conf_file

    word_index, embeddings_index = get_index(word_vector_file)
    num_words = max(max_words_num, len(word_index))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= max_words_num:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    X_train_pub, y_train_pub = load_YELP_data(public_dataset, word_index, max_sequence_length)

    public_dataset = {"X": X_train_pub, "y": y_train_pub}

    X_train_priv_train, y_train_priv_train = load_YELP_data(private_dataset_train, word_index, max_sequence_length)

    X_train_priv_test, y_train_priv_test = load_YELP_data(private_dataset_test, word_index, max_sequence_length)

    print("=" * 60)
    # generate private data
    private_data, total_private_data \
        = generate_bal_private_data(X_train_priv_train, y_train_priv_train,
                                    N_parties=N_parties,
                                    classes_in_use=private_classes,
                                    N_samples_per_class=N_samples_per_class,
                                    data_overlap=True, read_saved=True)

    private_test_data = {"X": X_train_priv_test, "y": y_train_priv_test}
    del X_train_priv_test, y_train_priv_test

    parties = []
    model_config_inuse = model_config
    if method == 'Feed':
        model_config_inuse = mmoe_model_config

    for i, item in enumerate(model_config_inuse):
        model_name = item["model_type"]
        model_params = item["params"]
        tmp = CANDIDATE_MODELS[model_name](n_classes=n_classes,
                                           num_words=num_words,
                                           embedding_dim=embedding_dim,
                                           max_sequence_length=max_sequence_length,
                                           embedding_matrix=embedding_matrix)
        print("model {0} : {1}".format(i, model_saved_names[i]))
        print(tmp.summary())
        parties.append(tmp)
        del model_name, model_params, tmp

    if method == 'Feed' or method == 'FedMD':
        print('------------------start pre-train-------------------')
        train_models(parties,
                     X_train_pub, y_train_pub,
                     X_train_pub, y_train_pub,
                     save_dir=model_saved_dir, save_names=model_saved_names,
                     early_stopping=is_early_stopping,
                     **pre_train_params
                     )

    print('-----------------start train ' + method + '--------------------')
    dataset_name = 'yelp'
    if method == 'Ditto':
        ditto = Ditto(parties,
                      public_dataset=public_dataset,
                      private_data=private_data,
                      total_private_data=total_private_data,
                      private_test_data=private_test_data,
                      N_rounds=N_rounds,
                      N_private_training_round=N_private_training_round,
                      private_training_batchsize=private_training_batchsize,
                      dataset=dataset_name)
        collaboration_performance = ditto.collaborative_training()

    elif method == 'FedAvg':
        fedavg = FedAvg(parties,
                        public_dataset=public_dataset,
                        private_data=private_data,
                        total_private_data=total_private_data,
                        private_test_data=private_test_data,
                        N_rounds=N_rounds,
                        N_private_training_round=N_private_training_round,
                        private_training_batchsize=private_training_batchsize,
                        dataset=dataset_name)
        collaboration_performance = fedavg.collaborative_training()

    elif method == 'FedBabu':
        fedbabu = FedBabu(parties,
                          public_dataset=public_dataset,
                          private_data=private_data,
                          total_private_data=total_private_data,
                          private_test_data=private_test_data,
                          N_rounds=N_rounds,
                          N_private_training_round=N_private_training_round,
                          private_training_batchsize=private_training_batchsize,
                          dataset=dataset_name)
        collaboration_performance = fedbabu.collaborative_training()

    elif method == 'FedMD':
        fedmd = FedMD(parties,
                      public_dataset=public_dataset,
                      private_data=private_data,
                      total_private_data=total_private_data,
                      private_test_data=private_test_data,
                      N_rounds=N_rounds,
                      N_alignment=N_alignment,
                      N_logits_matching_round=N_logits_matching_round,
                      logits_matching_batchsize=logits_matching_batchsize,
                      N_private_training_round=N_private_training_round,
                      private_training_batchsize=private_training_batchsize,
                      dataset=dataset_name)
        collaboration_performance = fedmd.collaborative_training()

    elif method == 'FedPer':
        fedper = FedPer(parties,
                        public_dataset=public_dataset,
                        private_data=private_data,
                        total_private_data=total_private_data,
                        private_test_data=private_test_data,
                        N_rounds=N_rounds,
                        N_private_training_round=N_private_training_round,
                        private_training_batchsize=private_training_batchsize,
                        dataset=dataset_name
                        )
        collaboration_performance = fedper.collaborative_training()

    elif method == 'FedPhp':
        fedphp = FedPhp(parties,
                        public_dataset=public_dataset,
                        private_data=private_data,
                        total_private_data=total_private_data,
                        private_test_data=private_test_data,
                        N_rounds=N_rounds,
                        N_alignment=N_alignment,
                        N_logits_matching_round=N_logits_matching_round,
                        logits_matching_batchsize=logits_matching_batchsize,
                        N_private_training_round=N_private_training_round,
                        private_training_batchsize=private_training_batchsize,
                        dataset=dataset_name
                        )
        collaboration_performance = fedphp.collaborative_training()

    elif method == 'FedProc':
        fedproc = FedProc(parties,
                          public_dataset=public_dataset,
                          private_data=private_data,
                          total_private_data=total_private_data,
                          private_test_data=private_test_data,
                          N_rounds=N_rounds,
                          N_private_training_round=N_private_training_round,
                          private_training_batchsize=private_training_batchsize,
                          dataset=dataset_name)
        collaboration_performance = fedproc.collaborative_training()

    elif method == 'FedProto':
        fedproto = FedProto(parties,
                            public_dataset=public_dataset,
                            private_data=private_data,
                            total_private_data=total_private_data,
                            private_test_data=private_test_data,
                            N_rounds=N_rounds,
                            N_private_training_round=N_private_training_round,
                            private_training_batchsize=private_training_batchsize,
                            dataset=dataset_name)
        collaboration_performance = fedproto.collaborative_training()

    elif method == 'FedProx':
        fedprox = FedProx(parties,
                          public_dataset=public_dataset,
                          private_data=private_data,
                          total_private_data=total_private_data,
                          private_test_data=private_test_data,
                          N_rounds=N_rounds,
                          N_private_training_round=N_private_training_round,
                          private_training_batchsize=private_training_batchsize,
                          dataset=dataset_name)
        collaboration_performance = fedprox.collaborative_training()

    elif method == 'FedRep':
        fedrep = FedRep(parties,
                        public_dataset=public_dataset,
                        private_data=private_data,
                        total_private_data=total_private_data,
                        private_test_data=private_test_data,
                        N_rounds=N_rounds,
                        N_private_training_round=N_private_training_round,
                        private_training_batchsize=private_training_batchsize,
                        dataset=dataset_name)
        collaboration_performance = fedrep.collaborative_training()

    elif method == 'LGFedAvg':
        lgfedavg = LGFedAvg(parties,
                            public_dataset=public_dataset,
                            private_data=private_data,
                            total_private_data=total_private_data,
                            private_test_data=private_test_data,
                            N_rounds=N_rounds,
                            N_private_training_round=N_private_training_round,
                            private_training_batchsize=private_training_batchsize,
                            dataset=dataset_name)
        collaboration_performance = lgfedavg.collaborative_training()

    elif method == 'MOON':
        moon = MOON(parties,
                    public_dataset=public_dataset,
                    private_data=private_data,
                    total_private_data=total_private_data,
                    private_test_data=private_test_data,
                    N_rounds=N_rounds,
                    N_private_training_round=N_private_training_round,
                    private_training_batchsize=private_training_batchsize,
                    dataset=dataset_name)
        collaboration_performance = moon.collaborative_training()

    elif method == 'Feed':
        feed = Feed(parties,
                    public_dataset=public_dataset,
                    private_data=private_data,
                    total_private_data=total_private_data,
                    private_test_data=private_test_data,
                    N_rounds=N_rounds,
                    N_alignment=N_alignment,
                    N_logits_matching_round=N_logits_matching_round,
                    logits_matching_batchsize=logits_matching_batchsize,
                    N_private_training_round=N_private_training_round,
                    private_training_batchsize=private_training_batchsize,
                    n_experts=n_experts,
                    dataset=dataset_name)
        collaboration_performance = feed.collaborative_training()

    else:
        print('no such method!')
