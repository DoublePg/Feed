import os
import errno
import argparse
import sys
import pickle
import time
import numpy as np
from tensorflow.keras.models import load_model
import random
from data_utils import load_CIFAR_data, load_CIFAR_from_local, generate_partial_data, generate_imbal_CIFAR_private_data
from Neural_Networks import train_models, cnn_2layer_fc_model, cnn_3layer_fc_model, mmoe_cnn2layer_model, \
    mmoe_cnn3layer_model
from methods.Feed import Feed
from methods.FedMD import FedMD
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
    parser.add_argument('--conf', type=str, default="conf/CIFAR_imbalanced_conf.json",
                        help='the config file.'
                        )

    parser.add_argument('--method', type=str, default='Feed',
                        help='the method used.'
                        )

    args = parser.parse_args()

    method = args.method
    conf_file = args.conf
    return conf_file, method


CANDIDATE_MODELS = {"2_layer_CNN": cnn_2layer_fc_model,
                    "3_layer_CNN": cnn_3layer_fc_model,
                    "mmoe_2_layer_CNN": mmoe_cnn2layer_model,
                    'mmoe_3_layer_CNN': mmoe_cnn3layer_model}

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    conf_file, method = parseArg()
    with open(conf_file, "r") as f:
        conf_dict = eval(f.read())

        model_config = conf_dict["models"]
        mmoe_model_config = conf_dict["mmoe_models"]
        md_model_config = conf_dict["md_models"]
        n_experts = conf_dict["N_experts"]
        pre_train_params = conf_dict["pre_train_params"]
        model_saved_dir = conf_dict["model_saved_dir"]
        model_saved_names = conf_dict["model_saved_names"]
        is_early_stopping = conf_dict["early_stopping"]
        public_classes = conf_dict["public_classes"]
        public_classes.sort()
        private_classes = conf_dict["private_classes"]
        private_classes.sort()
        n_classes = len(public_classes) + len(private_classes)

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

    X_train_CIFAR10, y_train_CIFAR10, X_test_CIFAR10, y_test_CIFAR10 \
        = load_CIFAR_data(data_type="CIFAR10",
                          standarized=True, verbose=True)

    public_dataset = {"X": X_train_CIFAR10, "y": y_train_CIFAR10}

    X_train_CIFAR100, y_train_CIFAR100, X_test_CIFAR100, y_test_CIFAR100 \
        = load_CIFAR_data(data_type="CIFAR100",
                          standarized=True, verbose=True)

    a_, y_train_super, b_, y_test_super \
        = load_CIFAR_data(data_type="CIFAR100", label_mode="coarse",
                          standarized=True, verbose=True)
    del a_, b_

    # Find the relations between superclasses and subclasses
    relations = [set() for i in range(np.max(y_train_super) + 1)]
    for i, y_fine in enumerate(y_train_CIFAR100):
        relations[y_train_super[i]].add(y_fine)
    for i in range(len(relations)):
        relations[i] = list(relations[i])

    print('relation of super classs and subclass:', relations)
    del i, y_fine

    fine_classes_in_use = [[relations[j][i % 5] for j in private_classes]
                           for i in range(N_parties)]

    print('fine classes in use', fine_classes_in_use)

    # Generate test set
    X_tmp, y_tmp = generate_partial_data(X_test_CIFAR100, y_test_super,
                                         class_in_use=private_classes,
                                         verbose=True)

    # relabel the selected CIFAR100 data for future convenience
    for index in range(len(private_classes) - 1, -1, -1):
        cls_ = private_classes[index]
        y_tmp[y_tmp == cls_] = index + len(public_classes)
    private_test_data = {"X": X_tmp, "y": y_tmp}
    del index, cls_, X_tmp, y_tmp

    print("=" * 60)

    # generate private data
    private_data, total_private_data, private_locdata_test \
        = generate_imbal_CIFAR_private_data(X_train_CIFAR100, y_train_CIFAR100, y_train_super,
                                            N_parties=N_parties,
                                            classes_per_party=fine_classes_in_use,
                                            samples_per_class=N_samples_per_class,
                                            read_saved=True)

    for index in range(len(private_classes) - 1, -1, -1):
        cls_ = private_classes[index]
        total_private_data["y"][total_private_data["y_temporary"] == cls_] = index + len(public_classes)
        for i in range(N_parties):
            private_data[i]["y"][private_data[i]["y_temporary"] == cls_] = index + len(public_classes)
            private_locdata_test[i]["y"][private_locdata_test[i]["y_temporary"] == cls_] = index + len(public_classes)

    del index, cls_

    mod_private_classes = np.arange(len(private_classes)) + len(public_classes)

    print("=" * 60)
    parties = []
    model_config_inuse = model_config
    nclasses = n_classes
    if method == 'FedMD':
        model_config_inuse = md_model_config
    elif method == 'Feed':
        model_config_inuse = mmoe_model_config
    elif method == 'MOON' or method == 'FedProc':
        nclasses = len(private_classes)
    for i, item in enumerate(model_config_inuse):
        model_name = item["model_type"]
        model_params = item["params"]
        tmp = CANDIDATE_MODELS[model_name](n_classes=nclasses,
                                           input_shape=(32, 32, 3),
                                           **model_params)
        print("model {0} : {1}".format(i, model_saved_names[i]))
        print(tmp.summary())
        parties.append(tmp)

        del model_name, model_params, tmp
    if method == 'Feed' or method == 'FedMD':
        print('------------------start pre-train-------------------')
        pre_train_result1 = train_models(parties,
                                         X_train_CIFAR10, y_train_CIFAR10,
                                         X_test_CIFAR10, y_test_CIFAR10,
                                         save_dir=model_saved_dir, save_names=model_saved_names,
                                         early_stopping=is_early_stopping,
                                         **pre_train_params
                                         )

    del X_train_CIFAR10, y_train_CIFAR10, X_test_CIFAR10, y_test_CIFAR10, \
        X_train_CIFAR100, y_train_CIFAR100, X_test_CIFAR100, y_test_CIFAR100, y_train_super, y_test_super

    print('-----------------start train ' + method + '--------------------')
    dataset_name = 'cifar'
    if method == 'Ditto':
        ditto = Ditto(parties,
                      public_dataset=public_dataset,
                      private_data=private_data,
                      total_private_data=total_private_data,
                      private_test_data=private_test_data,
                      N_rounds=N_rounds,
                      N_private_training_round=N_private_training_round,
                      private_training_batchsize=private_training_batchsize,
                      private_test_localdata=private_locdata_test,
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
                        private_test_localdata=private_locdata_test,
                        dataset=dataset_name)
        collaboration_performance = fedavg.collaborative_training()

    elif method == 'FedBabu':
        fedavg = FedBabu(parties,
                         public_dataset=public_dataset,
                         private_data=private_data,
                         total_private_data=total_private_data,
                         private_test_data=private_test_data,
                         N_rounds=N_rounds,
                         N_private_training_round=N_private_training_round,
                         private_training_batchsize=private_training_batchsize,
                         private_test_localdata=private_locdata_test,
                         dataset=dataset_name)
        collaboration_performance = fedavg.collaborative_training()

    elif method == 'FedMD':
        fedmd = FedMD(parties,
                      public_dataset=public_dataset,
                      private_data=private_data,
                      total_private_data=total_private_data,
                      private_test_data=private_test_data,
                      N_rounds=N_rounds,
                      N_private_training_round=N_private_training_round,
                      private_training_batchsize=private_training_batchsize,
                      private_test_localdata=private_locdata_test,
                      N_alignment=N_alignment,
                      N_logits_matching_round=N_logits_matching_round,
                      logits_matching_batchsize=logits_matching_batchsize,
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
                        private_test_localdata=private_locdata_test,
                        dataset=dataset_name)
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
                        private_test_localdata=private_locdata_test,
                        dataset=dataset_name)
        fedphp.collaborative_training()

    elif method == 'FedProc':
        fedproc = FedProc(parties,
                          public_dataset=public_dataset,
                          private_data=private_data,
                          total_private_data=total_private_data,
                          private_test_data=private_test_data,
                          N_rounds=N_rounds,
                          N_private_training_round=N_private_training_round,
                          private_training_batchsize=private_training_batchsize,
                          private_test_localdata=private_locdata_test,
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
                            private_test_localdata=private_locdata_test,
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
                          private_test_localdata=private_locdata_test,
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
                        private_test_localdata=private_locdata_test,
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
                            private_test_localdata=private_locdata_test,
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
                    private_test_localdata=private_locdata_test,
                    mu=0.1,
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
                    private_test_localdata=private_locdata_test,
                    dataset=dataset_name)
        feed.collaborative_training()

    else:
        print('no such method!')
