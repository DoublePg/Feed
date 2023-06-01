import os
import errno
import argparse
import sys
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import time
from data_utils import load_MNIST_data, load_EMNIST_data, generate_EMNIST_writer_based_data, generate_partial_data
from Neural_Networks import train_models, cnn_2layer_fc_model, cnn_3layer_fc_model, mmoe_cnn2layer_model
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
    parser.add_argument('--conf', type=str, default="conf/FEMNIST_imbalanced_conf.json",
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
                    "mmoe_2_layer_CNN": mmoe_cnn2layer_model
                    }

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    conf_file, method = parseArg()
    with open(conf_file, "r") as f:
        conf_dict = eval(f.read())
        model_config = conf_dict["models"]
        mmoe_model_config = conf_dict["mmoe_models"]
        n_experts = conf_dict['N_experts']
        md_model_config = conf_dict["md_models"]
        pre_train_params = conf_dict["pre_train_params"]
        model_saved_dir = conf_dict["model_saved_dir"]
        model_saved_names = conf_dict["model_saved_names"]
        is_early_stopping = conf_dict["early_stopping"]
        public_classes = conf_dict["public_classes"]
        private_classes = conf_dict["private_classes"]
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

    del conf_dict

    X_train_MNIST, y_train_MNIST, X_test_MNIST, y_test_MNIST \
        = load_MNIST_data(standarized=True, verbose=True)

    public_dataset = {"X": X_train_MNIST, "y": y_train_MNIST}

    X_train_EMNIST, y_train_EMNIST, X_test_EMNIST, y_test_EMNIST, \
        writer_ids_train_EMNIST, writer_ids_test_EMNIST \
        = load_EMNIST_data(emnist_data_dir,
                           standarized=True, verbose=True)

    y_train_EMNIST += len(public_classes)
    y_test_EMNIST += len(public_classes)

    # generate private data
    private_data, total_private_data, private_locdata_test \
        = generate_EMNIST_writer_based_data(X_train_EMNIST, y_train_EMNIST,
                                            writer_ids_train_EMNIST,
                                            N_parties=N_parties,
                                            classes_in_use=private_classes,
                                            N_priv_data_min=N_samples_per_class * len(private_classes),
                                            read_saved=True
                                            )

    X_tmp, y_tmp = generate_partial_data(X=X_test_EMNIST, y=y_test_EMNIST,
                                         class_in_use=private_classes, verbose=True)
    private_test_data = {"X": X_tmp, "y": y_tmp}
    del X_tmp, y_tmp

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
                                           input_shape=(28, 28),
                                           **model_params)
        print("model {0} : {1}".format(i, model_saved_names[i]))
        print(tmp.summary())
        parties.append(tmp)

        del model_name, model_params, tmp

    if method == 'Feed' or method == 'FedMD':
        print('----------start pre-train-----------------')
        train_models(parties,
                     X_train_MNIST, y_train_MNIST,
                     X_test_MNIST, y_test_MNIST,
                     save_dir=model_saved_dir,
                     save_names=model_saved_names,
                     early_stopping=is_early_stopping,
                     **pre_train_params
                     )

    print('-----------------start train ' + method + '--------------------')
    dataset_name = 'emnist'
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
        fedbabu = FedBabu(parties,
                          public_dataset=public_dataset,
                          private_data=private_data,
                          total_private_data=total_private_data,
                          private_test_data=private_test_data,
                          N_rounds=N_rounds,
                          N_private_training_round=N_private_training_round,
                          private_training_batchsize=private_training_batchsize,
                          private_test_localdata=private_locdata_test,
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
                      private_test_localdata=private_locdata_test,
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
                        dataset=dataset_name
                        )
        collaboration_performance = fedphp.collaborative_training()

    elif method == 'FedProc':
        print('--------------start training FedProc-------------------')
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
                    dataset=dataset_name
                    )
        collaboration_performance1 = feed.collaborative_training()

    else:
        print('no such method!')
