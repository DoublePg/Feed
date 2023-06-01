import numpy as np
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import pickle
from data_utils import generate_alignment_data
from Neural_Networks import remove_last_layer
from sklearn import metrics
from keras import backend as K


class FedRep():
    def __init__(self,
                 parties,
                 public_dataset,
                 private_data,
                 total_private_data,
                 private_test_data,
                 N_rounds,
                 N_private_training_round,
                 private_training_batchsize,
                 private_test_localdata=None,
                 dataset='cifar'):

        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.private_data = private_data
        self.private_test_data = private_test_data

        self.N_rounds = N_rounds
        self.N_private_training_round = N_private_training_round
        self.private_test_localdata = private_test_localdata
        self.private_training_batchsize = private_training_batchsize
        self.dataset = dataset

        self.collaborative_parties = []
        self.init_result = []

        self.weight_per_party = []
        sum_weight = 0
        for i in range(self.N_parties):
            self.weight_per_party.append(len(private_data[i]))
            sum_weight += len(private_data[i])
        self.weight_per_party = [x / sum_weight for x in self.weight_per_party]

        print("start model initialization: ")
        for i in range(self.N_parties):
            print("model ", i)
            model_A_twin = None
            model_A_twin = clone_model(parties[i])
            model_A_twin.set_weights(parties[i].get_weights())
            model_A_twin.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3, decay=0.001),
                                 loss="sparse_categorical_crossentropy",
                                 metrics=["accuracy"])

            print("start full stack training ... ")
            model_A_twin.fit(private_data[i]["X"], private_data[i]["y"],
                             batch_size=self.private_training_batchsize,
                             epochs=25,
                             shuffle=True, verbose=0,
                             validation_data=[private_test_data["X"], private_test_data["y"]],
                             callbacks=[EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10)]
                             )
            print("full stack training done")

            model_A = remove_last_layer(model_A_twin, loss="mean_absolute_error")

            self.collaborative_parties.append({"model_logits": model_A,
                                               "model_classifier": model_A_twin,
                                               'best_global_record': {"acc": 0.0, "f1": 0.0, "auc": 0.0},
                                               'best_global_model': model_A_twin.get_weights(),
                                               'best_loc_record': {"acc": 0.0, "f1": 0.0, "auc": 0.0},
                                               'best_loc_model': model_A_twin.get_weights(),
                                               'old_optim': model_A_twin.optimizer})

            del model_A, model_A_twin
        # END FOR LOOP

    def collaborative_training(self):
        # start collaborating training
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        r = 0
        while True:

            print("round ", r)
            print("test performance ... ")
            # test performance
            acc_lst = []
            f1_lst = []
            auc_lst = []
            for index, d in enumerate(self.collaborative_parties):
                y_pre_mul = d["model_classifier"].predict(self.private_test_data["X"], verbose=0)
                y_pred = y_pre_mul.argmax(axis=1)
                collaboration_performance[index].append(np.mean(self.private_test_data["y"] == y_pred))
                if self.dataset == 'yelp':
                    macro_f1 = metrics.f1_score(self.private_test_data["y"], y_pred, average='macro')
                    roc = metrics.roc_auc_score(self.private_test_data["y"], y_pred)
                else:
                    macro_f1 = metrics.f1_score(self.private_test_data["y"], y_pred, labels=[i for i in range(10, 16)],
                                                average='macro')
                    roc = metrics.roc_auc_score(self.private_test_data["y"], y_pre_mul, multi_class='ovo',
                                                labels=[i for i in range(16)])

                if collaboration_performance[index][-1] > d['best_global_record']['acc']:
                    d['best_global_record']['acc'] = collaboration_performance[index][-1]
                if macro_f1 > d['best_global_record']['f1']:
                    d['best_global_record']['f1'] = macro_f1
                if roc > d['best_global_record']['auc']:
                    d['best_global_record']['auc'] = roc

                acc_lst.append(collaboration_performance[index][-1])
                f1_lst.append(macro_f1)
                auc_lst.append(roc)
                del y_pred

            print('acc:')
            for acc in acc_lst:
                print(acc)
            print('f1:')
            for f1 in f1_lst:
                print(f1)
            print('auc:')
            for auc in auc_lst:
                print(auc)

            if self.private_test_localdata is not None:
                acc_lst = []
                f1_lst = []
                auc_lst = []
                print('test performance on local dataset')
                for index, d in enumerate(self.collaborative_parties):
                    y_pre_mul = d["model_classifier"].predict(self.private_test_localdata[index]["X"], verbose=0)
                    y_pred = y_pre_mul.argmax(axis=1)
                    collaboration_performance[index].append(np.mean(self.private_test_localdata[index]["y"] == y_pred))
                    if self.dataset == 'yelp':
                        macro_f1 = metrics.f1_score(self.private_test_localdata[index]["y"], y_pred,
                                                    average='macro')
                        roc = metrics.roc_auc_score(self.private_test_localdata[index]["y"], y_pred)
                    else:
                        macro_f1 = metrics.f1_score(self.private_test_localdata[index]["y"], y_pred,
                                                    labels=[i for i in range(10, 16)],
                                                    average='macro')
                        roc = metrics.roc_auc_score(self.private_test_localdata[index]["y"], y_pre_mul,
                                                    multi_class='ovo',
                                                    labels=[i for i in range(16)])
                    if collaboration_performance[index][-1] > d['best_loc_record']['acc']:
                        d['best_loc_record']['acc'] = collaboration_performance[index][-1]
                    if macro_f1 > d['best_loc_record']['f1']:
                        d['best_loc_record']['f1'] = macro_f1
                    if roc > d['best_loc_record']['auc']:
                        d['best_loc_record']['auc'] = roc
                    del y_pred
                    acc_lst.append(collaboration_performance[index][-1])
                    f1_lst.append(macro_f1)
                    auc_lst.append(roc)

                print('acc:')
                for acc in acc_lst:
                    print(acc)
                print('f1:')
                for f1 in f1_lst:
                    print(f1)
                print('auc:')
                for auc in auc_lst:
                    print(auc)

            r += 1
            if r > self.N_rounds:
                for index, d in enumerate(self.collaborative_parties):
                    print('party ' + str(index) + ' 的global best record is ' + str(d['best_global_record']))
                print('------------------------------------')
                for index, d in enumerate(self.collaborative_parties):
                    print('party ' + str(index) + ' 的local best record is ' + str(d['best_loc_record']))
                break

            sum_convs = [0 for k in range(3)]
            sum_bs = [0 for k in range(3)]
            sum_bns = [[0 for h in range(4)] for k in range(3)]

            for j in range(self.N_parties):
                wconv1, bias1 = self.collaborative_parties[j]["model_classifier"].get_layer(
                    name="Conv1").get_weights()
                wconv2, bias2 = self.collaborative_parties[j]["model_classifier"].get_layer(
                    name="Conv2").get_weights()
                sum_convs[0] += (wconv1 * self.weight_per_party[j])
                sum_convs[1] += (wconv2 * self.weight_per_party[j])
                sum_bs[0] += (bias1 * self.weight_per_party[j])
                sum_bs[1] += (bias2 * self.weight_per_party[j])

                if self.dataset != 'emnist':
                    wconv3, bias3 = self.collaborative_parties[j]["model_classifier"].get_layer(
                        name="Conv3").get_weights()
                    sum_convs[2] += (wconv3 * self.weight_per_party[j])
                    sum_bs[2] += (bias3 * self.weight_per_party[j])

                if self.dataset != 'yelp':
                    bn1 = self.collaborative_parties[j]["model_classifier"].get_layer(
                        name="BN1").get_weights()
                    bn2 = self.collaborative_parties[j]["model_classifier"].get_layer(
                        name="BN2").get_weights()
                    if self.dataset == 'cifar':
                        bn3 = self.collaborative_parties[j]["model_classifier"].get_layer(
                            name="BN3").get_weights()

                    for k in range(4):
                        sum_bns[0][k] += (bn1[k] * self.weight_per_party[j])
                        sum_bns[1][k] += (bn2[k] * self.weight_per_party[j])
                        if self.dataset == 'cifar':
                            sum_bns[2][k] += (bn3[k] * self.weight_per_party[j])

            print("updates models ...")
            for index, d in enumerate(self.collaborative_parties):
                d["model_classifier"].get_layer(
                    name="Conv1").set_weights([sum_convs[0], sum_bs[0]])
                d["model_classifier"].get_layer(
                    name="Conv2").set_weights([sum_convs[1], sum_bs[1]])
                if self.dataset != 'emnist':
                    d["model_classifier"].get_layer(
                        name="Conv3").set_weights([sum_convs[2], sum_bs[2]])

                if self.dataset != 'yelp':
                    d["model_classifier"].get_layer(
                        name="BN1").set_weights(sum_bns[0])
                    d["model_classifier"].get_layer(
                        name="BN2").set_weights(sum_bns[1])
                    if self.dataset == 'cifar':
                        d["model_classifier"].get_layer(
                            name="BN3").set_weights(sum_bns[2])

                print("model {0} starting training with private data... ".format(index))

                if self.dataset != 'emnist':
                    d["model_classifier"].get_layer(name="Conv3").trainable = False
                if self.dataset != 'yelp':
                    d["model_classifier"].get_layer(name="BN1").trainable = False
                    d["model_classifier"].get_layer(name="BN2").trainable = False
                    if self.dataset == 'cifar':
                        d["model_classifier"].get_layer(name="BN3").trainable = False
                d["model_classifier"].get_layer(name="fc1").trainable = True
                d["model_classifier"].compile(loss="sparse_categorical_crossentropy",
                                              metrics=["accuracy"])
                d["model_classifier"].optimizer = d['old_optim']

                d["model_classifier"].fit(self.private_data[index]["X"],
                                          self.private_data[index]["y"],
                                          batch_size=self.private_training_batchsize,
                                          epochs=self.N_private_training_round,
                                          shuffle=True, verbose=0)
                print('private data train loss: ', d["model_classifier"].history.history['loss'])
                print('private data train acc: ', d["model_classifier"].history.history['accuracy'])
                d['old_optim'] = d["model_classifier"].optimizer

                if self.dataset != 'emnist':
                    d["model_classifier"].get_layer(name="Conv3").trainable = True
                if self.dataset != 'yelp':
                    d["model_classifier"].get_layer(name="BN1").trainable = True
                    d["model_classifier"].get_layer(name="BN2").trainable = True
                    if self.dataset == 'cifar':
                        d["model_classifier"].get_layer(name="BN3").trainable = True
                d["model_classifier"].get_layer(name="fc1").trainable = False

                d["model_classifier"].compile(loss="sparse_categorical_crossentropy",
                                              metrics=["accuracy"])
                d["model_classifier"].optimizer = d['old_optim']
                d["model_classifier"].fit(self.private_data[index]["X"],
                                          self.private_data[index]["y"],
                                          batch_size=self.private_training_batchsize,
                                          epochs=self.N_private_training_round,
                                          shuffle=True, verbose=0)
                d['old_optim'] = d["model_classifier"].optimizer

                print('private data train loss: ', d["model_classifier"].history.history['loss'])
                print('private data train acc: ', d["model_classifier"].history.history['accuracy'])

                print("model {0} done private training. \n".format(index))
            # END FOR LOOP

        # END WHILE LOOP
        return collaboration_performance
