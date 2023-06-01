import numpy as np
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.utils.np_utils import *
from tensorflow.keras.layers import Input, Dense, add, concatenate, Conv2D, Dropout, \
    BatchNormalization, Flatten, MaxPooling2D, AveragePooling2D, Activation, Dropout, Reshape, RepeatVector, multiply, \
    add
from tensorflow.keras.models import save_model, Model

from data_utils import generate_alignment_data
from Neural_Networks import remove_last_layer
from losses import kd_loss, fedphp_hloss
from sklearn import metrics


class Feed():
    def __init__(self,
                 parties,
                 public_dataset,
                 private_data,
                 total_private_data,
                 private_test_data,
                 N_alignment,
                 N_rounds,
                 N_logits_matching_round,
                 logits_matching_batchsize,
                 N_private_training_round,
                 private_training_batchsize,
                 n_experts,
                 private_test_localdata=None,
                 model_path=None,
                 mu=0.9,
                 lambd=0.01,
                 dataset='cifar'
                 ):

        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.private_test_localdata = private_test_localdata
        self.N_alignment = N_alignment

        self.N_experts = n_experts
        self.N_rounds = N_rounds
        self.N_logits_matching_round = N_logits_matching_round
        self.logits_matching_batchsize = logits_matching_batchsize
        self.N_private_training_round = N_private_training_round
        self.private_training_batchsize = private_training_batchsize
        self.mu = mu
        self.lambd = lambd
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
            model_A_twin = clone_model(parties[i])
            model_A_twin.set_weights(parties[i].get_weights())
            model_A_twin.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3, decay=0.001),
                                 loss="sparse_categorical_crossentropy",
                                 metrics=["accuracy"])

            print("start full stack training ... ")

            model_A_twin.fit(private_data[i]["X"], private_data[i]["y"],
                             batch_size=self.private_training_batchsize, epochs=25, shuffle=True, verbose=0,
                             validation_data=[private_test_data["X"], private_test_data["y"]],
                             callbacks=[EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10)]
                             )

            print("full stack training done")
            model_classifier = clone_model(parties[i])
            model_classifier.set_weights(model_A_twin.get_weights())
            model_classifier_mulo = Model(inputs=model_classifier.inputs,
                                          outputs={"pred": model_classifier.outputs,
                                                   'embeddings': model_classifier.get_layer("embeddings").output})

            ploss = fedphp_hloss(model_classifier_mulo.get_layer("embeddings").output.shape[-1])
            model_classifier_mulo.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3, decay=0.001),
                                          loss={'pred': "sparse_categorical_crossentropy",
                                                'embeddings': ploss},
                                          loss_weights={
                                              'pred': 1.0 - self.lambd,
                                              'embeddings': self.lambd
                                          },
                                          metrics=["accuracy"])

            model_A = remove_last_layer(model_A_twin, loss="mean_absolute_error")
            print('compile model')
            model_A.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                            loss=kd_loss,
                            metrics=["accuracy"])

            HPM = clone_model(model_A)
            HPM.set_weights(model_A.get_weights())

            self.collaborative_parties.append({"model_logits": model_A,
                                               "model_classifier": model_classifier_mulo,
                                               "model_weights": model_A_twin.get_weights(),
                                               'best_global_record': {"acc": 0.0, "f1": 0.0, "auc": 0.0},
                                               'best_loc_record': {"acc": 0.0, "f1": 0.0, "auc": 0.0},
                                               'HPM': HPM,
                                               'mu': self.mu
                                               })

            del model_A, model_A_twin, model_classifier, model_classifier_mulo

    def collaborative_training(self):
        # start collaborating training
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        r = 0

        val_data = generate_alignment_data(self.private_test_data["X"],
                                           self.private_test_data["y"],
                                           200)

        while True:
            # At beginning of each round, generate new alignment dataset，取public_dataset的一个子集
            alignment_data = generate_alignment_data(self.public_dataset["X"],
                                                     self.public_dataset["y"],
                                                     self.N_alignment)

            print("round ", r)

            print("update logits ... ")
            # update logits
            logits = 0

            for d in self.collaborative_parties:
                d["model_logits"].set_weights(d["model_weights"])
                logits += d["model_logits"].predict(alignment_data["X"], verbose=0)

            logits /= self.N_parties

            sum_convs = [[0 for k in range(3)] for _ in range(self.N_experts)]
            sum_bs = [[0 for k in range(3)] for _ in range(self.N_experts)]
            sum_bns = [[[0 for h in range(4)] for k in range(3)] for _ in
                       range(self.N_experts)]

            for j in range(self.N_parties):
                for e in range(self.N_experts):
                    wconv1, bias1 = self.collaborative_parties[j]["model_classifier"].get_layer(
                        name="Conv1_expert" + str(e)).get_weights()
                    wconv2, bias2 = self.collaborative_parties[j]["model_classifier"].get_layer(
                        name="Conv2_expert" + str(e)).get_weights()
                    if self.dataset != 'emnist':
                        wconv3, bias3 = self.collaborative_parties[j]["model_classifier"].get_layer(
                            name="Conv3_expert" + str(e)).get_weights()

                    if self.dataset != 'yelp':
                        bn1 = self.collaborative_parties[j]["model_classifier"].get_layer(
                            name="BN1_expert" + str(e)).get_weights()
                        bn2 = self.collaborative_parties[j]["model_classifier"].get_layer(
                            name="BN2_expert" + str(e)).get_weights()
                        if self.dataset == 'cifar':
                            bn3 = self.collaborative_parties[j]["model_classifier"].get_layer(
                                name="BN3_expert" + str(e)).get_weights()

                    sum_convs[e][0] += (wconv1 * self.weight_per_party[j])
                    sum_convs[e][1] += (wconv2 * self.weight_per_party[j])
                    if self.dataset != 'emnist':
                        sum_convs[e][2] += (wconv3 * self.weight_per_party[j])
                    sum_bs[e][0] += (bias1 * self.weight_per_party[j])
                    sum_bs[e][1] += (bias2 * self.weight_per_party[j])
                    if self.dataset != 'emnist':
                        sum_bs[e][2] += (bias3 * self.weight_per_party[j])
                    if self.dataset != 'yelp':
                        # CNN1d没有BN层
                        for k in range(4):
                            sum_bns[e][0][k] += (bn1[k] * self.weight_per_party[j])
                            sum_bns[e][1][k] += (bn2[k] * self.weight_per_party[j])
                            if self.dataset == 'cifar':
                                sum_bns[e][2][k] += (bn3[k] * self.weight_per_party[j])

            print("test performance ... ")
            acc_lst = []
            f1_lst = []
            auc_lst = []
            for index, d in enumerate(self.collaborative_parties):
                embed, y_pre_mul = d["model_classifier"].predict(self.private_test_data["X"], verbose=0)
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
                    embed, y_pre_mul = d["model_classifier"].predict(self.private_test_localdata[index]["X"], verbose=0)
                    y_pred = y_pre_mul.argmax(axis=1)
                    collaboration_performance[index].append(np.mean(self.private_test_localdata[index]["y"] == y_pred))
                    if self.dataset == 'yelp':
                        macro_f1 = metrics.f1_score(self.private_test_localdata[index]["y"], y_pred, average='macro')
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
                print('--------------best Global record----------------------')
                for index, d in enumerate(self.collaborative_parties):
                    print(str(d['best_global_record']))
                print('---------------best Local record---------------------')
                for index, d in enumerate(self.collaborative_parties):
                    print(str(d['best_loc_record']))
                break

            print("updates models ...")
            for index, d in enumerate(self.collaborative_parties):
                print("model {0} starting alignment with public logits... ".format(index))

                weights_to_use = d["model_weights"]
                d["model_logits"].set_weights(weights_to_use)

                y_true_label = to_categorical(alignment_data["y"], num_classes=np.array(logits).shape[-1])
                y_train = np.c_[np.array(logits), y_true_label]

                print('start fit')
                d["model_logits"].fit(alignment_data["X"], y_train,
                                      batch_size=self.logits_matching_batchsize,
                                      epochs=self.N_logits_matching_round,
                                      shuffle=True, verbose=0)

                d["model_weights"] = d["model_logits"].get_weights()
                print("model {0} done alignment".format(index))

                print("model {0} starting training with private data... ".format(index))

                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)

                for e in range(self.N_experts):
                    d["model_classifier"].get_layer(
                        name="Conv1_expert" + str(e)).set_weights([sum_convs[e][0], sum_bs[e][0]])
                    d["model_classifier"].get_layer(
                        name="Conv2_expert" + str(e)).set_weights([sum_convs[e][1], sum_bs[e][1]])
                    if self.dataset != 'emnist':
                        d["model_classifier"].get_layer(
                            name="Conv3_expert" + str(e)).set_weights([sum_convs[e][2], sum_bs[e][2]])

                    if self.dataset != 'yelp':
                        d["model_classifier"].get_layer(
                            name="BN1_expert" + str(e)).set_weights(sum_bns[e][0])
                        d["model_classifier"].get_layer(
                            name="BN2_expert" + str(e)).set_weights(sum_bns[e][1])
                        if self.dataset == 'cifar':
                            d["model_classifier"].get_layer(
                                name="BN3_expert" + str(e)).set_weights(sum_bns[e][2])

                model_hpmh = Model(inputs=d['HPM'].inputs, outputs=d['HPM'].get_layer("embeddings").output)
                hpm_h = model_hpmh.predict(self.private_data[index]["X"], verbose=0)
                del model_hpmh

                d["model_classifier"].fit(self.private_data[index]["X"],
                                          {'pred': self.private_data[index]["y"], 'embeddings': hpm_h},
                                          batch_size=self.private_training_batchsize,
                                          epochs=self.N_private_training_round,
                                          shuffle=True, verbose=0)

                print('private data train loss: ', d["model_classifier"].history.history['loss'])
                print('private data train acc: ', d["model_classifier"].history.history['pred_accuracy'])

                d["model_weights"] = d["model_classifier"].get_weights()  # 更新保存的模型参数

                d['mu'] = self.mu * (r / self.N_rounds)

                for e in range(self.N_experts):
                    hpm_c1, hpm_b1 = d['HPM'].get_layer(name="Conv1_expert" + str(e)).get_weights()
                    hpm_c2, hpm_b2 = d['HPM'].get_layer(name="Conv2_expert" + str(e)).get_weights()
                    if self.dataset != 'emnist':
                        hpm_c3, hpm_b3 = d['HPM'].get_layer(name="Conv3_expert" + str(e)).get_weights()

                    if self.dataset != 'yelp':
                        hpm_bn1 = d['HPM'].get_layer(name="BN1_expert" + str(e)).get_weights()
                        hpm_bn2 = d['HPM'].get_layer(name="BN2_expert" + str(e)).get_weights()
                        if self.dataset == 'cifar':
                            hpm_bn3 = d['HPM'].get_layer(name="BN3_expert" + str(e)).get_weights()

                    c1, b1 = d['model_classifier'].get_layer(name="Conv1_expert" + str(e)).get_weights()
                    c2, b2 = d['model_classifier'].get_layer(name="Conv2_expert" + str(e)).get_weights()
                    if self.dataset != 'emnist':
                        c3, b3 = d['model_classifier'].get_layer(name="Conv3_expert" + str(e)).get_weights()

                    if self.dataset != 'yelp':
                        bn1 = d['model_classifier'].get_layer(name="BN1_expert" + str(e)).get_weights()
                        bn2 = d['model_classifier'].get_layer(name="BN2_expert" + str(e)).get_weights()
                        if self.dataset == 'cifar':
                            bn3 = d['model_classifier'].get_layer(name="BN3_expert" + str(e)).get_weights()

                    d["HPM"].get_layer(name="Conv1_expert" + str(e)).set_weights([c1 * (1 - d['mu']) + d['mu'] * hpm_c1,
                                                                                  b1 * (1 - d['mu']) + d[
                                                                                      'mu'] * hpm_b1])
                    d["HPM"].get_layer(name="Conv2_expert" + str(e)).set_weights([c2 * (1 - d['mu']) + d['mu'] * hpm_c2,
                                                                                  b2 * (1 - d['mu']) + d[
                                                                                      'mu'] * hpm_b2])
                    if self.dataset != 'emnist':
                        d["HPM"].get_layer(name="Conv3_expert" + str(e)).set_weights(
                            [c3 * (1 - d['mu']) + d['mu'] * hpm_c3,
                             b3 * (1 - d['mu']) + d[
                                 'mu'] * hpm_b3])

                    if self.dataset != 'yelp':
                        d["HPM"].get_layer(name="BN1_expert" + str(e)).set_weights(
                            [bn1[i] * (1 - d['mu']) + d['mu'] * hpm_bn1[i] for i in range(len(hpm_bn1))])
                        d["HPM"].get_layer(name="BN2_expert" + str(e)).set_weights(
                            [bn2[i] * (1 - d['mu']) + d['mu'] * hpm_bn2[i] for i in range(len(hpm_bn2))])
                        if self.dataset == 'cifar':
                            d["HPM"].get_layer(name="BN3_expert" + str(e)).set_weights(
                                [bn3[i] * (1 - d['mu']) + d['mu'] * hpm_bn3[i] for i in range(len(hpm_bn3))])

                print("model {0} done private training. \n".format(index))
            # END FOR LOOP

        # END WHILE LOOP
        return collaboration_performance
