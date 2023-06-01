import numpy as np
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from data_utils import generate_alignment_data
from Neural_Networks import remove_last_layer
from sklearn import metrics


class FedMD():
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
                 private_test_localdata=None,
                 dataset='cifar'):

        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.N_alignment = N_alignment

        self.N_rounds = N_rounds
        self.N_logits_matching_round = N_logits_matching_round
        self.logits_matching_batchsize = logits_matching_batchsize
        self.N_private_training_round = N_private_training_round
        self.private_test_localdata = private_test_localdata
        self.private_training_batchsize = private_training_batchsize

        self.collaborative_parties = []
        self.init_result = []
        self.dataset = dataset

        print("start model initialization: ")
        for i in range(self.N_parties):
            print("model ", i)
            model_A_twin = None
            model_A_twin = clone_model(parties[i])  # 复制了原模型模型的结构，重新构建了一个模型，但没有复制原模型的权重的值。
            model_A_twin.set_weights(parties[i].get_weights())  # 赋予原模型的权重
            model_A_twin.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                                 loss="sparse_categorical_crossentropy",
                                 metrics=["accuracy"])  # 配置训练方法

            print("start full stack training ... ")

            model_A_twin.fit(private_data[i]["X"], private_data[i]["y"],
                             batch_size=self.N_private_training_round, epochs=25, shuffle=True, verbose=0,
                             validation_data=[private_test_data["X"], private_test_data["y"]],
                             callbacks=[EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10)]
                             )

            print("full stack training done")

            model_A = remove_last_layer(model_A_twin, loss="mean_absolute_error")

            self.collaborative_parties.append({"model_logits": model_A,
                                               "model_classifier": model_A_twin,
                                               "model_weights": model_A_twin.get_weights(),
                                               'best_global_record': {"acc": 0.0, "f1": 0.0, "auc": 0.0},
                                               'best_global_model': model_A_twin.get_weights(),
                                               'best_loc_record': {"acc": 0.0, "f1": 0.0, "auc": 0.0},
                                               'best_loc_model': model_A_twin.get_weights()})

            del model_A, model_A_twin
        # END FOR LOOP

    def collaborative_training(self):
        # start collaborating training
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        r = 0
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

            print("test performance ... ")
            # test performance
            acc_lst = []
            f1_lst = []
            auc_lst = []
            for index, d in enumerate(self.collaborative_parties):
                y_pre_mul = d["model_classifier"].predict(self.private_test_data["X"], verbose=0)
                y_pred = y_pre_mul.argmax(axis=1)
                collaboration_performance[index].append(np.mean(self.private_test_data["y"] == y_pred))
                if self.dataset != 'yelp':
                    macro_f1 = metrics.f1_score(self.private_test_data["y"], y_pred, labels=[i for i in range(10, 16)],
                                                average='macro')
                    roc = metrics.roc_auc_score(self.private_test_data["y"], y_pre_mul, multi_class='ovo',
                                                labels=[i for i in range(16)])
                else:
                    macro_f1 = metrics.f1_score(self.private_test_data["y"], y_pred, average='macro')
                    roc = metrics.roc_auc_score(self.private_test_data["y"], y_pred)
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
                    if self.dataset != 'yelp':
                        macro_f1 = metrics.f1_score(self.private_test_localdata[index]["y"], y_pred,
                                                    labels=[i for i in range(10, 16)],
                                                    average='macro')
                        roc = metrics.roc_auc_score(self.private_test_localdata[index]["y"], y_pre_mul,
                                                    multi_class='ovo',
                                                    labels=[i for i in range(16)])
                    else:
                        macro_f1 = metrics.f1_score(self.private_test_localdata[index]["y"], y_pred,
                                                    average='macro')  # yelp二分类
                        roc = metrics.roc_auc_score(self.private_test_localdata[index]["y"], y_pred)  # yelp二分类
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

            print("updates models ...")
            for index, d in enumerate(self.collaborative_parties):
                print("model {0} starting alignment with public logits... ".format(index))

                weights_to_use = d["model_weights"]
                d["model_logits"].set_weights(weights_to_use)
                d["model_logits"].fit(alignment_data["X"], logits,
                                      batch_size=self.logits_matching_batchsize,
                                      epochs=self.N_logits_matching_round,
                                      shuffle=True, verbose=0)
                d["model_weights"] = d["model_logits"].get_weights()
                print("model {0} done alignment".format(index))

                print("model {0} starting training with private data... ".format(index))
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)

                d["model_classifier"].fit(self.private_data[index]["X"],
                                          self.private_data[index]["y"],
                                          batch_size=self.private_training_batchsize,
                                          epochs=self.N_private_training_round,
                                          shuffle=True, verbose=0)

                print('private data train loss: ', d["model_classifier"].history.history['loss'])
                print('private data train loss: ', d["model_classifier"].history.history['accuracy'])

                d["model_weights"] = d["model_classifier"].get_weights()  # 更新保存的模型参数
                print("model {0} done private training. \n".format(index))
            # END FOR LOOP

        # END WHILE LOOP
        return collaboration_performance
