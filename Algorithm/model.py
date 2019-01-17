import pickle
import time

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from Algorithm.data import RANDOM_STATE, QUESTIONS, DataHolder, L_SET
from Algorithm.utils import unfold_matrix, PLOTS_PATH, plot_confusion

MODELS_PATH = "models/"


class Model:
    def __init__(self, n_questions=len(QUESTIONS), file_name=None):
        self.N = n_questions
        self.train_params = None

        # Tensor variables
        self.u = None  # user profiles tensor
        self.p = None  # party profiles tensor
        self.d = None  # distance matrices values
        self.s = None  # output (post-softmax)

        # Functions
        self.predict = None
        self.get_latent = None

        # Initialize model
        if file_name is not None:
            if file_name == "Mendez":
                self.name = "Mendez"
                d_init = np.zeros((self.N, 13))
                for i in range(self.N):
                    d_init[i] = [1,
                                 0.5, 0.625,
                                 0, 0.25, 0.5,
                                 -0.5, -0.125,
                                 -1,
                                 0, 0, 0, 0]  # Hybrid model, NOs = 0
                print("Weights of the model by Mendez's hybrid matrix:")
                print(np.array(unfold_matrix(d_init[0])))
            else:
                self.name = file_name
                with open(MODELS_PATH + "model-" + file_name + ".pkl", 'rb') as f:
                    self.train_params, d_init = pickle.load(f)
                print("Weights of the model loaded from file " + file_name)
        else:
            self.name = "new"
            print("Weights of the model randomly initialized")
            d_init = np.random.randn(self.N, 13)  # 13 independent weights in the bi-symmetrical distance matrix

        self.build(d_init)

    def build(self, d_init):
        # Symbolic variables
        self.u = T.dtensor3("u")
        self.p = T.dtensor3("p")

        # Define learnable parameters
        self.d = theano.shared(d_init, name="D")  # 13 independent weights in the bi-symmetrical distance matrix

        full_d = []
        for j in range(self.N):
            full_dj = self.d[j]
            full_d0 = T.stack([full_dj[0], full_dj[1], full_dj[3], full_dj[6], full_dj[8], full_dj[9]])
            full_d1 = T.stack([full_dj[1], full_dj[2], full_dj[4], full_dj[7], full_dj[6], full_dj[10]])
            full_d2 = T.stack([full_dj[3], full_dj[4], full_dj[5], full_dj[4], full_dj[3], full_dj[11]])
            full_d3 = T.stack([full_dj[6], full_dj[7], full_dj[4], full_dj[2], full_dj[1], full_dj[10]])
            full_d4 = T.stack([full_dj[8], full_dj[6], full_dj[3], full_dj[1], full_dj[0], full_dj[9]])
            full_d5 = T.stack([full_dj[9], full_dj[10], full_dj[11], full_dj[10], full_dj[9], full_dj[12]])
            full_d.append(T.stack([full_d0, full_d1, full_d2, full_d3, full_d4, full_d5]))

        full_d = T.stack(full_d)
        # w = theano.shared(w_init, name="w")

        # Compute distance scores
        s = T.batched_dot(self.u.dimshuffle((1, 0, 2)), full_d)
        q = T.batched_dot(s, self.p.dimshuffle((1, 2, 0)))
        # s = T.nnet.sigmoid(s)

        # Aggregate without weights
        s = T.tensordot(q, np.ones(self.N), axes=[[0], [0]])

        # Aggregate with weights (deprecated)
        # s = T.tensordot(sd, w, axes=[[0],[0]])
        # s = T.nnet.sigmoid(s)

        # Final outcome
        self.s = T.nnet.softmax(s)

        self.predict = theano.function(inputs=[self.u, self.p], outputs=self.s)
        self.get_latent = theano.function(inputs=[self.u, self.p], outputs=q)

    def train(self, data_obj):
        U_train, P_train, V_train = data_obj.get_training_data()
        M = len(U_train)

        # Training parameters
        training_steps = 4000
        lambda0 = 0.01  # regularization parameter
        mu = 1  # learning rate

        print("Building training functions...")
        v = T.dmatrix("v")
        # Error function, cost and gradient
        err = T.nnet.categorical_crossentropy(self.s, v)
        # cost = err.mean() + lambda0 * ((w ** 2).sum() + (d ** 2).sum())
        cost = err.mean() + lambda0 * (self.d ** 2).sum()  # T.sum(abs(self.d))#
        # gw, gd = T.grad(cost, [w, d])
        gd = T.grad(cost, self.d)

        # Compile
        train = theano.function(
            inputs=[self.u, self.p, v],
            outputs=[self.s, err],
            updates=[(self.d, self.d - mu * gd)])
        # updates=((w, w - mu * gw), (d, d - mu * gd)))

        # Train
        sample_every = 5
        err_vec = []
        acc_vec = []

        v_max = np.argmax(V_train, axis=1)
        print("Iter  Logloss  Accuracy")
        try:
            for i in range(training_steps):
                pred, error = train(U_train, P_train, V_train)
                if not i % sample_every:
                    err_vec.append(error.mean())

                    p_max = np.argmax(self.predict(U_train, P_train), axis=1)
                    acc = sum(v_max == p_max) / float(M)

                    acc_vec.append(acc)

                    print("{:4d}, {:.5f}, {:.5f}".format(i, err_vec[i // sample_every], acc))
        except KeyboardInterrupt:
            print("Stopping training by user request...")

        print("Training completed")

        self.train_params = {"training_loss": err_vec, "training_accuracy": acc_vec, "iterations": i,
                             "random_state": RANDOM_STATE}

        acc, f1, _, _ = self.get_accuracy(data_obj, "test")
        print("Performance on test set: accuracy {:.5f}, f1 score {:.5f}".format(acc, f1))

    def get_accuracy(self, data_obj, split="all"):
        U, P, V = data_obj.get_data(split)
        M = len(U)
        v_max = np.argmax(V, axis=1)
        p_max = np.argmax(self.predict(U, P), axis=1)
        acc = sum(v_max == p_max) / float(M)

        fvec = f1_score(v_max, p_max, average=None)
        precvec = precision_score(v_max, p_max, average=None)
        f1 = f1_score(v_max, p_max, average='weighted')
        return acc, f1, fvec, precvec

    def get_confusion_matrices(self, data_obj, split="all"):
        U, P, V = data_obj.get_data(split)
        party_names = data_obj.party_names
        M, K = len(U), len(P)

        v_max = np.argmax(V, axis=1)
        p_max = np.argmax(self.predict(U, P), axis=1)
        print("Accuracy({:s}): {:.5f}".format(split, sum(v_max == p_max) / float(M)))

        conf = np.zeros((K, K))
        for i in range(M):
            conf[v_max[i], p_max[i]] += 1

        conf_names = ['Absolute values', 'Percentages by row (recall)', 'Percentages by column (precision)',
                      'Merge (f-score)']
        confs = []
        for _ in conf_names:
            confs.append(conf.copy())

        # fig3, ax3 = plt.subplots(1, len(conf_names), figsize=(20, 5))
        for i, el in enumerate(conf_names):
            conf = confs[i]
            for k in range(K):
                if i == 1:
                    conf[k, :] = conf[k, :] / sum(conf[k, :])
                elif i == 2:
                    conf[:, k] = conf[:, k] / sum(conf[:, k])
                elif i == 3:
                    conf = 2 * confs[1] * confs[2] / (confs[1] + confs[2])

            # plot_confusion(ax3[i], conf, title=el, party_names=party_names)

        # fig3.savefig(PLOTS_PATH + self.name + "_confusion_matrices.eps")
        # fig3.show()
        return confs

    def get_rank_info(self, data_obj, split="all"):
        U, P, V = data_obj.get_data(split=split)

        all_ranks = -self.predict(U, P)
        for i, pred in enumerate(all_ranks):
            pred = pred.argsort()
            all_ranks[i, pred] = np.arange(len(pred)) + 1  # Due to index mismatch

        ranks = all_ranks * V

        mean_rank = np.sum(ranks) / len(ranks)
        print("Mean rank ({:s}): {:.5f}".format(split, mean_rank))

        # for k in range(len(P)):
        # party_rank = ranks[:, k] != 0
        # plt.hist(ranks[party_rank, k])
        # plt.show()

        return ranks

    def get_full_d(self):
        return np.array([unfold_matrix(el) for el in self.d.get_value()])


class SVM:
    def __init__(self, dataobj=None, file_name=None):
        if file_name is None:
            self.model = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=RANDOM_STATE))
            self.train(dataobj)
        else:
            self.predictions = np.load(MODELS_PATH + file_name + ".npy")

    def train(self, data_obj):
        U, _, V = data_obj.get_training_data()
        U = np.argmax(U, axis=2)
        V = np.argmax(V, axis=1)
        print("Training SVM...")
        self.model.fit(U, V)
        print("SVM traininig complete")

        U, _, _ = dataobj.get_test_data()
        U = np.argmax(U, axis=2)
        self.predictions = self.model.predict_proba(U)

    def get_accuracy(self, data_obj, split="all"):
        U, _, V = data_obj.get_data(split)
        M = len(U)
        U = np.argmax(U, axis=2)
        v_max = np.argmax(V, axis=1)
        p_max = np.argmax(self.predictions, axis=1)
        acc = sum(v_max == p_max) / float(M)

        fvec = f1_score(v_max, p_max, average=None)
        precvec = precision_score(v_max, p_max, average=None)
        f1 = f1_score(v_max, p_max, average='weighted')
        return acc, f1, fvec, precvec

    def get_rank_info(self, data_obj, split="all"):
        U, _, V = data_obj.get_data(split=split)
        U = np.argmax(U, axis=2)

        all_ranks = -self.predict(U, None)
        for i, pred in enumerate(all_ranks):
            pred = pred.argsort()
            all_ranks[i, pred] = np.arange(len(pred)) + 1  # Due to index mismatch

        ranks = all_ranks * V

        mean_rank = np.sum(ranks) / len(ranks)
        print("Mean rank ({:s}): {:.5f}".format(split, mean_rank))

        # for k in range(len(P)):
        # party_rank = ranks[:, k] != 0
        # plt.hist(ranks[party_rank, k])
        # plt.show()

        return ranks

    def get_confusion_matrices(self, data_obj, split="all"):
        U, _, V = data_obj.get_data(split=split)
        M, K = len(U), len(data_obj.party_names)
        v_max = np.argmax(V, axis=1)
        p_max = np.argmax(self.predictions, axis=1)
        print("Accuracy({:s}): {:.5f}".format(split, sum(v_max == p_max) / float(M)))

        conf = np.zeros((K, K))
        for i in range(M):
            conf[v_max[i], p_max[i]] += 1

        conf_names = ['Absolute values', 'Percentages by row (recall)', 'Percentages by column (precision)',
                      'Merge (f-score)']
        confs = []
        for _ in conf_names:
            confs.append(conf.copy())

        # fig3, ax3 = plt.subplots(1, len(conf_names), figsize=(20, 5))
        for i, el in enumerate(conf_names):
            conf = confs[i]
            for k in range(K):
                if i == 1:
                    conf[k, :] = conf[k, :] / sum(conf[k, :])
                elif i == 2:
                    conf[:, k] = conf[:, k] / sum(conf[:, k])
                elif i == 3:
                    conf = 2 * confs[1] * confs[2] / (confs[1] + confs[2])

            # plot_confusion(ax3[i], conf, title=el, party_names=party_names)

        # fig3.savefig(PLOTS_PATH + self.name + "_confusion_matrices.eps")
        # fig3.show()
        return confs


class IntegerProgramming:
    proximity_matrix = \
        np.array([[1, 0.75, 0.25, -0.25, -1, 0],
                  [0.75, 1, 0.75, 0.25, -0.25, 0],
                  [0.25, 0.75, 1, 0.75, 0.25, 0],
                  [-0.25, 0.25, 0.75, 1, 0.75, 0],
                  [-1, -0.25, 0.25, 0.75, 1, 0],
                  [0, 0, 0, 0, 0, 0]])

    directionality_matrix = \
        np.array([[1, 1, 0, -1, -1, 0],
                  [1, 1, 0, -1, -1, 0],
                  [0, 0, 0, 0, 0, 0],
                  [-1, -1, 0, 1, 1, 0],
                  [-1, -1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0]])

    intensity_matrix = \
        np.array([[1, 0.5, 0, -0.5, -1, 0],
                  [0.5, 0.25, 0, -0.25, -0.5, 0],
                  [0, 0, 0, 0, 0, 0],
                  [-0.5, -0.25, 0, 0.25, 0.5, 0],
                  [-1, -0.5, 0, 0.5, 1, 0],
                  [0, 0, 0, 0, 0, 0]])

    def __init__(self, training_steps, n_questions=len(QUESTIONS), file_name=None):
        self.training_steps = training_steps
        self.N = n_questions
        self.train_params = None

        # Tensor variables
        self.u = None  # user profiles tensor
        self.p = None  # party profiles tensor
        self.d = None  # distance matrices values
        self.s = None  # output (post-softmax)

        # Functions
        self.predict = None
        self.get_latent = None

        # Initialize model
        self.name = "Integer Programming"
        if file_name is not None:
            # self.name = file_name
            # with open(MODELS_PATH + "model-" + file_name + ".pkl", 'rb') as f:
            #    self.train_params, d_init = pickle.load(f)
            # print("Weights of the model loaded from file " + file_name)
            raise Exception("Reading from stored model still not implemented")
        else:
            self.name = "new"
            print("Weights of the model randomly initialized")
            w_init = np.random.randn(self.N)
            w_init[w_init < 0] = -w_init[w_init < 0]

        self.build(w_init)

    def build(self, w_init):
        # Symbolic variables
        u = T.dtensor3("u")
        p = T.dtensor3("p")
        d = T.dtensor3("d")

        self.w = theano.shared(w_init, name="w")

        # Compute distance scores
        s = T.batched_dot(u.dimshuffle((1, 0, 2)), d)
        q = T.batched_dot(s, p.dimshuffle((1, 2, 0)))
        # s = T.nnet.sigmoid(s)

        # Aggregate with weights
        s = T.tensordot(q, self.w, axes=[[0], [0]])
        # s = T.nnet.sigmoid(s)

        # Final outcome
        s = T.nnet.softmax(s)

        # Training parameters
        lambda0 = 0.01  # regularization parameter
        mu = 1  # learning rate

        print("Building training functions...")
        v = T.dmatrix("v")
        # Error function, cost and gradient
        err = T.nnet.categorical_crossentropy(s, v)
        cost = err.mean() + lambda0 * (self.w ** 2).sum()  # T.sum(abs(self.w))
        gd = T.grad(cost, self.w)

        # Compile
        self.train_func = theano.function(
            inputs=[u, p, v, d],
            outputs=[s, err.mean()],
            updates=[(self.w, self.w - T.switch(T.gt(self.w - mu * gd, 0), mu * gd, 0))])
        self.get_err = theano.function(
            inputs=[u, p, v, d],
            outputs=err.mean())

        self.predict = theano.function(inputs=[u, p, d], outputs=s)
        self.get_latent = theano.function(inputs=[u, p, d], outputs=q)

    def train(self, U_train, P_train, V_train, matrix_combination):
        for i in range(self.training_steps):
            pred, error = self.train_func(U_train, P_train, V_train, self.get_full_d(matrix_combination))
        return error

    def get_accuracy(self, data_obj, matrix_combination, weights, split="all"):
        U, P, V = data_obj.get_data(split)
        self.w.set_value(weights)

        M = len(U)
        v_max = np.argmax(V, axis=1)
        p_max = np.argmax(self.predict(U, P, self.get_full_d(matrix_combination)), axis=1)
        acc = sum(v_max == p_max) / float(M)

        fvec = f1_score(v_max, p_max, average=None)
        precvec = precision_score(v_max, p_max, average=None)
        f1 = f1_score(v_max, p_max, average='weighted')
        return acc, f1, fvec, precvec

    def get_confusion_matrices(self, data_obj, split="all"):
        raise Exception("Not implemented yet")
        U, P, V = data_obj.get_data(split)
        party_names = data_obj.party_names
        M, K = len(U), len(P)

        v_max = np.argmax(V, axis=1)
        p_max = np.argmax(self.predict(U, P), axis=1)
        print("Accuracy({:s}): {:.5f}".format(split, sum(v_max == p_max) / float(M)))

        conf = np.zeros((K, K))
        for i in range(M):
            conf[v_max[i], p_max[i]] += 1

        conf_names = ['Absolute values', 'Percentages by row (recall)', 'Percentages by column (precision)',
                      'Merge (f-score)']
        confs = []
        for _ in conf_names:
            confs.append(conf.copy())

        # fig3, ax3 = plt.subplots(1, len(conf_names), figsize=(20, 5))
        for i, el in enumerate(conf_names):
            conf = confs[i]
            for k in range(K):
                if i == 1:
                    conf[k, :] = conf[k, :] / sum(conf[k, :])
                elif i == 2:
                    conf[:, k] = conf[:, k] / sum(conf[:, k])
                elif i == 3:
                    conf = 2 * confs[1] * confs[2] / (confs[1] + confs[2])

            # plot_confusion(ax3[i], conf, title=el, party_names=party_names)

        # fig3.savefig(PLOTS_PATH + self.name + "_confusion_matrices.eps")
        # fig3.show()
        return confs

    def get_rank_info(self, data_obj, split="all"):
        raise Exception("Not implemented yet")
        U, P, V = data_obj.get_data(split=split)

        all_ranks = -self.predict(U, P)
        for i, pred in enumerate(all_ranks):
            pred = pred.argsort()
            all_ranks[i, pred] = np.arange(len(pred)) + 1  # Due to index mismatch

        ranks = all_ranks * V

        mean_rank = np.sum(ranks) / len(ranks)
        print("Mean rank ({:s}): {:.5f}".format(split, mean_rank))

        # for k in range(len(P)):
        # party_rank = ranks[:, k] != 0
        # plt.hist(ranks[party_rank, k])
        # plt.show()

        return ranks

    @staticmethod
    def get_full_d(matrix_combination):
        L = len(IntegerProgramming.intensity_matrix)
        distance_matrices = np.zeros((len(matrix_combination), L, L))

        for i, choice in enumerate(matrix_combination):
            if choice:
                distance_matrices[i, ...] = IntegerProgramming.intensity_matrix
            else:
                distance_matrices[i, ...] = IntegerProgramming.proximity_matrix
        return distance_matrices


if __name__ == "__main__":
    dataobj = DataHolder()
    # model = SVM()
    # model.train(dataobj)

    model = IntegerProgramming()
    model.train(dataobj)

    # a, f, v, p = model.get_accuracy(dataobj, "test")
    # ranks = model.get_rank_info(dataobj, "test")
