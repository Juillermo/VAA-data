import pickle

import numpy as np
from sklearn.metrics import f1_score, precision_score
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from data import RANDOM_STATE, QUESTIONS
from utils import unfold_matrix, PLOTS_PATH, plot_confusion

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
                # TODO: Not working properly, problem with No Opinions
                d_init = np.zeros((self.N, 13))
                for i in range(self.N):
                    d_init[i] = [1,
                                 0.5, 0.625,
                                 0, 0.25, 0.5,
                                 -0.5, -0.125,
                                 -1,
                                 0, 0, 0, 0]  # Hybrid model, NOs = 0
                d_init = d_init
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

        acc, f1 = self.get_accuracy(data_obj, "test")
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

        #for k in range(len(P)):
            #party_rank = ranks[:, k] != 0
            #plt.hist(ranks[party_rank, k])
            #plt.show()

        return ranks

    def get_full_d(self):
        return np.array([unfold_matrix(el) for el in self.d.get_value()])
