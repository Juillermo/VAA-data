import pickle

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from utils import unfold_matrix, PLOTS_PATH


class Model:
    def __init__(self, n_questions=30, file_name=None):
        self.N = n_questions

        # Tensor variables
        self.u = None  # user profiles tensor
        self.p = None  # party profiles tensor
        self.d = None  # distance matrices values
        self.s = None  # output (post-softmax)

        # Functions
        self.predict = None
        self.get_latent = None

        if file_name is not None:
            if file_name != "Mendez":
                self.name = "ours"
                with open('data/' + file_name, 'rb') as f:
                    # w_init, d_init = pickle.load(f)
                    d_raw = pickle.load(f, encoding='latin1')
                    d_init = d_raw  # .get_value()
                print("Weights of the model loaded from file " + file_name)
            else:
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
            self.name = None
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

    def train(self, U, P, V):
        M = len(U)

        # Training parameters
        training_steps = 8000
        lambda0 = 0.01  # regularization parameter
        mu = 1  # learning rate

        v = T.dmatrix("v")
        # Error function, cost and gradient
        err = T.nnet.categorical_crossentropy(self.s, v)
        # cost = err.mean() + lambda0 * ((w ** 2).sum() + (d ** 2).sum())
        cost = err.mean() + lambda0 * (self.d ** 2).sum()
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
        try:
            v_max = np.argmax(V, axis=1)

            for i in range(training_steps):
                pred, error = train(U, P, V)
                if not i % sample_every:
                    err_vec.append(error.mean())

                    p_max = np.argmax(self.predict(U, P), axis=1)
                    acc = sum(v_max == p_max) / float(M)

                    acc_vec.append(acc)

                    print("{:d}, {:.5f}, {:.5f}".format(i, err_vec[i], acc))

        finally:

            print("Final model:")
            # print(w.get_value())
            print(self.d.get_value())
            print("target values for D:")
            print(V)
            print("prediction on D:")
            print(self.predict(U, P))

            return {"training_loss": err_vec, "training_accuracy": acc_vec, "iterations": i}

    def get_confusion_matrix(self, data_obj):
        U, P, V, party_names = data_obj.get_data()
        M, K = len(U), len(P)

        v_max = np.argmax(V, axis=1)
        p_max = np.argmax(self.predict(U, P), axis=1)
        acc = sum(v_max == p_max) / float(M)
        print("Accuracy:", acc)

        conf = np.zeros((K, K))
        for i in range(M):
            conf[v_max[i], p_max[i]] += 1

        conf_names = ['Absolute values', 'Percentages by row (recall)', 'Percentages by column (precision)',
                      'Merge (f-score)']
        confs = []
        for _ in conf_names:
            confs.append(conf.copy())

        fig3, ax3 = plt.subplots(1, len(conf_names), figsize=(20, 5))
        for i, el in enumerate(conf_names):
            conf = confs[i]
            for k in range(K):
                if i == 1:
                    conf[k, :] = conf[k, :] / sum(conf[k, :])
                elif i == 2:
                    conf[:, k] = conf[:, k] / sum(conf[:, k])
                elif i == 3:
                    conf = 2 * confs[1] * confs[2] / (confs[1] + confs[2])

            ax3[i].xaxis.set(ticks=range(K), ticklabels=party_names)  # , ticks_position="top", label_position="top")
            ax3[i].set_xticklabels(party_names, rotation=90)
            ax3[i].yaxis.set(ticks=range(K),
                             ticklabels=party_names)  # , ticks_position="right", label_position="right")
            ax3[i].set(xlabel="Max recommendation", ylabel="Voting intention", title=el)
            cax = ax3[i].imshow(100 * conf, cmap='Blues')  # , vmin=0, vmax=100)

        fig3.savefig(PLOTS_PATH + self.name + "_confusion_matrices.eps")
        fig3.show()

    def get_weighted_mean_rank(self, data_obj):
        U, P, V, _ = data_obj.get_data()
        ranks = -self.predict(U, P)
        for i, pred in enumerate(ranks):
            pred = pred.argsort()
            ranks[i, pred] = np.arange(len(pred))

        mean_rank = np.sum(ranks * V) / len(ranks) + 1
        print(mean_rank)
        return mean_rank

    def get_full_d(self):
        return np.array([unfold_matrix(el) for el in self.d.get_value()])
