import pickle
from datetime import datetime
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score
import theano
import theano.tensor as T

from data import RANDOM_STATE, DataHolder
from utils import unfold_matrix

MODELS_PATH = "models/"
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")


class DistanceMatrices:
    proximity = np.array([[1, 0.5, 0, -0.5, -1, 0],
                          [0.5, 1, 0.5, 0, -0.5, 0],
                          [0, 0.5, 1, 0.5, 0, 0],
                          [-0.5, 0, 0.5, 1, 0.5, 0],
                          [-1, -0.5, 0, 0.5, 1, 0],
                          [0, 0, 0, 0, 0, 0]])  # Proximity model, NOs = 0
    intensity = np.array([[1, 0.5, 0, -0.5, -1, 0],
                          [0.5, 0.25, 0, -0.25, -0.5, 0],
                          [0, 0, 0, 0, 0, 0],
                          [-0.5, -0.25, 0, 0.25, 0.5, 0],
                          [-1, -0.5, 0, 0.5, 1, 0],
                          [0, 0, 0, 0, 0, 0]])

    def __init__(self, type_of_matrix, N):
        self.full_d = np.zeros((N, 6, 6))

        if type_of_matrix == "Hybrid":
            self.full_d[0] = [[1, 0.5, 0, -0.5, -1, 0],
                              [0.5, 0.625, 0.25, -0.125, -0.5, 0],
                              [0, 0.25, 0.5, 0.25, 0, 0],
                              [-0.5, -0.125, 0.25, 0.625, 0.5, 0],
                              [-1, -0.5, 0, 0.5, 1, 0],
                              [0, 0, 0, 0, 0, 0]]  # Hybrid model, NOs = 0
            print("Weights of the model by [Mendez 2017] hybrid matrix:")
        elif type_of_matrix == "Manhattan":
            self.full_d[0] = DistanceMatrices.proximity
            print("Weights of the model by [Mendez 2017] proximity matrix:")
        elif type_of_matrix == "aquienvoto":
            self.full_d[0] = np.array([[1, 0.5, 0, -0.5, -1, 0],
                                       [1 / 6, 1, 1 / 6, -1 / 6, -1, 0],
                                       [-1, 0, 1, 0, -1, 0],
                                       [-1, -1 / 6, 1 / 6, 1, 1 / 6, 0],
                                       [-1, -0.5, 0, 0.5, 1, 0],
                                       [0, 0, 0, 0, 0, 0]]).T
            print("Weights of the model by aquienvoto method:")

        print(self.full_d[0])
        for i in range(1, N):
            self.full_d[i] = self.full_d[0]


def load_models(data_obj, file_name, with_weights, full_model=True):
    with open(MODELS_PATH + "models-" + file_name + ".pkl", 'rb') as f:
        models_data = pickle.load(f)
    print("Weights of the model loaded from file " + file_name)

    models = []
    for model_data in models_data:
        if file_name[:3] != "svm":
            if full_model:
                models.append(Model(data_obj, with_weights, file_name, params_ini=model_data))
            else:
                models.append(models_data[1:])
        else:
            models.append(SVM(data_obj, params_ini=model_data))

    return models


class Model:
    def __init__(self, dataobj, with_weights, file_name=None, with_importance=False, params_ini=None, with_no=False,
                 with_distance=True):
        self.N = len(dataobj.Q)
        self.dataobj = dataobj
        self.train_params = None
        self.with_weights = with_weights
        self.with_distance = with_distance
        self.with_importance = with_importance
        self.with_no = with_no

        # Tensor variables
        self.u = None  # user profiles tensor
        self.i = None  # user importance tensor
        self.p = None  # party profiles tensor
        self.d = None  # distance matrices values
        self.w = None  # saliency weights
        self.s = None  # output (post-softmax)

        # Functions
        self.predict = None
        self.get_latent = None

        # Initialize model
        if file_name is not None:
            self.name = file_name
            if file_name in ("Hybrid", "Manhattan", "aquienvoto", "combination"):
                self.with_distance = False
                d = DistanceMatrices(file_name, self.N)
                w_init, d_init = np.ones(self.N), d.full_d
            else:
                if params_ini is None:
                    with open(MODELS_PATH + "model-" + file_name + ".pkl", 'rb') as f:
                        self.train_params, d_init, w_init = pickle.load(f)
                else:
                    self.train_params, d_init, w_init = params_ini
                print("Parameters of the model loaded from file " + file_name)

        else:
            print("Weights of the model randomly initialized")
            w_init = np.random.rand(self.N)
            if with_no:
                d_init = np.random.randn(self.N, 13)  # 13 parameters in the bi-sym distance matrix
            else:
                d_init = np.random.randn(self.N,
                                         9)  # 9 parameters in the bi-sym distance matrix w/o no opinions
                d_init = np.array(
                    [[0 if el < 0 and i in (0, 2, 5) or el > 0 and i in (6, 8) else el for i, el in enumerate(dj)] for
                     dj in d_init])

            self.name = dataobj.country + "-" + dataobj.data_conditions + "-" + TIMESTAMP

        self.build(d_init, w_init)

    def build_d(self, d_init):
        self.d = theano.shared(d_init, name="D")
        if self.with_distance:
            full_d = []
            for j in range(self.N):
                full_dj = self.d[j]
                if self.with_no:
                    full_d0 = T.stack([full_dj[0], full_dj[1], full_dj[3], full_dj[6], full_dj[8], full_dj[9]])
                    full_d1 = T.stack([full_dj[1], full_dj[2], full_dj[4], full_dj[7], full_dj[6], full_dj[10]])
                    full_d2 = T.stack([full_dj[3], full_dj[4], full_dj[5], full_dj[4], full_dj[3], full_dj[11]])
                    full_d3 = T.stack([full_dj[6], full_dj[7], full_dj[4], full_dj[2], full_dj[1], full_dj[10]])
                    full_d4 = T.stack([full_dj[8], full_dj[6], full_dj[3], full_dj[1], full_dj[0], full_dj[9]])
                    full_d5 = T.stack([full_dj[9], full_dj[10], full_dj[11], full_dj[10], full_dj[9], full_dj[12]])
                    full_d.append(T.stack([full_d0, full_d1, full_d2, full_d3, full_d4, full_d5]))
                else:
                    full_d0 = T.stack([full_dj[0], full_dj[1], full_dj[3], full_dj[6], full_dj[8], 0])
                    full_d1 = T.stack([full_dj[1], full_dj[2], full_dj[4], full_dj[7], full_dj[6], 0])
                    full_d2 = T.stack([full_dj[3], full_dj[4], full_dj[5], full_dj[4], full_dj[3], 0])
                    full_d3 = T.stack([full_dj[6], full_dj[7], full_dj[4], full_dj[2], full_dj[1], 0])
                    full_d4 = T.stack([full_dj[8], full_dj[6], full_dj[3], full_dj[1], full_dj[0], 0])
                    full_d5 = T.stack([0, 0, 0, 0, 0, 0])
                    full_d.append(T.stack([full_d0, full_d1, full_d2, full_d3, full_d4, full_d5]))

            return T.stack(full_d)  # (N, L, L)
        else:
            return self.d

    def build(self, d_init, w_init):
        # Symbolic variables
        self.u = T.dtensor3("u")  # (M, N, L)
        self.i = T.dmatrix("i")  # (M, N)
        self.p = T.dtensor3("p")  # (K, N, L)
        self.d = self.build_d(d_init)

        self.w = theano.shared(w_init, name="w")  # (N)

        # Compute distance scores
        s = T.batched_dot(self.u.dimshuffle((1, 0, 2)), self.d)  # (N, M, L)
        q = T.batched_dot(s, self.p.dimshuffle((1, 2, 0)))  # (N, M, K) (order?)
        # s = T.nnet.sigmoid(s)

        if self.with_importance:
            s = (q * self.i.dimshuffle(1, 0, 'x')).sum(0)  # (M, K)
        elif self.with_weights:
            s = T.tensordot(q, self.w, axes=[[0], [0]])  # (M, K)
            # s = T.nnet.sigmoid
        else:
            s = T.tensordot(q, np.ones(self.N), axes=[[0], [0]])  # (M, K)

        # Final outcome
        self.s = T.nnet.softmax(s)  # (M, K)

        inputs = [self.u, self.p]
        if self.with_importance:
            inputs += [self.i]

        self.predict = theano.function(inputs=inputs, outputs=self.s)
        self.get_latent = theano.function(inputs=inputs, outputs=q)

    def train(self, training_steps=1000, lambda0=0.01, mu=1):
        U_train, P_train, V_train, I_train = self.dataobj.get_training_data()
        M = len(U_train)

        print("Building training functions...")
        v = T.dmatrix("v")

        # Error function, cost, gradient, and update rule
        err = T.nnet.categorical_crossentropy(self.s, v)
        if self.with_no:
            raise Exception("No opinions update rule has been ignored!")
        cost = err.mean()
        updates = []
        if self.with_weights:
            cost += lambda0 * (self.w ** 2).sum()
            gw = T.grad(cost, self.w)

            stepW = - mu * gw
            new_w = self.w + T.switch(T.gt(self.w + stepW, 0), stepW, -self.w)
            updates += [(self.w, new_w)]

        if self.with_distance:
            cost += lambda0 * (self.d ** 2).sum()  # T.sum(abs(self.d))#
            gd = T.grad(cost, self.d)

            new_d = self.d

            CACD, stepCACD = new_d[:, 8], - mu * gd[:, 8]
            new_d = T.inc_subtensor(CACD, T.switch(T.lt(CACD + stepCACD, 0), stepCACD, -CACD))
            CAD, stepCAD = new_d[:, 6], - mu * gd[:, 6]
            new_d = T.inc_subtensor(CAD, T.switch(T.lt(CAD + stepCAD, 0), stepCAD, -CAD))

            NN, stepNN = new_d[:, 5], - mu * gd[:, 5]
            new_d = T.inc_subtensor(NN, T.switch(T.gt(NN + stepNN, 0), stepNN, -NN))
            AA, stepAA = new_d[:, 2], - mu * gd[:, 2]
            new_d = T.inc_subtensor(AA, T.switch(T.gt(AA + stepAA, 0), stepAA, -AA))
            CACA, stepCACA = new_d[:, 0], - mu * gd[:, 0]
            new_d = T.inc_subtensor(CACA, T.switch(T.gt(CACA + stepCACA, 0), stepCACA, -CACA))

            rest = (1, 3, 4, 7)
            new_d = T.inc_subtensor(new_d[:, rest], - mu * gd[:, rest])
            updates += [(self.d, new_d)]

        # Compile
        inputs = [self.u, self.p, v]
        if self.with_importance:
            inputs += [self.i]

        train = theano.function(inputs=inputs, outputs=[self.s, err], updates=updates)
        self.get_err = theano.function(inputs=inputs, outputs=err.mean())

        # Train
        sample_every = 5
        err_vec = []
        acc_vec = []
        acc_test_vec = []

        v_max = np.argmax(V_train, axis=1)
        print("Iter  Logloss  Accuracy")
        try:
            inputs = [U_train, P_train, V_train]
            if self.with_importance:
                inputs += [I_train]

            for i in range(training_steps):
                pred, error = train(inputs)

                if not i % sample_every:
                    err_vec.append(error.mean())
                    if self.with_importance:
                        p_max = np.argmax(self.predict(U_train, P_train, I_train), axis=1)
                    else:
                        p_max = np.argmax(self.predict(U_train, P_train), axis=1)
                    acc = sum(v_max == p_max) / float(M)
                    acc_vec.append(acc)

                    # acc_test, _, _, _ = self.get_accuracy(self.dataobj, "test")

                    print("{:4d}, {:.5f}, {:.5f}".format(i, err_vec[i // sample_every], acc))  # , acc_test))
        except KeyboardInterrupt:
            print("Stopping training by user request...")

        # Post-training
        print("Training completed")
        self.train_params = {"training_loss": err_vec, "training_accuracy": acc_vec, "iterations": i,
                             "random_state": RANDOM_STATE}

        acc, f1, _, _ = self.get_accuracy("test")
        print("Performance on test set: accuracy {:.5f}, f1 score {:.5f}".format(acc, f1))

    def get_accuracy(self, split="all"):
        U, P, V, I = self.dataobj.get_data(split)
        M = len(U)
        v_max = np.argmax(V, axis=1)
        if self.with_importance:
            p_max = np.argmax(self.predict(U, P, I), axis=1)
        else:
            p_max = np.argmax(self.predict(U, P), axis=1)
        acc = sum(v_max == p_max) / float(M)

        fvec = f1_score(v_max, p_max, average=None)
        precvec = precision_score(v_max, p_max, average=None)
        f1 = f1_score(v_max, p_max, average='weighted')
        return acc, f1, fvec, precvec

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
    def __init__(self, dataobj=None, params_ini=None):
        self.dataobj = dataobj
        if params_ini is None:
            self.model = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=RANDOM_STATE))
            # self.train(dataobj)
        else:
            self.predictions = params_ini

    def train(self, training_steps=None, lambda0=None):
        U, _, V, _ = self.dataobj.get_training_data()
        U = np.argmax(U, axis=2)
        V = np.argmax(V, axis=1)
        print("Training SVM...")
        self.model.fit(U, V)
        print("SVM traininig complete")

        U, _, _, _ = self.dataobj.get_test_data()
        U = np.argmax(U, axis=2)
        self.predictions = self.model.predict_proba(U)

        self.name = "svm-" + self.dataobj.country + "-" + self.dataobj.data_conditions + "-" + TIMESTAMP

    def get_accuracy(self, split="all"):
        U, _, V, _ = self.dataobj.get_data(split)
        M = len(U)
        # U = np.argmax(U, axis=2)
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

        return confs


if __name__ == "__main__":
    ISSUE_VOTERS, EDUCATION, POLITICALLY_ACTIVE = "issue", "highschool", "very_poli"
    # ISSUE_VOTERS, EDUCATION, POLITICALLY_ACTIVE = "no_issue", "no_education", "no_poli"
    data_conditions = ISSUE_VOTERS + "-" + EDUCATION + "-" + POLITICALLY_ACTIVE

    dataobj = DataHolder()
    # model = SVM()
    # model.train(dataobj)

    model = IntegerProgramming()
    model.train(dataobj)

    # a, f, v, p = model.get_accuracy(dataobj, "test")
    # ranks = model.get_rank_info(dataobj, "test")
