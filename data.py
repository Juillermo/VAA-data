import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import matplotlib.pyplot as plt

from utils import PLOTS_PATH

RANDOM_STATE = 33

L_SET = ["CA", "A", "N", "D", "CD", "NO"]
DATA_ROOT = "data/"


class DataHolder:
    def __init__(self, country, data_pars, test_split=0.1):
        self.U, self.P, self.V = None, None, None
        self.I = None  # User importance, not available for this dataset
        self.party_names = None

        self.country = country
        data_conditions = data_pars["issue_voters"] + "-" + data_pars["education"] + "-" + data_pars[
            "politically_active"]
        self.data_conditions = data_conditions
        with open(DATA_ROOT + "questions.json", "r") as r:
            self.Q = json.load(r)[country]

        self.test_split = test_split
        self.idx_train = None
        self.idx_test = None

        self.freq_abs = None
        self.freq_party = None

        self.load_data()

    def load_data(self):
        datapath = DATA_ROOT + self.country + "/"
        answer_cols = ["Ans_" + str(i + 1) for i in range(len(self.Q))]
        N = len(answer_cols)  # Number of questions
        L = len(L_SET)  # Number of possible answers

        ## Reading user info
        U = pd.read_csv(datapath + 'user_answers-' + self.data_conditions + '.csv', index_col=0, low_memory=False)
        M = len(U)  # Number of users
        assert U.shape[1] == N, str(U.shape[1]) + " != " + str(N) + ", U: " + str(U.shape)
        print("Users input size (M x N): ", U.shape)

        ans_enc = OneHotEncoder(sparse=False)
        U = ans_enc.fit_transform(U)
        U = U.reshape(M, N, L)
        print("One-hot (M x N x L):", U.shape, "\n")

        ## Reading party info
        P = pd.read_csv(datapath + 'party_XYZ.csv', delim_whitespace=True)
        print(P)
        self.party_names = P['Party_ID']
        P = P[answer_cols]
        P[P == 99] = 6
        K = len(P)  # Number of parties
        print("Parties input size (K x N): ", P.shape)
        if np.min(P.values) == 1 and np.max(P.values) == 6:
            P = P - 1

        P = ans_enc.transform(P)
        P = P.reshape(K, N, L)
        print("One-hot (K x N x L):", P.shape, "\n")

        ## Reading voting intention
        v_max = pd.read_csv(datapath + 'voting_intention-' + self.data_conditions + '.csv', header=None, index_col=0)
        assert len(v_max) == M, str(len(v_max)) + " != " + str(M)
        print("Voting intention input size (M):", len(v_max))

        party_enc = OneHotEncoder(sparse=False)
        V = party_enc.fit_transform(v_max)
        assert K == V.shape[1], str(K) + " != " + str(V.shape[1])
        print("One-hot (M x K)", V.shape, "\n")

        self.U, self.P, self.V = U, P, V

    def get_frequencies(self):
        if self.freq_abs is None or self.freq_party is None:
            U, P, V = self.U, self.P, self.V
            (M, N, L), K = U.shape, len(P)
            v_max = np.argmax(V, axis=1)

            freq_abs = np.zeros((N, L, L))
            freq_party = np.zeros((N, K, L))
            for j in range(N):
                for k in range(K):
                    for i in range(M):
                        u_choice = np.argmax(U[i, j])
                        p_choice = np.argmax(P[k, j])

                        freq_abs[j, u_choice, p_choice] += 1

                        if v_max[i] == k:
                            freq_party[j, k, u_choice] += 1

            self.freq_abs, self.freq_party = freq_abs, freq_party
            return freq_abs, freq_party
        else:
            return self.freq_abs, self.freq_party

    def get_data(self, split="all"):
        if split == "train":
            return self.get_training_data()
        elif split == "test":
            return self.get_test_data()
        elif split == "all":
            return self.U, self.P, self.V, self.I
        else:
            raise ValueError("Split " + split + " not recongnized.")

    def get_training_data(self):
        if self.idx_train is None or self.idx_test is None:
            self.compute_splits()
        return self.U[self.idx_train], self.P, self.V[self.idx_train], self.I

    def get_test_data(self):
        if self.idx_train is None or self.idx_test is None:
            self.compute_splits()
        return self.U[self.idx_test], self.P, self.V[self.idx_test], self.I

    def get_random_accuracy(self, fname=None):
        K = len(self.P)
        u_per_k = sum(self.V)
        u_per_k_norm = u_per_k / sum(u_per_k)

        print("Users per party: " + str(u_per_k))

        if fname is None:
            fname = self.country + self.data_conditions
        fig, ax = plt.subplots(figsize=(10, 2))
        rects = ax.bar(range(K), u_per_k)
        # ax.set(title="Users per party")
        ax.xaxis.set(ticks=range(K), ticklabels=self.party_names)

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%d' % int(height),
                    ha='center', va='bottom')
        plt.tight_layout()
        fig.savefig(PLOTS_PATH + "parties-" + fname + ".eps")
        fig.show()

        print("Random_acc:", sum(u_per_k_norm ** 2))
        print("Dumb acc:", max(u_per_k_norm))

        ranks = -u_per_k.copy()
        pred = ranks.argsort()
        ranks[pred] = np.arange(len(pred)) + 1  # Due to index mismatch

        print("Random mean rank: ", np.sum(ranks * u_per_k_norm))

        return fig

    def compute_splits(self, random_state=RANDOM_STATE):
        folds = list(StratifiedShuffleSplit(n_splits=1, test_size=self.test_split, random_state=random_state).split(
            self.U, np.argmax(self.V, axis=1)))

        self.idx_train = folds[0][0]
        self.idx_test = folds[0][1]
        print("Partition of the data: Training size " + str(
            int(len(self.U) * (1 - self.test_split))) + ", Test set size " +
              str(int(len(self.U) * self.test_split)))

    def get_P(self):
        return self.P


if __name__ == "__main__":
    data_pars = {"issue_voters": "issue", "education": "highschool", "politically_active": "very_poli"}
    # data_pars = {"issue_voters": "no_issue", "education": "no_education", "politically_active": "no_poli"}

    data_obj = DataHolder(country="Greece", data_pars=data_pars)
    data_obj.get_random_accuracy()
