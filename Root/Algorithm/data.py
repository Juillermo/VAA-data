import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

L_SET = ["CA", "A", "N", "D", "CD", "NO"]

class DataHolder():
    def __init__(self):
        self.U = None
        self.P = None
        self.V = None
        self.party_names = None

        self.freq_abs = None
        self.freq_party = None

        self.load_data()

    def load_data(self):
        answer_cols = ["Ans_" + str(i + 1) for i in range(30)]
        N = len(answer_cols)  # Number of questions
        L = len(L_SET)  # Number of possible answers

        ## Reading user info
        U = pd.read_csv('data/input_to_algorithm.csv', index_col=0, low_memory=False)
        M = len(U)  # Number of users
        assert U.shape[1], N
        print("Users input size (M x N): ", U.shape)

        ans_enc = OneHotEncoder(sparse=False)
        U = ans_enc.fit_transform(U)
        U = U.reshape(M, N, L)
        print("One-hot (M x N x L):", U.shape, "\n")

        ## Reading party info
        P = pd.read_csv('data/es_party_XYZ.csv', delim_whitespace=True)
        self.party_names = P['Party_ID']
        P = P[answer_cols]
        P[P == 99] = 6
        K = len(P)  # Number of parties
        print("Parties input size (K x N): ", P.shape)

        P = ans_enc.transform(P)
        P = P.reshape(len(P), len(answer_cols), L)
        print("One-hot (K x N x L):", P.shape, "\n")

        ## Reading voting intention
        v_max = pd.read_csv('data/voting_intention.csv', header=None, index_col=0)
        assert len(v_max), M
        print("Voting intention input size (M):", len(v_max))

        party_enc = OneHotEncoder(sparse=False)
        V = party_enc.fit_transform(v_max)
        assert K, V.shape[1]
        print("One-hot (M x K)", V.shape, "\n")

        self.U, self.P, self.V = U, P, V

    def get_frequencies(self):
        if self.freq_abs is None or self.freq_party is None:
            U, P, V = self.U, self.P, self.V
            (M, N, L), K = U.shape, len(P)
            v_max = np.argmax(V, axis=1)

            freq_abs = np.zeros((L, L))
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

    def get_data(self):
        return self.U, self.P, self.V, self.party_names

    def get_random_accuracy(self):
        K = len(self.P)
        u_per_k = sum(self.V)
        u_per_k_norm = u_per_k / sum(u_per_k)

        fig, ax = plt.subplots(figsize=(10, 2))
        ax.bar(range(K), u_per_k)
        ax.set(title="Users per party")
        ax.xaxis.set(ticks=range(K), ticklabels=self.party_names)
        fig.show()

        print("Random_acc:", sum(u_per_k_norm ** 2))
        print("Dumb acc:", max(u_per_k_norm))
