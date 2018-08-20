import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import matplotlib.pyplot as plt

L_SET = ["CA", "A", "N", "D", "CD", "NO"]
RANDOM_STATE = 33
QUESTIONS = ["Spain should drop the Euro as a currency",
             "A single member state should be able to block a treaty change, even if all the other members states agree to it",
             "The right of EU citizens to work in Spain should be restricted",
             "There should be a common EU foreign policy even if this limits the capacity of Spain to act independently",
             "The EU should redistribute resources from richer to poorer EU regions",
             "Overall, EU membership has been a bad thing for Spain",
             "EU treaties should be decided by the Cortes Generales rather than by citizens in a referendum.",
             "To address financial crises, the EU should be able to borrow money just like states can",
             "Free market competition makes the health care system function better",
             "The number of public sector employees should be reduced",
             "The state should intervene as little as possible in the economy",
             "Wealth should be redistributed from the richest people to the poorest",
             "Cutting government spending is a good way to solve the economic crisis",
             "It should be easy for companies to fire people",
             "External loans from institutions such as the IMF are a good solution to crisis situations.",
             "Protecting the environment is more important than fostering economic growth",
             "Immigrants must adapt to the values and culture of Spain",
             "Restrictions on citizen privacy are acceptable in order to combat crime",
             "To maintain public order, governments should be able to restrict demonstrations",
             "Less serious crimes should be punished with community service, not imprisonment",
             "Same sex couples should enjoy the same rights as heterosexual couples to marry",
             "Women should be free to decide on matters of abortion",
             "The recreational use of cannabis should be legal",
             "Islam is a threat to the values of Spain",
             "The government is making unnecessary concessions to ETA",
             "The Church enjoys too many privileges",
             "The process of territorial decentralisation has gone too far in Spain",
             "Citizens should directly elect their candidates through primary elections",
             "The possibility of independence for an Autonomous Community should be recognized by the Constitution",
             "Spain should toughen up its immigration policy"]


class DataHolder:
    def __init__(self):
        self.U = None
        self.P = None
        self.V = None
        self.party_names = None

        self.idx_train = None
        self.idx_test = None

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
            return self.U, self.P, self.V
        else:
            raise ValueError("Split " + split + " not recongnized.")

    def get_training_data(self):
        if self.idx_train is None or self.idx_test is None:
            self.compute_splits()
        return self.U[self.idx_train], self.P, self.V[self.idx_train]

    def get_test_data(self):
        if self.idx_train is None or self.idx_test is None:
            self.compute_splits()
        return self.U[self.idx_test], self.P, self.V[self.idx_test]

    def get_random_accuracy(self):
        K = len(self.P)
        u_per_k = sum(self.V)
        u_per_k_norm = u_per_k / sum(u_per_k)

        print("Users per party: " + str(u_per_k))

        fig, ax = plt.subplots(figsize=(10, 2))
        ax.bar(range(K), u_per_k)
        ax.set(title="Users per party")
        ax.xaxis.set(ticks=range(K), ticklabels=self.party_names)
        fig.show()

        print("Random_acc:", sum(u_per_k_norm ** 2))
        print("Dumb acc:", max(u_per_k_norm))

        ranks = -u_per_k.copy()
        pred = ranks.argsort()
        ranks[pred] = np.arange(len(pred)) + 1  # Due to index mismatch

        print("Random mean rank: ", np.sum(ranks * u_per_k_norm))

    def compute_splits(self):
        folds = list(StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE).split(
            self.U, np.argmax(self.V, axis=1)))

        self.idx_train = folds[0][0]
        self.idx_test = folds[0][1]
        print("Partition of the data: Training size " + str(int(len(self.U) * 0.9)) + ", Test set size " +
              str(int(len(self.U) * 0.1)))

    # data_obj = DataHolder()
    # folds = data_obj.get_splits()


#data_obj = DataHolder()
#data_obj.get_random_accuracy()
