from datetime import datetime
import pickle

import numpy as np
import matplotlib.pyplot as plt

from data import DataHolder, L_SET
from model import Model
from utils import QUESTIONS, plotLLmatrix, unfold_sum_matrix

MODELS_PATH = "models/"


def train_new_model(data_obj):
    U, P, V = data_obj.get_data()
    the_model = Model()

    # Train model
    train_params = the_model.train(U, P, V)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open((MODELS_PATH + "model-" + timestamp + ".npy"), 'wb') as f:
        pickle.dump((train_params, the_model, the_model.d), f, pickle.HIGHEST_PROTOCOL)

    err_vec = train_params['training_loss']
    plt.semilogy(err_vec[0:3])
    return the_model


def print_distance_matrices(data_obj, the_model):
    freq_abs, freq_party = data_obj.get_frequencies()
    full_d = the_model.get_full_d()
    N = len(full_d)

    scores = np.zeros_like(full_d)
    for j in range(N):
        scores[j] = np.multiply(freq_abs[j], full_d[j])

    letters_per_line = 27
    _max_D = np.amax(abs(full_d))
    # _max_s = np.amax(abs(scores))

    for j in range(N):
        fig3, axes = plt.subplots(1, 7, figsize=(20, 3))
        i = 0

        # Distance matrices
        plotLLmatrix(axes[i], full_d[j], vmax=_max_D)
        axes[i].set(title=QUESTIONS[j][0:letters_per_line])
        i += 1

        # Mirrored frequencies (for verifying there was enough training data for the weights)
        plotLLmatrix(axes[i], unfold_sum_matrix(freq_abs[j]), cmap='Blues')
        axes[i].set(title=QUESTIONS[j][letters_per_line:2 * letters_per_line])
        i += 1

        # Absolute frequencies
        plotLLmatrix(axes[i], freq_abs[j], cmap='Blues')
        axes[i].set(title=QUESTIONS[j][2 * letters_per_line:3 * letters_per_line])
        i += 1

        # Middle scores (for verifying which weights are being significative)
        plotLLmatrix(axes[i], np.multiply(full_d[j], freq_abs[j] / np.max(freq_abs[j])), vmax=_max_D)
        axes[i].set(title=QUESTIONS[j][3 * letters_per_line:4 * letters_per_line])
        i += 1

        # Party frequencies
        axes[i].xaxis.set(ticks=range(L), ticklabels=L_SET)
        axes[i].yaxis.set(ticks=range(K), ticklabels=party_info)
        cax = axes[i].imshow(freq_party[j], cmap='Blues', vmin=0)  # , vmax=M)
        i += 1

        # Party frequencies (per-party)
        axes[i].xaxis.set(ticks=range(L), ticklabels=L_SET)
        axes[i].yaxis.set(ticks=range(K), ticklabels=party_info)
        tot_party = np.sum(freq_party[j], axis=1)
        cax = axes[i].imshow(freq_party[j] / tot_party[:, None], cmap='Blues', vmin=0)  # , vmax=M)
        i += 1

        # Party answers
        cax = axes[i].imshow(P[:, j, :], cmap='Blues')
        axes[i].xaxis.set(ticks=range(len(L_SET)), ticklabels=L_SET)
        axes[i].yaxis.set(ticks=range(K), ticklabels=party_info)
        i += 1

        fig3.savefig("pesos_matrices_" + str(j) + ".jpg")


def main():
    data_obj = DataHolder()
    # data_obj.get_random_accuracy()

    # the_model = train_new_model(data_obj)
    # the_model = Model(file_name="without_weights.pickle")
    the_model = Model(file_name="Mendez")

    # the_model.get_confusion_matrix(data_obj)
    # the_model.get_weighted_mean_rank(data_obj)

    print_distance_matrices(data_obj, the_model)

if __name__ == "__main__":
    main()
