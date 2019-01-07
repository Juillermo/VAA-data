from datetime import datetime
import pickle

import numpy as np
import matplotlib.pyplot as plt

from Algorithm.data import DataHolder, L_SET, QUESTIONS
from Algorithm.model import Model, MODELS_PATH
from Algorithm.utils import plotLLmatrix, plotKLmatrix, unfold_sum_matrix, PLOTS_PATH, plot_confusion


def train_new_model(data_obj):
    the_model = Model()

    # Train model
    print("Training the model...")
    train_params = the_model.train(data_obj)

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = MODELS_PATH + "model-" + timestamp + ".pkl"
    print("Saving model in " + fname)
    with open(fname, 'wb') as f:
        pickle.dump((the_model.train_params, the_model.d.get_value()), f, pickle.HIGHEST_PROTOCOL)

    err_vec = the_model.train_params['training_loss']
    plt.semilogy(err_vec)
    plt.show()
    return the_model


def print_distance_matrices(data_obj, the_model):
    party_names = data_obj.party_names
    full_d = the_model.get_full_d()
    N = len(full_d)

    print("Obtaining frequencies...")
    freq_abs, freq_party = data_obj.get_frequencies()

    scores = np.zeros_like(full_d)
    for j in range(N):
        scores[j] = np.multiply(freq_abs[j], full_d[j])

    print("Plotting matrices...")
    letters_per_line = 27
    _max_D = np.amax(abs(full_d))
    # _max_s = np.amax(abs(scores))

    fig3, tot_axes = plt.subplots(N, 7, figsize=(15, 2 * N))
    for j in range(N):
        axes = tot_axes[j]
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
        axes[i].set(title=QUESTIONS[j][3 * letters_per_line:4 * letters_per_line] + " " + str(j))
        i += 1

        # Party frequencies
        plotKLmatrix(axes[i], freq_party[j], party_names)
        i += 1

        # Party frequencies (per-party)
        tot_party = np.sum(freq_party[j], axis=1)
        plotKLmatrix(axes[i], freq_party[j] / tot_party[:, None], party_names)
        i += 1

        # Party answers
        plotKLmatrix(axes[i], data_obj.P[:, j, :], party_names)
        i += 1

    plt.tight_layout()
    fig3.savefig(PLOTS_PATH + "pesos_matrices_" + the_model.name + ".png")
    plt.show()


def print_weights(the_model):
    full_d = the_model.get_full_d()
    N = len(full_d)
    # full_d = np.array([unfold_matrix(el) for el in d_init])
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.bar(range(N), [np.linalg.norm(el) for el in full_d])
    # ax.set(title="Relative weights of the issue questions", xlabel="Issue question")
    ax.yaxis.grid()
    ax.xaxis.set(ticks=range(30), ticklabels=range(1, 31))
    plt.tight_layout()
    fig.savefig(PLOTS_PATH + "weights" + the_model.name + ".eps", format="eps")
    plt.show()


def main():
    data_obj = DataHolder()
    fig = data_obj.get_random_accuracy()
    #fig.savefig(PLOTS_PATH + "users_per_party.eps")

    # our_model = train_new_model(data_obj)
    # our_model = Model(file_name="20180817-150608")
    # mendez_model = Model(file_name="Mendez")

    # print(our_model.get_accuracy(data_obj, "test"), mendez_model.get_accuracy(data_obj, "test"))

    # our_model.get_rank_info(data_obj, "test")
    # mendez_model.get_rank_info(data_obj, "test")

    # our_model.get_confusion_matrices(data_obj, "train")

    # print_distance_matrices(data_obj, our_model)
    # print_weights(our_model)


if __name__ == "__main__":
    main()
