import numpy as np
import matplotlib.pyplot as plt

from Algorithm.data import DataHolder, L_SET, QUESTIONS
from Algorithm.model import Model, SVM
from Algorithm.utils import plotLLmatrix, PLOTS_PATH, plot_confusion, plotKLmatrix
from Algorithm.algorithm import print_weights


def plot_dist_matrices_Mendez():
    L_set = L_SET[:-1]
    cmap = 'bwr'

    fig, ax = plt.subplots(1, 3, figsize=((10.5, 3)))

    mat = [[1, 0.5, 0, -0.5, -1],
           [0.5, 1, 0.5, 0, -0.5],
           [0, 0.5, 1, 0.5, 0],
           [-0.5, 0, 0.5, 1, 0.5],
           [-1, -0.5, 0, 0.5, 1]]

    ax3 = ax[0]
    plotLLmatrix(ax3, mat, lset=L_set)
    ax3.set(title="Proximity matrix")

    mat = [[1, 0.5, 0, -0.5, -1],
           [0.5, 0.25, 0, -0.25, -0.5],
           [0, 0, 0, 0, 0],
           [-0.5, -0.25, 0, 0.25, 0.5],
           [-1, -0.5, 0, 0.5, 1]]

    ax3 = ax[1]
    plotLLmatrix(ax3, mat, lset=L_set)
    ax3.set(title="Directionality matrix")

    mat = [[1, 0.5, 0, -0.5, -1],
           [0.5, 0.625, 0.25, -0.125, -0.5],
           [0, 0.25, 0.5, 0.25, 0],
           [-0.5, -0.125, 0.25, 0.625, 0.5],
           [-1, -0.5, 0, 0.5, 1]]

    ax3 = ax[2]
    cax = plotLLmatrix(ax3, mat, lset=L_set)
    ax3.set(title="Hybrid matrix")
    fig.colorbar(cax)

    plt.tight_layout()
    # fig.savefig(PLOTS_PATH + "prox_dir_hyb.eps")
    plt.show()


def plot_comparison_confmats(data_obj, mendez_model, social_model, our_model):
    confm = mendez_model.get_confusion_matrices(data_obj, "test")
    confs = social_model.get_confusion_matrices(data_obj, "test")
    confo = our_model.get_confusion_matrices(data_obj, "test")

    fig, ax = plt.subplots(1, 3, figsize=(15 / 4 * 3, 4))
    plot_confusion(ax[0], confm[1], "Deductive VAAs", data_obj.party_names)
    plot_confusion(ax[1], confs[1], "Social VAAs", data_obj.party_names)
    cax = plot_confusion(ax[2], confo[1], "Learning VAAs", data_obj.party_names)
    plt.colorbar(cax)

    plt.tight_layout()
    fig.savefig(PLOTS_PATH + "comparison_confusion_matrices_3.eps")
    fig.show()


def plot_party_fscores(data_obj, models):
    ind = np.arange(len(data_obj.party_names))  # the x locations for the groups
    width = 0.25  # the width of the bars
    fig, ax = plt.subplots(figsize=(8, 2))

    for i, model in enumerate(models):
        print(model)
        acc, f1, fvec, _ = model[0].get_accuracy(data_obj, "test")
        print(acc, f1)
        ax.bar(ind + i * width, fvec, width, label=model[1])

    ax.set_xticks(ind + (len(models) - 1) * width / 2)
    ax.set_xticklabels(data_obj.party_names)
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_PATH + "fscores.eps")
    fig.show()

    # fig, ax = plt.subplots(figsize=(8, 1.8))
    # ax.bar(ind, pm, width, label="Current VAAs")
    # ax.bar(ind + width, p, width, label="Learning VAA")
    # ax.set_xticks(ind + width / 2)
    # ax.set_xticklabels(data_obj.party_names)
    # ax.legend()
    # plt.tight_layout()
    # fig.savefig(PLOTS_PATH + "pscores.eps")
    # fig.show()


def plot_five_distance_matrices(our_model):
    full_d = our_model.get_full_d()
    N = len(full_d)

    # _max_D = np.amax(abs(full_d))

    fig, ax = plt.subplots(1, 5, figsize=(12, 2.5))
    titles = ["Proximity-like", "Directionality-like", "Hybrid-like", "Other possible paradigms", "Rather nonsense"]
    for i, j in enumerate([0, 8, 18, 28, 20]):
        _max_D = np.amax(abs(full_d[j]))
        plotLLmatrix(ax[i], full_d[j], vmax=_max_D)
        ax[i].set(title=titles[i])

    # plt.colorbar(cax)

    plt.tight_layout()
    # fig.savefig(PLOTS_PATH + "sample_distances.png")
    fig.show()


def print_all_distance_matrices(data_obj, the_model):
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

    pages = 5
    for page in range(pages):
        i = 1
        fig3, big_axes = plt.subplots(figsize=(11.0, 2.7 * N / pages), nrows=int(N / pages), ncols=1)  # , sharey=True)
        for num, big_ax in enumerate(big_axes):
            j = int(page * N / pages + num)
            big_ax.set_title(str(j + 1) + ": " + QUESTIONS[j])
            big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
            big_ax._frameon = False

            # Distance matrices
            axes = fig3.add_subplot(int(N / pages), 5, i)
            plotLLmatrix(axes, full_d[j], vmax=_max_D)
            i += 1

            # Absolute frequencies
            axes = fig3.add_subplot(int(N / pages), 5, i)
            plotLLmatrix(axes, freq_abs[j], cmap='Blues')
            i += 1

            # Party frequencies
            axes = fig3.add_subplot(int(N / pages), 5, i)
            plotKLmatrix(axes, freq_party[j], party_names)
            i += 1

            # Party frequencies (per-party normalization)
            tot_party = np.sum(freq_party[j], axis=1)
            axes = fig3.add_subplot(int(N / pages), 5, i)
            plotKLmatrix(axes, freq_party[j] / tot_party[:, None], party_names)
            i += 1

            # Party answers
            axes = fig3.add_subplot(int(N / pages), 5, i)
            plotKLmatrix(axes, data_obj.P[:, j, :], party_names)
            i += 1

        fig3.set_facecolor('w')
        plt.tight_layout()
        fig3.savefig(PLOTS_PATH + "distance_matrices_{:d}_{:s}.png".format(page, the_model.name))
        plt.show()


if __name__ == "__main__":
    dataobj = DataHolder()

    mendez_model = Model(file_name="Mendez")
    social_model = SVM(file_name="svm")
    our_model = Model(file_name="20180817-150608")

    plot_dist_matrices_Mendez()
    plot_comparison_confmats(dataobj, mendez_model, social_model, our_model)
    plot_party_fscores(dataobj,
                       ((mendez_model, "Deductive VAAs"), (social_model, "Social VAAs"), (our_model, "Learning VAAs")))
    plot_five_distance_matrices(our_model)
    print_weights(our_model)
    print_all_distance_matrices(dataobj, our_model)
