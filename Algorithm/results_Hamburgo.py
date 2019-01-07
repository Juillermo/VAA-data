import numpy as np
import matplotlib.pyplot as plt

from Algorithm.data import DataHolder, L_SET
from Algorithm.model import Model
from Algorithm.utils import plotLLmatrix, PLOTS_PATH, plot_confusion
from Algorithm.algorithm import print_weights


def plot_currentVAA_confmat(data_obj, mendez_model):
    K = len(data_obj.party_names)
    confm = mendez_model.get_confusion_matrices(data_obj, "all")

    fig, ax = plt.subplots(figsize=(7, 5))
    cax = plot_confusion(ax, confm[2], "Current VAAs (precision)", data_obj.party_names)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(K):
        for j in range(K):
            text = ax.text(j, i, "{:.0f}".format(100 * confm[2][i, j]), ha="center", va="center",
                           color="w", size=16)

    # fig.colorbar(cax)
    plt.tight_layout()
    fig.savefig(PLOTS_PATH + "confVAA.eps")
    fig.show()


def plot_comparison_confmats(data_obj, mendez_model, our_model):
    confm = mendez_model.get_confusion_matrices(data_obj, "test")
    confo = our_model.get_confusion_matrices(data_obj, "test")

    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    plot_confusion(ax[0], confm[1], "Current VAAs (recall)", data_obj.party_names)
    plot_confusion(ax[1], confm[2], "Current VAAs (precision)", data_obj.party_names)
    plot_confusion(ax[2], confo[1], "Learning VAAs (recall)", data_obj.party_names)
    plot_confusion(ax[3], confo[2], "Learning VAAs (precision)", data_obj.party_names)

    plt.tight_layout()
    fig.savefig(PLOTS_PATH + "comparison_confusion_matrices.png")
    fig.show()


def plot_distance_matrices(our_model):
    full_d = our_model.get_full_d()
    N = len(full_d)

    # _max_D = np.amax(abs(full_d))

    fig, ax = plt.subplots(1, 5, figsize=(12, 2.5))
    titles = ["Proximity-like", "Directionality-like", "Hybryd-like", "Other possible paradigms", "Rather nonsense"]
    for i, j in enumerate([0, 8, 18, 28, 20]):
        _max_D = np.amax(abs(full_d[j]))
        plotLLmatrix(ax[i], full_d[j], vmax=_max_D)
        ax[i].set(title=titles[i])

    # plt.colorbar(cax)

    plt.tight_layout()
    fig.savefig(PLOTS_PATH + "sample_distances.png")
    fig.show()


def plot_party_fscores(our_model, mendez_model, data_obj):
    acc, f1, fvec, p = our_model.get_accuracy(data_obj, "test")
    accm, f1m, fvecm, pm = mendez_model.get_accuracy(data_obj, "test")
    print(acc, f1, accm, f1m)

    ind = np.arange(len(fvec))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 1.8))
    ax.bar(ind, fvecm, width, label="Current VAAs")
    ax.bar(ind + width, fvec, width, label="Learning VAA")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(data_obj.party_names)
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_PATH + "fscores.eps")
    fig.show()

    fig, ax = plt.subplots(figsize=(8, 1.8))
    ax.bar(ind, pm, width, label="Current VAAs")
    ax.bar(ind + width, p, width, label="Learning VAA")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(data_obj.party_names)
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_PATH + "pscores.eps")
    fig.show()


def plot_party_ranks(our_model, mendez_model, data_obj):
    K = len(data_obj.party_names)
    ranks = our_model.get_rank_info(data_obj, "test")
    rankm = mendez_model.get_rank_info(data_obj, "test")

    p_ranks = np.zeros(K)
    p_rankm = np.zeros(K)
    for k in range(K):
        party_rank = ranks[:, k] != 0
        party_rankm = rankm[:, k] != 0

        p_ranks[k] = np.sum(ranks[party_rank, k]) / float(np.sum(party_rank))
        p_rankm[k] = np.sum(rankm[party_rankm, k]) / float(np.sum(party_rankm))

    ind = np.arange(K)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 1.8))
    ax.bar(ind, p_rankm, width, label="Current VAAs")
    ax.bar(ind + width, p_ranks, width, label="Learning VAA")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(data_obj.party_names)
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_PATH + "pscores.eps")
    fig.show()


def plot_Mendez():
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
    fig.savefig(PLOTS_PATH + "prox_dir_hyb.eps")
    plt.show()


if __name__ == "__main__":
    data_obj = DataHolder()
# data_obj.get_random_accuracy()

    our_model = Model(file_name="20180817-150608")
# mendez_model = Model(file_name="Mendez")

# plot_currentVAA_confmat(data_obj, mendez_model)
# plot_comparison_confmats(data_obj, mendez_model, our_model)
# plot_party_fscores(our_model, mendez_model, data_obj)
# plot_distance_matrices(our_model)
# plot_party_ranks(our_model, mendez_model,data_obj)

# our_model.get_rank_info(data_obj, "test")
# mendez_model.get_rank_info(data_obj, "test")

# our_model.get_confusion_matrices(data_obj, "train")
#print_weights(our_model)
    #plot_Mendez()
