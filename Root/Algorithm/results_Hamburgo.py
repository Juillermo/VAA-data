from datetime import datetime
import pickle

import numpy as np
import matplotlib.pyplot as plt

from data import DataHolder, L_SET, QUESTIONS
from model import Model, MODELS_PATH
from utils import plotLLmatrix, plotKLmatrix, unfold_sum_matrix, PLOTS_PATH, plot_confusion


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

    #_max_D = np.amax(abs(full_d))

    fig, ax = plt.subplots(1, 5, figsize=(12, 2.5))
    titles = ["Proximity-like", "Directionality-like", "Hybryd-like", "Other possible paradigms", "Rather nonsense"]
    for i, j in enumerate([0, 8, 18, 28, 20]):
        _max_D = np.amax(abs(full_d[j]))
        plotLLmatrix(ax[i], full_d[j], vmax=_max_D)
        ax[i].set(title=titles[i])

    #plt.colorbar(cax)

    plt.tight_layout()
    fig.savefig(PLOTS_PATH+"sample_distances.png")
    fig.show()


data_obj = DataHolder()
# data_obj.get_random_accuracy()

# our_model = train_new_model(data_obj)
our_model = Model(file_name="20180817-150608")
mendez_model = Model(file_name="Mendez")

# print(our_model.get_accuracy(data_obj, "test"), mendez_model.get_accuracy(data_obj, "test"))

plot_comparison_confmats(data_obj, mendez_model, our_model)

#plot_distance_matrices(our_model)

# our_model.get_rank_info(data_obj, "test")
# mendez_model.get_rank_info(data_obj, "test")

# our_model.get_confusion_matrices(data_obj, "train")

# print_distance_matrices(data_obj, our_model)
# print_weights(our_model)
