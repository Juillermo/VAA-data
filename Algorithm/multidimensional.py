import numpy as np
import matplotlib.pyplot as plt

from Algorithm.data import DataHolder, L_SET
from Algorithm.utils import plotLLmatrix, PLOTS_PATH, plot_confusion
from Algorithm.model import IntegerProgramming


def ideal_matrices():
    L_set = L_SET[:-1]

    fig, axes = plt.subplots(1, 4, figsize=((14, 3)))

    mat1 = IntegerProgramming.proximity_matrix
    mat2 = IntegerProgramming.directionality_matrix
    mat3 = IntegerProgramming.intensity_matrix

    ax = axes[0]
    plotLLmatrix(ax, mat1, lset=L_set)
    ax.set(title="Proximity matrix")

    ax = axes[1]
    plotLLmatrix(ax, mat2, lset=L_set)
    ax.set(title="Pure Directionality matrix")

    ax = axes[2]
    plotLLmatrix(ax, mat3, lset=L_set)
    ax.set(title="Directional Intensity matrix")

    ax = axes[3]
    cax = plotLLmatrix(ax, 0.5 * mat1 + 0.25 * mat2 + 0.25 * mat3, lset=L_set)
    ax.set(title="Unified model")
    fig.colorbar(cax)

    plt.tight_layout()
    fig.savefig(PLOTS_PATH + "ideal_matrices.eps")
    plt.show()


if __name__ == "__main__":
    ideal_matrices()
