import numpy as np
import matplotlib.pyplot as plt

PLOTS_PATH = "plots/"
L_SET = ["CA", "A", "N", "D", "CD", "NO"]


def unfold_matrix(D_w):
    '''Gets the 13 independent parameters of the folded matrix and generate the double-symmetry matrix.
    -----------------------------------
    INPUT: a numeric vector of lenght 13.
    The order of the 13 parameters correspond to the following loctions in the matrix:
    0
    1 2
    3 4 5
    6 7
    8
    9 10 11    12'''

    if len(D_w) == 13:
        return np.array([[D_w[0], D_w[1], D_w[3], D_w[6], D_w[8], D_w[9]],
                         [D_w[1], D_w[2], D_w[4], D_w[7], D_w[6], D_w[10]],
                         [D_w[3], D_w[4], D_w[5], D_w[4], D_w[3], D_w[11]],
                         [D_w[6], D_w[7], D_w[4], D_w[2], D_w[1], D_w[10]],
                         [D_w[8], D_w[6], D_w[3], D_w[1], D_w[0], D_w[9]],
                         [D_w[9], D_w[10], D_w[11], D_w[10], D_w[9], D_w[12]]])

    elif len(D_w) == 9:
        return np.array([[D_w[0], D_w[1], D_w[3], D_w[6], D_w[8], 0],
                         [D_w[1], D_w[2], D_w[4], D_w[7], D_w[6], 0],
                         [D_w[3], D_w[4], D_w[5], D_w[4], D_w[3], 0],
                         [D_w[6], D_w[7], D_w[4], D_w[2], D_w[1], 0],
                         [D_w[8], D_w[6], D_w[3], D_w[1], D_w[0], 0],
                         [0, 0, 0, 0, 0, 0]])


def unfold_sum_matrix(mat):
    '''Gets a matrix, adds up all the elements that correspond to double-symmetrical positions and assign the
    result to such positions.'''
    L = 6
    temp_mat = np.copy(mat)

    ## Symmetry in the main diagonal
    for i in range(L):
        for j in range(L):
            if i != j:
                temp_mat[i, j] += mat[j, i]

    temp_mat2 = np.copy(temp_mat)
    ## Symmetry in the second diagonal (without No Opinions)
    for i in range(L):
        for j in range(L):
            if (j != 4 - i) and (j != 5 and i != 5):
                temp_mat2[i, j] += temp_mat[4 - j, 4 - i]
            else:
                temp_mat2[i, j] = temp_mat[i, j]
    return temp_mat2


def plotLLmatrix(ax, data, vmax=None, cmap='bwr', lset=None, vmin=None):
    if lset is None:
        lset = L_SET
    L = len(lset)

    cmap = plt.cm.get_cmap(cmap)
    cmap.set_bad((0, 1, 0, 1))

    if vmax is not None:
        if vmin is not None:
            cax = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            cax = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        cax = ax.imshow(data, cmap=cmap)
    ax.xaxis.set(ticks=range(L), ticklabels=lset)
    ax.yaxis.set(ticks=range(L), ticklabels=lset)
    return cax


def plotKLmatrix(ax, data, party_names, l_set=L_SET):
    K, L = len(party_names), len(l_set)
    ax.xaxis.set(ticks=range(L), ticklabels=l_set)
    ax.yaxis.set(ticks=range(K), ticklabels=party_names)
    cax = ax.imshow(data, vmin=0, cmap='Blues')  # , vmax=M)


def plot_confusion(ax, conf, title, party_names):
    K = len(party_names)
    ax.xaxis.set(ticks=range(K), ticklabels=party_names)  # , ticks_position="top", label_position="top")
    ax.set_xticklabels(party_names, rotation=90)
    ax.yaxis.set(ticks=range(K),
                 ticklabels=party_names)  # , ticks_position="right", label_position="right")
    ax.set(xlabel="Recommendation", ylabel="Voting intention", title=title)
    cax = ax.imshow(100 * conf, cmap='Blues')  # , vmin=0, vmax=100)
    return cax
