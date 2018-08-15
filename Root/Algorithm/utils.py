import numpy as np

from data import L_SET

PLOTS_PATH = "plots/"

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
             "The possibility of independence for an Autonomous Community should be recognized by the Constitution			",
             "Spain should toughen up its immigration policy"]


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

    return [[D_w[0], D_w[1], D_w[3], D_w[6], D_w[8], D_w[9]],
            [D_w[1], D_w[2], D_w[4], D_w[7], D_w[6], D_w[10]],
            [D_w[3], D_w[4], D_w[5], D_w[4], D_w[3], D_w[11]],
            [D_w[6], D_w[7], D_w[4], D_w[2], D_w[1], D_w[10]],
            [D_w[8], D_w[6], D_w[3], D_w[1], D_w[0], D_w[9]],
            [D_w[9], D_w[10], D_w[11], D_w[10], D_w[9], D_w[12]]]


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


def plotLLmatrix(ax, data, vmax=None, cmap='seismic'):
    L = len(L_SET)
    if vmax is not None:
        cax = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        cax = ax.imshow(data, cmap=cmap)
    ax.xaxis.set(ticks=range(L), ticklabels=L_SET)
    ax.yaxis.set(ticks=range(L), ticklabels=L_SET)
