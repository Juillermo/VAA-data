import pickle

import numpy as np
import matplotlib.pyplot as plt

from data import DataHolder, RANDOM_STATE
from model import Model, MODELS_PATH, SVM
from results_paper import print_weights, print_distance_matrices


def train_new_model(data_obj, with_weights, n_of_models, training_steps, svm=False, with_importance=False):
    LAMBDA = 0.005

    models = []
    accuracies, fscores = np.zeros(n_of_models), np.zeros(n_of_models)

    for k in range(n_of_models):  # Cross-validation
        data_obj.compute_splits(RANDOM_STATE + k)

        if not svm:
            the_model = Model(data_obj, with_weights=with_weights, with_importance=with_importance)
        else:
            the_model = SVM(data_obj)

        print("\nTraining model {:d} ({:s})".format(k, data_obj.country))
        print("===============================")

        the_model.train(training_steps=training_steps, lambda0=LAMBDA)

        models.append(the_model)
        accuracies[k], fscores[k], _, _ = the_model.get_accuracy("test")

    print("Test accuracy (cross-validation): ", np.mean(accuracies), np.std(accuracies), accuracies)
    print("Test fscore (cross-validation): ", np.mean(fscores), np.std(fscores), fscores)

    # Save models
    fname = MODELS_PATH + "models-" + models[-1].name + ".pkl"
    print("Saving model in " + fname)
    if not svm:
        if with_weights:
            models_data = [(model.train_params, model.d.get_value(), model.w.get_value()) for model in models]
        else:
            models_data = [(model.train_params, model.d.get_value()) for model in models]
    else:
        models_data = [model.predictions for model in models]

    with open(fname, 'wb') as f:
        pickle.dump(models_data, f, pickle.HIGHEST_PROTOCOL)

    if not svm:
        # Plot training loss
        err_vec = models[-1].train_params['training_loss']
        plt.semilogy(range(5, training_steps, 5), err_vec[1:])
        plt.show()

    return models


if __name__ == "__main__":
    # Data filters
    data_pars = {"issue_voters": "issue", "education": "highschool", "politically_active": "very_poli"}
    # data_pars = {"issue_voters": "no_issue", "education": "no_education", "politically_active": "no_poli"}
    data_conditions = data_pars["issue_voters"] + "-" + data_pars["education"] + "-" + data_pars[
        "politically_active"]

    COUNTRIES = [
        "England",
        "Portugal",
        "Greece",
        "Italy",
        "Spain",
    ]

    WITH_WEIGHTS = True
    N_OF_MODELS = 10
    TRAINING_STEPS = 300
    USE_SVM = True

    names = []
    for country in COUNTRIES:
        data_obj = DataHolder(country, data_pars)
        models = train_new_model(data_obj, WITH_WEIGHTS, N_OF_MODELS, TRAINING_STEPS, svm=USE_SVM)
        names.append(models[-1].name)

        if not USE_SVM:
            print_distance_matrices(models, pages=1)
            print_weights(models)

    print(names)
