from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data import DataHolder, RANDOM_STATE
from model import Model, load_models
from utils import plotLLmatrix, PLOTS_PATH, plot_confusion, plotKLmatrix, unfold_sum_matrix

TAB_DIR = "tables/"


def obtain_performance_table(models, ax):
    data_obj = models["Standard VAA"].dataobj
    C = len(models["Learning VAA"])
    fname = models["Learning VAA"][0].name

    # Initializing plot for parties' f-scores
    ind = np.arange(len(data_obj.party_names))  # the x locations for the groups
    width = 0.25  # the width of the bars

    performances = {}
    for model_name in models.keys():
        performances[model_name] = {"acc": np.zeros(C), "f1": np.zeros(C), "fvec": np.zeros((C, len(ind))),
                                    "mean_acc": np.zeros(2), "mean_f1": np.zeros(2),
                                    "mean_fvec": np.zeros((len(ind), 2))}

    models["Standard VAA"] = [models["Standard VAA"] for el in range(C)]
    for k in range(C):
        data_obj.compute_splits(random_state=RANDOM_STATE + k)
        for model_name, model in models.items():
            mod_inf = performances[model_name]
            mod_inf["acc"][k], mod_inf["f1"][k], mod_inf["fvec"][k], _ = model[k].get_accuracy("test")

    if "Social VAA" in models:
        table_order = ("Standard VAA", "Social VAA", "Learning VAA")
    else:
        table_order = ("Standard VAA", "Learning VAA")

    for i, model_name in enumerate(table_order):
        perf = performances[model_name]
        perf["mean_acc"] = (np.mean(perf["acc"]), np.std(perf["acc"]))
        perf["mean_f1"] = (np.mean(perf["f1"]), np.std(perf["f1"]))
        perf["mean_fvec"] = (np.mean(perf["fvec"], axis=0), np.std(perf["acc"], axis=0))
        ax.bar(ind + i * width, perf["mean_fvec"][0], width, yerr=perf["mean_fvec"][1], label=model_name)

    ax.set_xticks(ind + (len(models) - 1) * width / 2)
    ax.set_xticklabels(data_obj.party_names)
    ax.set(ylabel=data_obj.country, ylim=[0, 1])
    if data_obj.country == "Portugal":
        ax.legend()

    with open(TAB_DIR + "table-" + fname + ".txt", 'w') as f:
        def printout(line):
            print(line)
            f.write(line + "\n")

        print("\nAccuracies and fscores for " + data_obj.country + "\n=================================")
        for model_name, perf in performances.items():
            printout(model_name + " accuracy: {:.6f} +- {:.6f}".format(perf["mean_acc"][0], perf["mean_acc"][1]))
            printout(model_name + " accuracy: {:.6f} +- {:.6f}".format(perf["mean_f1"][0], perf["mean_f1"][1]))
            printout(str(perf["acc"]))
            printout(str(perf["f1"]))

        dec = "{:.3f}"

        string = r"    \multirow{2}{*}{" + data_obj.country + r"} & Accuracy &" + r" & ".join(
            dec.format(performances[model_name]["mean_acc"][0]) + r" $\pm$ " + dec.format(
                performances[model_name]["mean_acc"][1]) for model_name in
            table_order) + r"\\" + "\n" + r"    & F-score & " + r" & ".join(
            dec.format(performances[model_name]["mean_f1"][0]) + r" $\pm$ " + dec.format(
                performances[model_name]["mean_f1"][1]) for model_name in
            table_order) + r"\\" + "\n"

        f.write("\n" + string)
    return string


def get_confusion_matrices(model, data_obj, split="all"):
    U, P, V, _ = data_obj.get_data(split)
    M, K = len(U), len(P)

    v_max = np.argmax(V, axis=1)
    p_max = np.argmax(model.predict(U, P), axis=1)
    print("Accuracy ({:s}): {:.5f}".format(split, sum(v_max == p_max) / float(M)))

    conf = np.zeros((K, K))
    for i in range(M):
        conf[v_max[i], p_max[i]] += 1

    conf_names = ['Absolute values', 'Percentages by row (recall)', 'Percentages by column (precision)',
                  'Merge (f-score)']
    confs = []
    for _ in conf_names:
        confs.append(conf.copy())

    for i, el in enumerate(conf_names):
        conf = confs[i]
        for k in range(K):
            if i == 1:
                conf[k, :] = conf[k, :] / sum(conf[k, :])
            elif i == 2:
                conf[:, k] = conf[:, k] / sum(conf[:, k])
            elif i == 3:
                conf = 2 * confs[1] * confs[2] / (confs[1] + confs[2])

    return confs


def plot_comparison_confmats(data_obj, models, names=("Deductive VAAs", "Social VAAs", "Learning VAAs")):
    assert len(models), len(names)
    confs = []
    for model in models:
        confs.append(get_confusion_matrices(model, data_obj, "test"))

    fig, ax = plt.subplots(1, len(models), figsize=(15 / 4 * len(models), 4))
    for i, conf in enumerate(confs):
        cax = plot_confusion(ax[i], confs[i][1], names[i], data_obj.party_names)
    plt.colorbar(cax)

    plt.tight_layout()
    fig.savefig(PLOTS_PATH + "comparison_confusion_matrices.eps")
    fig.show()


def print_distance_matrices(models, pages=5):
    data_obj = models[0].dataobj
    party_names = data_obj.party_names
    P = data_obj.get_P()  # For compatibility among different DBs
    N, L = P.shape[1:]

    full_ds = np.zeros((len(models), N, L, L))
    for i, model in enumerate(models):
        if model.with_weights:
            w = model.w.get_value()
        else:
            w = np.ones(N)
        full_ds[i] = np.array([el * w[j] for j, el in enumerate(model.get_full_d())])

    full_d = np.mean(full_ds, axis=0)
    max_means = np.max(np.max(np.max(np.abs(full_d), axis=1, keepdims=True), axis=2, keepdims=True), axis=0,
                       keepdims=True)
    var_d = np.std(full_ds, axis=0) / max_means

    print("Obtaining frequencies...")
    freq_abs, freq_party = data_obj.get_frequencies()

    # scores = np.zeros_like(full_d)
    # for j in range(N):
    #     scores[j] = np.multiply(freq_abs[j], full_d[j])

    print("Plotting matrices...")
    _max_D = np.amax(abs(full_d))
    # _max_s = np.amax(abs(scores))

    num_panels = 5
    for page in range(pages):
        i = 1
        fig3, big_axes = plt.subplots(figsize=(2.2 * num_panels, 2.7 * N / pages), nrows=int(N / pages),
                                      ncols=1)  # , sharey=True)
        for num, big_ax in enumerate(big_axes):
            j = int(page * N / pages + num)
            big_ax.set_title(str(j + 1) + ": " + data_obj.Q[j][:110])
            big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
            big_ax._frameon = False

            # Distance matrices (mean)
            sym_freq = unfold_sum_matrix(freq_abs[j])
            sym_freq[-1], sym_freq[:, -1] = 1, 1
            full_d[j, sym_freq == 0] = np.nan
            axes = fig3.add_subplot(int(N / pages), num_panels, i)
            plotLLmatrix(axes, full_d[j], vmax=_max_D)
            i += 1

            # Distance matrices (std)
            var_d[j, sym_freq == 0] = np.nan
            axes = fig3.add_subplot(int(N / pages), num_panels, i)
            plotLLmatrix(axes, var_d[j], vmin=0, vmax=1, cmap='Blues')
            i += 1

            # Absolute frequencies
            # axes = fig3.add_subplot(int(N / pages), num_panels, i)
            # plotLLmatrix(axes, freq_abs[j], cmap='Blues')
            # i += 1

            # Party frequencies
            axes = fig3.add_subplot(int(N / pages), num_panels, i)
            plotKLmatrix(axes, freq_party[j], party_names)
            i += 1

            # Party frequencies (per-party normalization)
            tot_party = np.sum(freq_party[j], axis=1)
            axes = fig3.add_subplot(int(N / pages), num_panels, i)
            plotKLmatrix(axes, freq_party[j] / tot_party[:, None], party_names)
            i += 1

            # Party answers
            axes = fig3.add_subplot(int(N / pages), num_panels, i)
            plotKLmatrix(axes, P[:, j, :], party_names)
            i += 1

        fig3.set_facecolor('w')
        plt.tight_layout()
        fig3.savefig(PLOTS_PATH + "distance_matrices-{:d}-{:s}.png".format(page, models[0].name))
        plt.show()


def print_weights(models, ax=None, use_color=False):
    N = models[0].get_full_d().shape[0]

    combined_weights = np.zeros((len(models), N))
    for i, model in enumerate(models):
        if model.with_weights:
            w = model.w.get_value()
        else:
            w = np.ones(N)
        combined_weights[i] = np.abs(np.multiply(w, [np.linalg.norm(el) for el in model.get_full_d()]))

    if not models[0].with_weights:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.bar(range(N), np.mean(combined_weights, axis=0))
        ax.yaxis.grid()
        ax.xaxis.set(ticks=range(N), ticklabels=range(1, 31))
    else:
        if False:
            fig, ax = plt.subplots(4, 1, figsize=(10, 10))
            ax[0].bar(range(N), np.mean([np.abs(model.w.get_value()) for model in models], axis=0),
                      yerr=np.std([np.abs(model.w.get_value()) for model in models], axis=0))
            ax[0].set(title="w")
            ax[1].bar(range(N),
                      np.mean([[np.linalg.norm(el) for el in model.get_full_d()] for model in models], axis=0),
                      yerr=np.std([[np.linalg.norm(el) for el in model.get_full_d()] for model in models], axis=0))
            ax[1].set(title="norm of d")
            ax[2].bar(range(N), np.mean([[np.max(el) for el in model.get_full_d()] for model in models], axis=0),
                      yerr=np.std([[np.max(el) for el in model.get_full_d()] for model in models], axis=0))
            ax[2].set(title="max of d")
            ax[3].bar(range(N), np.mean(combined_weights, axis=0), yerr=np.std(combined_weights, axis=0))
            ax[3].set(title="final decision")
            for ax in ax:
                ax.yaxis.grid()
                ax.xaxis.set(ticks=range(N), ticklabels=range(1, 31))
        else:
            colors = ["grey" for _ in range(N)]
            color = {"Europe": "blue", "Healthcare": "yellow", "Inequality": "red", "Environment": "green",
                     "Feminism": "purple", "Catalonia": "orange",
                     "Public order": "black", "Immigration": "brown"}
            coloring = {"Spain": {"Europe": [1, 6], "Inequality": [14, 15], "Environment": [16],
                                  "Feminism": [22], "Catalonia": [27, 29], },  # "Participatory democracy": [28]},
                        "England": {"Europe": [6, 9, 10], "Healthcare": [11], "Inequality": [14, 15],
                                    "Environment": [20],
                                    "Immigration": [21]},
                        "Greece": {"Europe": [1, 6], "Inequality": [13], "Public order": [23, 24], "Immigration": [28]},
                        "Italy": {"Europe": [1, 6], "Inequality": [13, 18], "Environment": [15], "Immigration": [27]}}
            data_obj = models[0].dataobj
            if data_obj.country != "Portugal" and data_obj.country in coloring:
                for topic, questions in coloring[data_obj.country].items():
                    for question in questions:
                        colors[question - 1] = color[topic]

            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 2.5))
            else:
                fig = None
            ax.set(ylabel=data_obj.country)
            if not use_color:
                ax.bar(range(1, N + 1), np.mean(combined_weights, axis=0), yerr=np.std(combined_weights, axis=0))
                ax.set_xticks([])
                ax.set_xticklabels([])
            else:
                ax.bar(range(1, N + 1), np.mean(combined_weights, axis=0), yerr=np.std(combined_weights, axis=0),
                       color=colors)
                legend = []
                if data_obj.country != "Portugal" and data_obj.country in coloring:
                    for topic in coloring[data_obj.country]:
                        legend.append(mpatches.Patch(color=color[topic], label=topic))
                    ax.legend(handles=legend)

    # ax.set(title="Relative weights of the issue questions", xlabel="Issue question")
    if fig is not None:
        fig.show()


def plot_sample_distance_matrices(models, matrix_numbers, axes):
    data_obj = models[0].dataobj
    P = data_obj.get_P()  # For compatibility among different DBs
    N, L = P.shape[1:]

    full_ds = np.zeros((len(models), N, L, L))
    for i, model in enumerate(models):
        if model.with_weights:
            w = model.w.get_value()
        else:
            w = np.ones(N)
        full_ds[i] = np.array([el * w[j] for j, el in enumerate(model.get_full_d())])

    full_d = np.mean(full_ds, axis=0)
    # var_d = np.std(full_ds, axis=0) / np.max(abs(full_d))

    print("Obtaining frequencies...")
    freq_abs, freq_party = data_obj.get_frequencies()

    print("Plotting matrices...")
    titles = ["Proximity-like", "Directionality-like", "Hybrid-like", "Other possible paradigms", "Rather nonsense"]
    for i, (country, j) in enumerate(matrix_numbers):
        if country == data_obj.country:
            sym_freq = unfold_sum_matrix(freq_abs[j])
            sym_freq[-1], sym_freq[:, -1] = 1, 1

            # Distance matrices (mean)
            _max_D = np.amax(abs(full_d[j]))
            full_d[j, sym_freq == 0] = np.nan
            plotLLmatrix(axes[i], full_d[j], vmax=_max_D)
            axes[i].set(title=titles[i])

            # Distance matrices (std)
            # var_d[j, sym_freq == 0] = np.nan
            # plotLLmatrix(axes[1][i], var_d[j], vmin=0, vmax=1, cmap="Blues")


### THIS SCRIPT IS FOR PRODUCING THE RESULTS OF ALREADY-TRAINED MODELS. THEY CAN BE TRAINED AT THE 'algorithm.py' SCRIPT
if __name__ == "__main__":
    # With NO as changeable
    # data_pars = {"issue_voters": "issue", "education": "high_ed", "politically_active": "somewhat_poli"}
    # models = [{"country": "England", "number": "20190710-163841"},
    # {"country": "Portugal", "number": "20190710-173443"}, {"country": "Greece", "number": "20190710-172109"},
    # {"country": "Spain", "number": "20190710-175755"}, {"country": "Italy", "number": "20190710-181919"}]

    # data_pars = {"issue_voters": "no_issue", "education": "no_education", "politically_active": "no_poli"}
    # models = [{"country": "England", "number": "20190717-204933"},
    # {"country": "Portugal", "number": "20190717-211304"}, {"country": "Greece", "number": "20190717-213852"},
    # {"country": "Spain", "number": "20190717-224505"}, {"country": "Italy", "number": "20190717-230459"},

    data_pars = {"issue_voters": "issue", "education": "highschool", "politically_active": "very_poli"}
    models = [
        {"country": "England", "number": "20190812-090044", "svm": "20190812-203601"},
        {"country": "Portugal", "number": "20190812-090044", "svm": "20190812-204224"},
        {"country": "Greece", "number": "20190812-090044", "svm": "20190812-205024"},
        {"country": "Spain", "number": "20190812-193821", "svm": "20190813-043139"},
        {"country": "Italy", "number": "20190812-090044", "svm": "20190812-205307"},
    ]

    data_conditions = data_pars["issue_voters"] + "-" + data_pars["education"] + "-" + data_pars[
        "politically_active"]

    # Choose which results to activate
    PERF_TABLES = False
    WEIGHT_PLOTS = False
    SAMPLE_MATRICES = True

    if PERF_TABLES:
        fig, axes = plt.subplots(len(models), figsize=(9, 1.3 * len(models)))  # For the f-scores
        string = r"""\begin{tabular}{ccccc}
    \hline
    &  & Traditional VAA & \textbf{Learning VAA} & Social VAA (SVM) \\
    \hline""" + "\n"

    if WEIGHT_PLOTS:
        fig2, axes2 = plt.subplots(len(models), figsize=(11, 1.5 * len(models)))

    if SAMPLE_MATRICES:
        matrix_numbers = (("England", 5), ("Italy", 5), ("England", 19), ("Spain", 28), ("Spain", 20))
        fig3, axes3 = plt.subplots(1, len(matrix_numbers), figsize=(12 / 5 * len(matrix_numbers), 2.5))

    for i, model_info in enumerate(models):
        data_obj = DataHolder(model_info['country'], data_pars)
        fname = model_info['country'] + "-" + data_conditions + "-"

        # data_obj.get_random_accuracy(fname)
        learning_models = load_models(data_obj, file_name=fname + model_info['number'], with_weights=True)

        if PERF_TABLES:
            standard_model = Model(data_obj, file_name="Hybrid", with_weights=False)
            svm_models = load_models(data_obj, file_name="svm-" + fname + model_info['svm'], with_weights=False)
            models = {"Standard VAA": standard_model, "Social VAA": svm_models, "Learning VAA": learning_models}
            string += obtain_performance_table(models, axes[i])

        if WEIGHT_PLOTS:
            print_weights(learning_models, axes2[i])

        if SAMPLE_MATRICES:
            if model_info["country"] in [el[0] for el in matrix_numbers]:
                plot_sample_distance_matrices(learning_models, matrix_numbers, axes3)

        # print_distance_matrices(learning_models, pages=5)
        # plot_sample_distance_matrices(learning_models)

        # plot_comparison_confmats(dataobj, mendez_model, social_model, our_model)


    if PERF_TABLES:
        string += r"\end{tabular}"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(TAB_DIR + "table-" + data_conditions + timestamp + ".txt", 'w') as f:
            f.write(string)

        fig.tight_layout()
        fig.savefig(PLOTS_PATH + "fscores-" + data_conditions + model_info["number"] + ".eps")
        fig.show()

    if WEIGHT_PLOTS:
        fig2.tight_layout()
        fig2.savefig(PLOTS_PATH + "weights-" + data_conditions + model_info["number"] + ".eps")
        fig2.show()

    if SAMPLE_MATRICES:
        # plt.colorbar(cax)
        fig3.tight_layout()
        fig3.savefig(PLOTS_PATH + "sample_distances-{:s}-{:s}.eps".format(data_conditions, model_info["number"]))
        fig3.show()
