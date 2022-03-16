#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics.pairwise import pairwise_kernels
from src.utilities.data import load_mnist
from src.utilities.plot.settings import *
from secml.ml.classifiers import CClassifierSVM, CClassifierLogistic, CClassifierRidge
from secml.ml.classifiers import CClassifierPyTorch
from src.attacks.backdoor.c_backdoor_dataset import CDatasetPoisoner
from src.utilities.attack import increasing_label, train_on_poison
from src.attacks.backdoor.trigger_data import Trigger
from src.utilities.metrics import eval_accuracy
from secml.array import CArray
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score as AC
from sklearn.metrics import log_loss
import torch
from torchvision import models
from src.classifiers.model.pretrained import PretrainedNet
from secml.ml.features.normalization import CNormalizerMeanStd
import pickle
import os


def save_stats(stats, name):
    with open(name + ".pickle", "wb") as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_stats(name):
    with open(name + ".pickle", "rb") as handle:
        stats = pickle.load(handle)
    return stats


seed = 999

n_alpha = 100

colors = sns.color_palette("flare")
c_color = [colors[-1], colors[2]]
alphas = [0.7, 1]
markers = ["s", "D"]


def gen_log_space(limit, n):
    result = [1]
    if n > 1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
    while len(result) < n:
        next_value = result[-1] * ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1] + 1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x) - 1, result)), dtype=np.uint64)


# In[3]:


def save_object(obj, filename):
    with open(filename, "wb") as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HGHEST_PROTOCOL)


def ds_to_numpy(ds):
    x = ds.X.deepcopy()
    y = ds.Y.deepcopy()
    return x.tondarray(), y.tondarray()


def pair2string(pair):
    return "{}-{}".format(pair[0], pair[1])


def hinge_loss(ds, clf):
    x, y = ds_to_numpy(ds)
    scores = clf.decision_function(x)
    loss = 1 - (2 * y - 1) * scores
    loss[loss < 0] = 0
    return loss.mean()


def square_loss(ds, clf):
    x, y = ds_to_numpy(ds)
    y[y == 0] = -1
    scores = clf.decision_function(x)
    return ((1.0 - y * scores) ** 2).mean()


def logistic_loss(ds, clf):
    x, y = ds_to_numpy(ds)
    scores = clf.predict_proba(x)
    return log_loss(y_true=y, y_pred=scores)


def sklearn_accuracy(clf, ds):
    x, y = ds_to_numpy(ds)
    return AC(clf.predict(x), y)


def backdoor_path(classifiers, tr_p, ts_p, poison_idx, beta_lst, n_features):
    linear_clfs = [classifiers[x] for x in classifiers.keys() if "RBF" not in x]
    rbf_clfs = [classifiers[x] for x in classifiers.keys() if "RBF" in x]

    linear_path = backdoor_path_linear(
        linear_clfs, tr_p, ts_p, poison_idx, beta_lst, n_features
    )
    rbf_path = backdoor_dual_path(
        rbf_clfs, tr_p, ts_p, poison_idx, beta_lst, n_features
    )
    return linear_path + rbf_path


def backdoor_path_linear(classifiers, tr_p, ts_p, poison_idx, beta_lst, n_features):
    classifiers_paths = []

    for i, clf in enumerate(classifiers):
        classifiers_paths += [
            {
                "clf_name": clf["name"],
                "C": [],
                "gamma": clf["gamma"],
                "weights": [],
                "intercept": [],
                "angle": [],
                "l2": [],
                "backdoor_accuracy": [],
            }
        ]
        c_range = clf["regularizer"]

        for c, C in enumerate(c_range):
            weights = np.zeros(shape=(len(beta_lst), n_features))
            intercept = np.zeros(shape=len(beta_lst))
            angle = np.zeros(shape=len(beta_lst))
            l2 = np.zeros(shape=len(beta_lst))
            weights_0 = None

            for j, beta in enumerate(beta_lst):
                beta_weight = beta / 100
                samples_weight = np.ones(tr_p.X.shape[0])
                samples_weight[poison_idx.tolist()] = beta_weight

                if j % 9 == 0:
                    print("clf_name: {} C: {}  beta: {}".format(clf["name"], C, beta))

                clf_init = clf["init"]
                clf_p = clf_init(c=C)
                p_x, p_y = ds_to_numpy(tr_p)
                print(p_x.shape, p_x.max(), p_x.min())
                status = clf_p.fit(p_x, p_y, sample_weight=samples_weight)

                # linear classifier
                weights[j] = status.coef_
                intercept[j] = status.intercept_

                if beta == 0:
                    weights_0 = status.coef_
                norm = np.linalg.norm(weights_0, 2) * np.linalg.norm(weights[j], 2)
                angle[j] = (weights_0 @ weights[j].T) / norm
                l2[j] = np.linalg.norm(weights[j], 2)

            backdoor_acc = sklearn_accuracy(clf_p, ts_p)
            classifiers_paths[i]["backdoor_accuracy"] += [backdoor_acc]
            classifiers_paths[i]["C"] += [C]
            classifiers_paths[i]["weights"] += [weights]
            classifiers_paths[i]["intercept"] += [intercept]
            classifiers_paths[i]["angle"] += [angle]
            classifiers_paths[i]["l2"] += [l2]
    return classifiers_paths


def backdoor_dual_path(classifiers, tr_p, ts_p, poison_idx, beta_lst, n_features):
    classifiers_paths = []

    for i, clf in enumerate(classifiers):
        gamma = clf["gamma"]
        classifiers_paths += [
            {
                "clf_name": clf["name"],
                "C": [],
                "gamma": gamma,
                "weights": [],
                "intercept": [],
                "angle": [],
                "l2": [],
                "backdoor_accuracy": [],
            }
        ]
        c_range = clf["regularizer"]

        p_x, p_y = ds_to_numpy(tr_p)
        K = pairwise_kernels(p_x, metric="rbf", gamma=gamma)
        for c, C in enumerate(c_range):
            alpha_b = np.zeros(shape=(len(beta_lst), tr_p.X.shape[0]))

            angle = np.zeros(shape=len(beta_lst))
            l2 = np.zeros(shape=len(beta_lst))

            alpha_0 = None

            for j, beta in enumerate(beta_lst):
                beta_weight = beta / 100
                samples_weight = np.ones(tr_p.X.shape[0])
                samples_weight[poison_idx.tolist()] = beta_weight

                if j % 9 == 0:
                    print("clf_name: {} C: {}  beta: {}".format(clf["name"], C, beta))

                clf_init = clf["init"]
                clf_p = clf_init(c=C)

                status = clf_p.fit(p_x, p_y, sample_weight=samples_weight)
                alpha_b[j, status.support_] = status.dual_coef_

                if beta == 0:
                    alpha_0 = alpha_b[j, :]

                alpha_0_norm = (alpha_0.T @ K @ alpha_0.T) ** (1 / 2)
                alpha_b_norm = (alpha_b[j].T @ K @ alpha_b[j].T) ** (1 / 2)
                angle[j] = (alpha_0.T @ K @ alpha_b[j, :]) / (
                    alpha_0_norm * alpha_b_norm + 1e-16
                )
                # ratio[j] = alpha_b_norm / alpha_0_norm
                l2[j] = alpha_b_norm

            backdoor_acc = sklearn_accuracy(clf_p, ts_p)
            classifiers_paths[i]["backdoor_accuracy"] += [backdoor_acc]
            classifiers_paths[i]["C"] += [C]
            classifiers_paths[i]["weights"] += [alpha_b]
            classifiers_paths[i]["angle"] += [angle]
            # classifiers_paths[i]["ratio"] += [ratio]
            classifiers_paths[i]["l2"] += [l2]

    return classifiers_paths


def plot_backdoor_ratio(classifiers_paths, name):
    dataset = name.split("/")[0]
    for i, clf_path in enumerate(classifiers_paths):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        clf_name = clf_path["clf_name"]

        ax.set_ylabel("$\\rho$", fontsize=18)
        ax.set_xlabel("$\\nu$", fontsize=18)
        ax.set_title("{}".format(clf_name), fontsize=20)
        # ax.legend(fontsize=12, markerscale=1.2)
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(axis="x", labelsize=14)
        style = ["dotted", "dashed"]
        c_range = clf_path["C"]

        for j, C in enumerate(c_range):
            angle = clf_path["angle"][j]
            ratio = clf_path["l2"][j]

            legend = "$\\lambda$={:g} BA={:.2f}".format(
                1 / C, clf_path["backdoor_accuracy"][j]
            )

            sns.lineplot(
                0.5 * (1 - angle),
                ratio,
                ax=ax,
                linewidth=3,
                color=c_color[j],
                alpha=alphas[j],
                label=legend,
                linestyle=style[j],
            )
            ax.legend(
                loc="best",
                # fancybox=True,
                markerscale=0.5,
                borderaxespad=0.0,
                fontsize=14,
                framealpha=0.3,
                handletextpad=0.1,
                labelspacing=0.6,
            )

        plt.tight_layout()
        os.makedirs(name, exist_ok=True)

        if "RBF" in clf_name:
            gamma = clf_path["gamma"]
            out = "{}/angle_backdoor_path_SVM_RBF_gamma={:.0e}_{}.pdf".format(
                name, gamma, dataset
            )
        else:
            out = "{}/angle_backdoor_path_{}_{}.pdf".format(name, clf_name, dataset)
        plt.savefig(
            out, bbox_inches="tight", pad_inches=0,
        )
        plt.show()


def plot_weights_backdoor_path(
    classifiers_paths,
    weights_to_show,
    name,
    beta_lst,
    ylabel="$\mathbf{w}^\star(\\beta)$",
):
    dataset = name.split("/")[0]
    for i, clf_path in enumerate(classifiers_paths):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        clf_name = clf_path["clf_name"]
        c_range = clf_path["C"]

        ax.set_xlabel("$\\beta$")
        ax.set_ylabel(ylabel)
        ax.set_title("{}".format(clf_name), fontsize=14)
        ax.legend(fontsize=12, markerscale=1.4)

        for j, C in enumerate(c_range[::-1]):
            weights = clf_path["weights"][::-1][j]

            for k, w in enumerate(weights_to_show):
                legend = None

                if k == 0:
                    legend = "$\\lambda$={}".format(1 / C)
                sns.lineplot(
                    beta_lst / 100,
                    weights[:, w],
                    ax=ax,
                    linewidth=3,
                    color=c_color[::-1][j],
                    alpha=alphas[j],
                    label=legend,
                    marker=markers[j],
                )
        plt.tight_layout()
        os.makedirs(name, exist_ok=True)

        plt.savefig(
            "{}/weights_backdoor_path_{}_{}.jpeg".format(name, clf_name, dataset),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.show()


def incremental_loss(classifiers, tr, ts, ts_p, poisoner, beta_lst, preprocess=None):
    classifiers_paths = []
    train = tr.deepcopy()

    for i, clf_key in enumerate(classifiers.keys()):
        clf = classifiers[clf_key]
        classifiers_paths += [
            {
                "clf_name": clf["name"],
                "C": [],
                "gamma": clf["gamma"],
                "p": [],
                "val_loss_backdoor": [],
                "val_loss_clean": [],
            }
        ]
        c_range = clf["regularizer"]
        for j, C in enumerate(c_range):

            for p in [0.01, 0.1, 0.2]:
                tr_p, poison_idx = poisoner.poison(train, proportion=p)
                val_loss_backdoor = np.zeros(shape=len(beta_lst))
                val_loss_clean = np.zeros(shape=len(beta_lst))

                if preprocess is not None:
                    tr_p.X = preprocess.forward(tr_p.X)

                if clf["name"] == "ridge":
                    ts_p.Y[ts_p.Y == 0] = -1
                    tr_p.Y[tr_p.Y == 0] = -1

                for a, beta in enumerate(beta_lst):
                    if a % 9 == 0:
                        print(
                            "clf_name: {} C: {}  beta: {}".format(clf["name"], C, beta)
                        )

                    beta_weight = beta / 100
                    samples_weight = np.ones(tr_p.X.shape[0])
                    samples_weight[poison_idx.tolist()] = beta_weight

                    clf_init = clf["init"]
                    loss = clf["loss"]
                    clf_p = clf_init(c=C)
                    p_x, p_y = ds_to_numpy(tr_p)

                    clf_p.fit(p_x, p_y, sample_weight=samples_weight)

                    val_loss_backdoor[a] = loss(ts_p, clf_p)
                    val_loss_clean[a] = loss(ts, clf_p)
                classifiers_paths[i]["C"] += [C]
                classifiers_paths[i]["p"] += [p]
                classifiers_paths[i]["val_loss_backdoor"] += [val_loss_backdoor]
                classifiers_paths[i]["val_loss_clean"] += [val_loss_clean]
    return classifiers_paths


def plot_incremental_loss(incremental_loss, name, beta_lst):
    colors_palette = [
        sns.color_palette("mako_r"),
        sns.color_palette("gist_heat_r"),
        sns.color_palette("summer_r"),
        sns.color_palette("RdPu"),
        sns.color_palette("RdPu"),
    ]

    dataset = name.split("/")[0]

    for clf_idx, clf in enumerate(incremental_loss):
        colors = colors_palette[clf_idx]
        clf_name = clf["clf_name"]
        if clf_name == "Logistic":
            clf_name = "LC"
        if clf_name == "Ridge":
            clf_name = "RC"
        c_range = np.unique(clf["C"])
        p_range = np.unique(clf["p"])

        fig, axs = plt.subplots(1, len(c_range), figsize=(4 * len(c_range), 4))
        axs = axs.flatten()
        j = 0
        for i, C in enumerate(c_range):

            loss_labels = clean_label = None
            for p_idx, p in enumerate(p_range):
                if i == (len(c_range) - 1):
                    clean_label = "TS p=%g" % p
                    loss_labels = "TS+BT p=%g" % p
                sns.lineplot(
                    beta_lst / 100,
                    clf["val_loss_clean"][j],
                    ax=axs[i],
                    label=clean_label,
                    linestyle="dotted",
                    linewidth=3,
                    color=colors[p_idx * 2],
                    alpha=0.5,
                )

                sns.lineplot(
                    beta_lst / 100,
                    clf["val_loss_backdoor"][j],
                    ax=axs[i],
                    label=loss_labels,
                    linewidth=3,
                    color=colors[p_idx * 2 + 1],
                    alpha=1,
                )
                j += 1
            axs[i].set_xlabel("$\\beta$", fontsize=18)
            axs[i].set_ylabel("Test loss", fontsize=18)
            axs[i].set_title(
                "{} $\\lambda$={:g}".format(clf_name, 1 / C), fontsize=18,
            )

            axs[i].tick_params(axis="y", labelsize=14)
            axs[i].tick_params(axis="x", labelsize=14)
        plt.legend(
            markerscale=0.5,
            framealpha=0.2,
            handletextpad=0.2,
            labelspacing=0.25,
            fontsize=15,
        )
        plt.tight_layout()
        os.makedirs(name, exist_ok=True)

        if "RBF" in clf_name:
            gamma = clf["gamma"]
            out = "{}/incremental_backdoor_SVM_RBF_gamma={:.0e}_{}.pdf".format(
                name, gamma, dataset
            )
        else:
            out = "{}/incremental_backdoor_{}_{}.pdf".format(name, clf_name, dataset)
        plt.savefig(
            out, bbox_inches="tight", pad_inches=0,
        )
        plt.show()



def run_mnist():
    # define classifiers to test
    mnist_classifiers = {
        "SVM": {
            "init": lambda c: LinearSVC(C=c, loss="hinge"),
            "loss": hinge_loss,
            "regularizer": [1e-02, 10],
            "gamma": None,
            "name": "SVM",
        },
        "Logistic": {
            "init": lambda c: LogisticRegression(C=c, solver="liblinear"),
            "loss": logistic_loss,
            "regularizer": [1e-1, 100],
            "gamma": None,
            "name": "LC",
        },
        "Ridge": {
            "init": lambda c: RidgeClassifier(alpha=1 / (2 * c)),
            "loss": square_loss,
            "regularizer": [1e-03, 1],
            "gamma": None,
            "name": "RC",
        },
        "SVM RBF small gamma": {
            "init": lambda c: SVC(C=c, kernel="rbf", gamma=5e-04, cache_size=4000),
            "loss": hinge_loss,
            "regularizer": [1, 100],
            "gamma": 5e-04,
            "name": "RBF SVM $\gamma=5\mathrm{e}-4$",
        },
        "SVM RBF large gamma": {
            "init": lambda c: SVC(C=c, kernel="rbf", gamma=5e-03, cache_size=4000),
            "loss": hinge_loss,
            "regularizer": [1, 100],
            "gamma": 5e-03,
            "name": "RBF SVM $\gamma=5\mathrm{e}-3$",
        },
    }

    # define dataset and number of training samples
    n_tr = 1500  # Number of training set samples
    n_val = 1  # Number of validation set samples
    n_ts = 500  # Number of test set samples
    # how much step for beta
    beta_lst = gen_log_space(100, 20)

    # In[8]:

    for digits in [(7, 1), (3, 0), (5, 2)]:
        print(digits)
        data_name = pair2string(digits)
        tr, val, ts = load_mnist(
            n_tr=n_tr, n_val=n_val, n_ts=n_ts, digits=digits, random_state=seed
        )
        trigger = Trigger(
            input_size=(1, 28, 28),
            trigger_size=(6, 6),  # trigger_size=(3, 3),
            trigger_type="badnet",
            n_triggers=2,
            random_state=seed,
        )
        poisoner = CDatasetPoisoner(
            trigger=trigger, target=increasing_label(n_labels=2), random_state=seed,
        )
        tr_p, poison_idx = poisoner.poison(tr, proportion=0.1)
        ts_p, _ = poisoner.poison(ts, proportion=1)

        classifiers_paths_mnist = backdoor_path(
            mnist_classifiers, tr_p, ts_p, poison_idx, beta_lst=beta_lst, n_features=784
        )
        plot_backdoor_ratio(
            classifiers_paths_mnist, name="mnist-t6/{}".format(data_name)
        )
        save_stats(classifiers_paths_mnist, "mnist-t6/mnist-{}-path".format(data_name))

        print("\n BACKDOOR LEARNING CURVES \n ")
        mnist_incremental_loss = incremental_loss(
            mnist_classifiers, tr, ts, ts_p, poisoner, beta_lst=beta_lst
        )
        plot_incremental_loss(
            mnist_incremental_loss,
            beta_lst=beta_lst,
            name="mnist-t6/{}".format(data_name),
        )
        save_stats(
            mnist_incremental_loss,
            "mnist-t6/mnist-{}-incremental-curves".format(data_name),
        )


def run_cifar():

    # # Incremental Backdoor CIFAR10
    # In[17]:
    from src.utilities.data import load_bin_cifar

    torch.cuda.set_device("cuda:0")
    torch.device("cuda:0")
    seed = 999
    n_tr = 1500  # Number of training set samples
    n_val = 1  # Number of validation set samples
    n_ts = 500  # Number of test set samples
    beta_lst = gen_log_space(100, 20)

    alexnet = models.alexnet(pretrained=True)

    # freeze convolution weights
    for param in alexnet.features.parameters():
        param.requires_grad = False
    alexnet.classifier[6].out_feature = 2

    # use pretrained module with input normalization
    pre_net = PretrainedNet(alexnet, in_shape=(3, 224, 224), n_classes=2)

    net = CClassifierPyTorch(
        model=pre_net, input_shape=(3, 32, 32), pretrained=True, batch_size=256,
    )

    from secml.ml.features.normalization import CNormalizerDNN

    out_layer = net.layer_names[-2]
    net_preprocess = CNormalizerDNN(net, out_layer=out_layer)


    cifar_classifiers = {
        "SVM": {
            "init": lambda c: LinearSVC(C=c, loss="hinge"),
            "loss": hinge_loss,
            "regularizer": [1e-04, 10],
            "gamma": None,
            "name": "SVM",
        },
        "Logistic": {
            "init": lambda c: LogisticRegression(C=c, solver="liblinear"),
            "loss": logistic_loss,
            "regularizer": [1e-04, 1e-02],
            "gamma": None,
            "name": "LC",
        },
        "Ridge": {
            "init": lambda c: RidgeClassifier(alpha=1 / (2 * c)),
            "loss": square_loss,
            "regularizer": [1e-04, 1],
            "gamma": None,
            "name": "RC",
        },
        "SVM RBF small gamma": {
            "init": lambda c: SVC(C=c, kernel="rbf", gamma=1e-04),
            "loss": hinge_loss,
            "regularizer": [1e-02, 1],
            "gamma": 1e-04,
            "name": "RBF SVM $\gamma=1\mathrm{e}-4$",
        },
        "SVM RBF large gamma": {
            "init": lambda c: SVC(C=c, kernel="rbf", gamma=1e-03),
            "loss": hinge_loss,
            "regularizer": [1e-02, 1],
            "gamma": 1e-03,
            "name": "RBF SVM $\gamma=1\mathrm{e}-3$",
        },
    }

    for labels in [(2, 5), (0, 9), (6, 0)]:
        data_name = pair2string(labels)
        tr, val, ts = load_bin_cifar(
            labels=labels, n_tr=n_tr, n_val=1, n_ts=n_ts, random_state=seed
        )

        trigger = Trigger(
            input_size=(3, 32, 32),
            trigger_size=(16, 16),  # trigger_size=(8, 8),
            trigger_type="badnet",
            n_triggers=2,
        )
        poisoner = CDatasetPoisoner(
            trigger=trigger, target=increasing_label(n_labels=2), random_state=seed,
        )
        tr_p, poison_idx = poisoner.poison(tr, proportion=0.1)
        ts_p, _ = poisoner.poison(ts, proportion=1)

        phi_tr = tr.deepcopy()
        phi_tr_p = tr_p.deepcopy()

        phi_ts = ts.deepcopy()
        phi_ts_p = ts_p.deepcopy()

        phi_tr.X = net_preprocess.forward(tr.X)
        phi_tr_p.X = net_preprocess.forward(tr_p.X)
        phi_ts.X = net_preprocess.forward(ts.X)
        phi_ts_p.X = net_preprocess.forward(ts_p.X)

        classifiers_paths_cifar = backdoor_path(
            cifar_classifiers,
            phi_tr_p,
            phi_ts_p,
            poison_idx,
            beta_lst=beta_lst,
            n_features=4096,
        )
        plot_backdoor_ratio(
            classifiers_paths_cifar, name="cifar-t16/{}".format(data_name)
        )
        save_stats(classifiers_paths_cifar, "cifar-t16/cifar-{}-path".format(data_name))

        cifar_incremental_loss = incremental_loss(
            cifar_classifiers,
            tr,
            phi_ts,
            phi_ts_p,
            poisoner,
            beta_lst=beta_lst,
            preprocess=net_preprocess,
        )

        plot_incremental_loss(
            cifar_incremental_loss,
            beta_lst=beta_lst,
            name="cifar-t16/{}".format(data_name),
        )
        save_stats(
            cifar_incremental_loss,
            "cifar-t16/cifar-{}-incremental-curves".format(data_name),
        )


def run_imagenette():

    from src.utilities.data import load_bin_imagenette

    torch.cuda.set_device("cuda:0")
    torch.device("cuda:0")
    seed = 999
    n_tr = 1500  # Number of training set samples
    n_val = 1  # Number of validation set samples
    n_ts = 500  # Number of test set samples
    beta_lst = gen_log_space(100, 20)

    alexnet = models.alexnet(pretrained=True)

    # freeze convolution weights
    for param in alexnet.features.parameters():
        param.requires_grad = False
    alexnet.classifier[6].out_feature = 2

    # use pretrained module with input normalization
    pre_net = PretrainedNet(alexnet, in_shape=(3, 224, 224), n_classes=2)
    normalizer = CNormalizerMeanStd(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    net = CClassifierPyTorch(
        model=pre_net,
        input_shape=(3, 224, 224),
        pretrained=True,
        batch_size=256,
        preprocess=normalizer,
    )
    from secml.ml.features.normalization import CNormalizerDNN

    out_layer = net.layer_names[-2]
    net_preprocess = CNormalizerDNN(net, out_layer=out_layer)

    imagenette_classifiers = {
        "SVM": {
            "init": lambda c: LinearSVC(C=c, loss="hinge"),
            "loss": hinge_loss,
            "regularizer": [1e-04, 100],
            "gamma": None,
            "name": "SVM",
        },
        "Logistic": {
            "init": lambda c: LogisticRegression(C=c, solver="liblinear"),
            "loss": logistic_loss,
            "regularizer": [1e-04, 100],
            "gamma": None,
            "name": "LC",
        },
        "Ridge": {
            "init": lambda c: RidgeClassifier(alpha=1 / (2 * c)),
            "loss": square_loss,
            "regularizer": [1e-04, 100],
            "gamma": None,
            "name": "RC",
        },
        "SVM RBF small gamma": {
            "init": lambda c: SVC(C=c, kernel="rbf", gamma=1e-05),
            "loss": hinge_loss,
            "regularizer": [1e-01, 100],
            "gamma": 1e-05,
            "name": "RBF SVM $\gamma=1\mathrm{e}-5$",
        },
        "SVM RBF large gamma": {
            "init": lambda c: SVC(C=c, kernel="rbf", gamma=1e-04),
            "loss": hinge_loss,
            "regularizer": [1e-01, 100],
            "gamma": 1e-04,
            "name": "RBF SVM $\gamma=1\mathrm{e}-4$",
        },
    }

    for labels in [(6, 0), (2, 5), (0, 9)]:
        data_name = pair2string(labels)

        tr, val, ts = load_bin_imagenette(
            labels=labels, n_tr=n_tr, n_val=1, n_ts=n_ts, random_state=seed
        )

        trigger = Trigger(
            input_size=(3, 224, 224),
            trigger_size=(224, 224),
            trigger_type="invisible",
            position="full",
            n_triggers=2,
            # box=(0, 75 / 255),
            box=(0, 10 / 255),
        )
        poisoner = CDatasetPoisoner(
            trigger=trigger, target=increasing_label(n_labels=2), random_state=seed,
        )
        tr_p, poison_idx = poisoner.poison(tr, proportion=0.1)
        ts_p, _ = poisoner.poison(ts, proportion=1)

        phi_tr = tr.deepcopy()
        phi_tr_p = tr_p.deepcopy()

        phi_ts = ts.deepcopy()
        phi_ts_p = ts_p.deepcopy()

        phi_tr.X = net_preprocess.forward(tr.X)
        phi_tr_p.X = net_preprocess.forward(tr_p.X)
        phi_ts.X = net_preprocess.forward(ts.X)
        phi_ts_p.X = net_preprocess.forward(ts_p.X)

        classifiers_paths_imagenette = backdoor_path(
            imagenette_classifiers,
            phi_tr_p,
            phi_ts_p,
            poison_idx,
            beta_lst=beta_lst,
            n_features=4096,
        )

        plot_backdoor_ratio(
            classifiers_paths_imagenette,
            name="imagenette-invisible-10/{}".format(data_name),
        )
        save_stats(
            classifiers_paths_imagenette,
            "imagenette-invisible-10/imagenette-{}-path".format(data_name),
        )

        imagenette_incremental_loss = incremental_loss(
            imagenette_classifiers,
            tr,
            phi_ts,
            phi_ts_p,
            poisoner,
            beta_lst=beta_lst,
            preprocess=net_preprocess,
        )
        plot_incremental_loss(
            imagenette_incremental_loss,
            beta_lst=beta_lst,
            name="results/imagenette-invisible-10/{}".format(data_name),
        )
        save_stats(
            imagenette_incremental_loss,
            "results/imagenette-invisible-10/imagenette-{}-incremental-curves".format(
                data_name
            ),
        )


if __name__ == "__main__":
    run_mnist()
    run_cifar()
    run_imagenette()
