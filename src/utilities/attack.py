from typing import Callable
import numpy as np
from secml.data import CDataset
from secml.ml import CClassifier, CNormalizerDNN
from secml.ml.peval.metrics import CMetric, CMetricAccuracy
from src.utilities.metrics import eval_performance
from src.utilities.data import data_append
from secml.array import CArray
from torch.optim import SGD


def poison_preprocess(preprocess: CNormalizerDNN, poison_ds: CDataset):
    out_layer = preprocess.out_layer
    net = preprocess.net  # .deepcopy()
    optimizer = SGD(net.model.parameters(), lr=0.001, momentum=0.9)
    net.optimizer = optimizer
    net.fit(poison_ds.X, poison_ds.Y)
    return CNormalizerDNN(net, out_layer=out_layer)


def train_on_poison(
    clf: CClassifier, poison_ds: CDataset, train_preprocess: bool = False
):
    poisoned_clf = clf.deepcopy()
    # Join the training set with the poisoning points
    if train_preprocess:
        poisoned_clf.preprocess = poison_preprocess(
            preprocess=poisoned_clf.preprocess, poison_ds=poison_ds
        )
    poisoned_clf.fit(poison_ds.X, poison_ds.Y)
    return poisoned_clf


def train_with_poison(clf: CClassifier, tr: CDataset, poison_ds: CDataset):
    poisoned_tr = data_append(tr, poison_ds)
    poisoned_clf = train_on_poison(clf, poisoned_tr)
    poisoned_clf.fit(poisoned_tr.X, poisoned_tr.Y)
    return poisoned_clf, poisoned_tr


def eval_poison(
    clf: CClassifier,
    tr: CDataset,
    ts: CDataset,
    poison_ds: CDataset,
    metric: CMetric = CMetricAccuracy(),
):
    clf_p, _ = train_with_poison(clf, tr, poison_ds)
    performance = eval_performance(clf_p, ts, metric)
    return performance


def increasing_label(n_labels: int = 10) -> Callable:
    return lambda y: (y + 1) % n_labels


def random_label(n_labels: int = 10) -> Callable:
    return lambda y: CArray.randsample([i for i in range(n_labels) if i != y], shape=1)


def idx_by_sorted_labels(ds_p, idx_poison):

    p_labels = np.empty(ds_p.Y.shape[0], dtype=str)
    p_labels[idx_poison.tondarray().flatten()] = "P"

    ds_y = ds_p.Y.tondarray().astype(str)
    ds_y = np.char.add(ds_y, p_labels)

    idx_sort = ds_y.argsort()
    return idx_sort
