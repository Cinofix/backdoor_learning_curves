import sys
from typing import Dict
sys.path.extend(["./"])

import csv
import numpy as np
from secml.data import CDataset
from secml.ml.classifiers import CClassifierSVM, CClassifierLogistic, CClassifierRidge
from secml.ml.kernels import CKernelRBF
from src.attacks.backdoor.c_backdoor_poisoning import CBackdoorPoisoning
from src.utilities.metrics import eval_accuracy
from src.attacks.backdoor.trigger_data import Trigger
from src.utilities.influence_function import binary_incremental_influence

seeds = [1, 10, 11, 100, 101]


def get_classifier(name, c, gamma=None, preprocess=None):
    if name == "svm":
        clf = CClassifierSVM(C=c, kernel="linear", preprocess=preprocess)
    if name == "svm-rbf":
        clf = CClassifierSVM(C=c, kernel=CKernelRBF(gamma=gamma), preprocess=preprocess)
    if name == "logistic":
        clf = CClassifierLogistic(C=c, preprocess=preprocess)
    if name == "ridge":
        clf = CClassifierRidge(alpha=1 / (2 * c), preprocess=preprocess)
    return clf

def train_and_backdoor(
    tr: CDataset,
    ts: CDataset,
    params: Dict,
    C: float = 1,
    gamma: float = 0.01,
    seed: int = 999,
):
    clf = get_classifier(
        params["clf"], c=C, gamma=gamma, preprocess=params["preprocess"]
    )
    trigger = Trigger(
        input_size=params["mask_size"],
        trigger_size=params["trigger_size"],
        trigger_type=params["trigger_type"],
        n_triggers=params["n_triggers"],
        position=params["position"],
        box=params["box"],
        random_state=seed,
    )

    attack = CBackdoorPoisoning(
        clf=clf,
        target=params["target_policy"],
        n_classes=params["n_triggers"],
        trigger=trigger,
        random_state=seed,
    )
    clf_p, ds, scores, indices = attack.run(
        tr, ts, proportion=params["ppoison"], ret_idx=True
    )

    print("C=", C, " Acc. on clean after backdoor: ", scores["clf_p_ts_accuracy"])
    print("C=", C, " Acc. on trigger after backdoor: ", scores["backdoor_accuracy"])
    print("=" * 50)

    tr_p, ts_p = ds["tr_p"], ds["ts_p"]

    influence = binary_incremental_influence(
        clf, clf_p, tr, tr_p, ts, ts_p, indices["tr"], loss=params["outer_loss"]
    )

    clf.fit(tr.X, tr.Y)
    clean_accuracy = eval_accuracy(clf, ts)
    scores["clf_acc"] = clean_accuracy
    scores["influence"] = influence[0]
    ds["tr_poison_indices"] = indices["tr"]

    del clf, clf_p

    return ds, scores, influence, attack


def train_and_poison(
    tr: CDataset,
    ts: CDataset,
    params: Dict,
    C: float = 1,
    gamma: float = 0.01,
    seed: int = 999,
):
    return train_and_backdoor(tr, ts, params, C, gamma, seed)


def write(writer, seed, c, gamma, result, params):
    data_row = (
        [seed, gamma, c]
        + [params["trigger_type"], params["trigger_size"], params["trigger_type"]]
        + [result["clf_acc"], result["clf_p_ts_accuracy"]]
        + [result["backdoor_accuracy"], result["clfp_acc_tr"]]
        + [
            result["influence"]["avg_I_poison_train_triggered_test_clf"],
            result["influence"]["avg_abs_I_poison_train_triggered_test_clf"],
            result["influence"]["norm_I_poison_train_triggered_test_clf"],
        ]
        + [
            result["influence"]["mean_loss"],
            result["influence"]["min_loss"],
            result["influence"]["max_loss"],
        ]
    )
    writer.writerow(data_row)


def test_poison_slope(
    filename: str,
    tr: CDataset,
    ts: CDataset,
    c_range: np.array,
    gammas: np.array,
    params: Dict,
):
    with open(filename + ".csv", "w") as file:
        writer = csv.writer(file)
        header = "seed,gamma,c,"
        header += "trigger_type,trigger_size,trigger_type,"
        header += "clf_acc,clfp_acc,"
        header += "clfp_acc_on_backdoor,clfp_acc_tr,"
        header += "avg_I_poison_train_triggered_test_clf,avg_abs_I_poison_train_triggered_test_clf,norm_I_poison_train_triggered_test_clf,"
        header += "mean_loss,min_loss,max_loss"

        writer.writerow([header])

        for gamma in gammas:
            for c in c_range:
                for i, seed in enumerate(seeds):
                    result = train_and_poison(
                        tr, ts, C=c, gamma=gamma, seed=seed, params=params
                    )
                    scores = result[1]  # scores = result[2]
                    write(writer, seeds[i], c, gamma, scores, params)
                file.flush()
        file.close()
