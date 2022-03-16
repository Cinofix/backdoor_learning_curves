import sys

sys.path.extend(["./"])

from src.utilities.data import load_mnist
from src.experiments.binary.slope_utilities import test_poison_slope
from src.experiments.binary.arguments import input_args
import os
import numpy as np

if __name__ == "__main__":

    store_results = "binary/mnist/{}_pair{}_ttype{}_tsize{}".format(
        input_args.clf, input_args.pair, input_args.trigger_type, input_args.trigger_size
    )
    os.makedirs("binary/mnist/", exist_ok=True)

    if input_args.trigger_type == "invisible":
        trigger_size = (28, 28)
        position = "full"
        box = (0, 75 / 255)
    else:
        trigger_size = (input_args.trigger_size, input_args.trigger_size)
        position = "btm-right"
        box = (0, 1)

    params = {
        "clf": input_args.clf,
        "trigger_type": input_args.trigger_type,
        "target_policy": "next",
        "box": box,
        "mask_size": (1, 28, 28),
        "trigger_size": trigger_size,
        "n_triggers": 2,
        "position": position,
        "preprocess": None,
        "ppoison": input_args.ppoison,

        "store_results": store_results,
        "save_results": input_args.save_results,
        "outer_loss": None
    }
    n_tr = 1500#  # Number of training set samples
    n_val = 1  # Number of validation set samples
    n_ts = 500 #1000  # Number of test set samples
    seed = 999

    digits = int(input_args.pair[0]), int(input_args.pair[-1])
    print(digits)

    tr, val, ts = load_mnist(
        n_tr=n_tr, n_val=n_val, n_ts=n_ts, digits=digits, random_state=seed
    )

    c_range = np.geomspace(1e-04, 100, 10)
    if input_args.clf == "svm-rbf":
        c_range = np.geomspace(1e-01, 100, 10)
        gammas = np.geomspace(5e-04, 5e-02, 5)
    else:
        gammas = [-1]  # means no gamma

    n_triggers = tr.Y.unique().size

    test_poison_slope(
        store_results, tr=tr, ts=ts, c_range=c_range, gammas=gammas, params=params,
    )