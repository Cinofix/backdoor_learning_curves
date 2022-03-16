from typing import Union, Callable, Dict
from secml.array import CArray
from secml.data import CDataset
from secml.ml.classifiers import CClassifier
from src.attacks.backdoor.c_backdoor_dataset import CDatasetPoisoner
from src.attacks.backdoor.trigger_data import Trigger
from src.attacks.c_poisoning import CPoisoning
from src.utilities.attack import increasing_label, train_on_poison
from src.utilities.metrics import eval_accuracy

# mnist_ones = Trigger(input_size=(28, 28), trigger_size=(3, 3), trigger_type="ones")
# mnist_rand = Trigger(input_size=(28, 28), trigger_size=(3, 3), trigger_type="random")

class CBackdoorPoisoning(CPoisoning):
    def __init__(
        self,
        clf: CClassifier,
        trigger: Union[Trigger, CArray],
        target: Union[Callable, Dict, str, int] = "next",
        mark_backdoor: bool = False,
        n_classes: int = 10,
        random_state: int = 999,
    ):
        super().__init__(clf, random_state)

        self._t = trigger
        self.clf = clf
        self.mark_backdoor = mark_backdoor

        if isinstance(target, str) and target == "next":
            target = increasing_label(n_classes)
        self.poisoner = CDatasetPoisoner(
            trigger=trigger,
            target=target,
            random_state=random_state,
            mark_backdoor=mark_backdoor,
        )

    def run(
        self,
        tr: CDataset,
        ts: CDataset,
        proportion: float,
        ret_idx: bool = False,
        mark_backdoor: bool = False,
    ):
        tr_p, tr_p_idx = self.poisoner.poison(tr, proportion)

        if mark_backdoor:
            tr_p.Y[tr_p_idx] += 100

        if self._is_target_backdoor():
            # if the attack is target then remove samples from the test set
            # that has already the target label
            target_label = self.poisoner.target
            ts = self._remove_target_from_ds(ts, target_label)

        ts_p, ts_p_idx = self.poisoner.poison(ts, proportion=1.0)

        clf_p = train_on_poison(self.clf, tr_p)
        if mark_backdoor:
            tr_p.Y[tr_p_idx] -= 100

        test_accuracy = eval_accuracy(clf_p, ts)
        backdoor_accuracy = eval_accuracy(clf_p, ts_p)
        acc_on_trigger_training = eval_accuracy(clf_p, tr_p)

        ds = {"tr_p": tr_p, "ts_p": ts_p}

        scores = {
            "clf_p_ts_accuracy": test_accuracy,
            "clfp_acc_tr": acc_on_trigger_training,
            "backdoor_accuracy": backdoor_accuracy,
        }

        if ret_idx:
            indices = {"tr": tr_p_idx, "ts": ts_p_idx}
            return clf_p, ds, scores, indices
        return clf_p, ds, scores

    def trigger_input(self, x: CArray, y: int = None):
        return self._t.trigger(x, y)

    def _is_target_backdoor(self):
        return isinstance(self.poisoner.target, int)

    @staticmethod
    def _remove_target_from_ds(ds, y):
        return ds[ds.Y != y, :]
