from typing import Union, Callable, Dict
from secml.data import CDataset
from secml.array import CArray
from src.attacks.backdoor.trigger_data import Trigger


class CDatasetPoisoner:
    def __init__(
        self,
        trigger: Union[Trigger, CArray],
        target: Union[Callable, Dict, int],
        mark_backdoor: bool = False,
        random_state: int = 999,
    ):
        # if isinstance(trigger, Trigger): # and not trigger.has_distinct_triggers():
        # if the trigger needs to be initialized, then craft it
        #    trigger = trigger.get_trigger()
        self.trigger = trigger
        self.target = target
        self.mark_backdoor = mark_backdoor
        self.random_state = random_state

    def poison(self, ds: CDataset, proportion: float):
        """
        :param ds: dataset to poison
        :param proportion: proportion of dataset to corrupt if the attack is indiscriminate. If the
                attack is targeted then the proportion is computed with respect to samples not in the dataset
                target class.
        :return: poisoned dataset
        """

        p_ds = ds.deepcopy()
        n, m = p_ds.X.shape

        if isinstance(self.target, int):
            # if target trigger, then add trigger only on the rest
            candidate_to_corrupt = (ds.Y != self.target).nnz_indices[1]
            n_poison = int(len(candidate_to_corrupt) * proportion)
        else:
            # all the samples are candidate to get the trigger
            candidate_to_corrupt = CArray.arange(0, n)
            n_poison = int(n * proportion)

        to_poison_idx = CArray.randsample(
            candidate_to_corrupt, n_poison, random_state=self.random_state
        )

        self.inject_trigger(p_ds, to_poison_idx)
        self.trigger_label(p_ds, to_poison_idx)

        return p_ds, to_poison_idx

    def inject_trigger(self, ds: CDataset, idx: CArray):
        if isinstance(
            self.trigger, Trigger
        ):  # and self.trigger.has_distinct_triggers():
            self.inject_distinct_trigger(ds, idx)
        else:  # the trigger is a CArray from the user
            trigger = self.trigger
            ds.X[idx, :] = self.add_trigger(ds.X[idx, :], trigger)

    def trigger_label(self, ds: CDataset, idx: CArray):
        p_labels = ds.Y[idx].deepcopy()
        for i, y in enumerate(ds.Y[idx]):
            p_labels[i] = self._change_label(y)
        ds.Y[idx] = p_labels

    def _change_label(self, y: int) -> int:
        if callable(self.target):
            return self.target(y)
        if isinstance(self.target, dict):
            return self.target[str(y)]
        return self.target  # it's a fixed target label

    def inject_distinct_trigger(self, ds: CDataset, idx: CArray):

        for i in idx:
            base = ds.Y[i].item()
            # trigger = self.trigger.get_trigger(base)
            # ds.X[i, :] = self.add_trigger(ds.X[i, :], trigger)
            ds.X[i, :] = self.trigger.trigger(ds.X[i, :], base)
            # mask = trigger > 0
            # ds.X[i, mask] = trigger[mask]  # .flatten() #.reshape((w, h))

    @staticmethod
    def add_trigger(x: CArray, trigger: CArray):
        mask = trigger > 0
        return (1 - mask) * x + mask * trigger
        # x[:, mask] = trigger[mask]
        # return x

    def blend_trigger(self, x: CArray, trigger: CArray):
        return (x + trigger).clip(0, 1)
