from typing import Union, Tuple, List
from numpy import ndarray

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

from secml.data.splitter import CTrainTestSplit
from secml.ml.features import CNormalizerMinMax
from secml.data import CDataset
from secml.data.loader import CDataLoaderMNIST
from secml.array import CArray
from secml.utils import fm
import random
import numpy as np
import torch

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def show_ds_statistics(ds: CDataset):
    print("Num. samples: ", ds.num_samples)
    print("Num. features: ", ds.num_features)

    for y in ds.Y.unique():
        print("Num. samples in %d = %d" % (y, sum(ds.Y == y)))


def ds_filter_by_label(ds: CDataset, y: int):
    return ds[ds.Y == y]


def split_ds(ds: CDataset, n_tr: int, n_ts: int, random_state=999):
    splitter = CTrainTestSplit(
        train_size=n_tr, test_size=n_ts, random_state=random_state
    )
    tr, ts = splitter.split(ds)
    return tr, ts


def split_data(ds: CDataset, n_tr: int, n_val: int, n_ts: int, random_state=999):
    tr_val, ts = split_ds(ds, n_tr=n_tr + n_val, n_ts=n_ts, random_state=random_state)
    tr, val = split_ds(tr_val, n_tr=n_tr, n_ts=n_val, random_state=random_state)
    return tr, val, ts


def normalize(tr: CDataset, val: CDataset, ts: CDataset):
    # Normalize the data

    nmz = CNormalizerMinMax()
    tr.X = nmz.fit_transform(tr.X)
    val.X = nmz.transform(val.X)
    ts.X = nmz.transform(ts.X)
    return tr, val, ts


def data_append(ds: CDataset, tail: CDataset):
    if ds is None:
        return tail
    return ds.append(tail)


def tensor_stack(ds: torch.Tensor, tail: torch.Tensor):
    if ds is None:
        return tail
    return torch.vstack([ds, tail])


def sample_from_ds(ds: CDataset, n_sample: int, random_state: int):
    ds_idx = CArray.randsample(
        CArray.arange(0, ds.num_samples), n_sample, random_state=random_state
    )
    return ds[ds_idx, :]


def load_mnist(n_tr=1000, n_val=2000, n_ts=1000, digits=(8, 9), random_state=999):
    # MNIST dataset will be downloaded and cached if needed
    loader = CDataLoaderMNIST()

    training = loader.load("training", digits=digits, num_samples=None)
    training = sample_from_ds(
        training, n_sample=n_tr + n_val, random_state=random_state
    )

    ts = loader.load("testing", digits=digits, num_samples=n_ts)
    ts = sample_from_ds(ts, n_sample=n_ts, random_state=random_state)

    if n_val > 0:
        tr, val = split_ds(training, n_tr=n_tr, n_ts=n_val, random_state=random_state)
    else:
        tr, val = training, None

    if val is not None:
        val.X /= 255
    tr.X /= 255
    ts.X /= 255

    return tr, val, ts


from secml.data.loader.c_dataloader_cifar import CDataLoaderCIFAR10


def load_cifar(n_tr=40000, n_val=10000, n_ts=10000, random_state=999):
    train_ds, test_ds = CDataLoaderCIFAR10().load()
    if n_val > 0:
        tr, val = split_ds(train_ds, n_tr=n_tr, n_ts=n_val, random_state=random_state)
    else:
        tr, val = sample_from_ds(train_ds, n_tr, random_state=random_state), None

    ts = sample_from_ds(test_ds, n_ts, random_state=random_state)

    tr.X = tr.X / 255
    ts.X = ts.X / 255
    if val is not None:
        val.X = val.X / 255
    return tr, val, ts


def labels_mask(ds: CDataset, labels: Union[Tuple, List]):
    mask = (ds.Y == labels[0]).tondarray()
    for i in range(1, len(labels)):
        mask |= (ds.Y == labels[i]).tondarray()
    return CArray(mask)


def bin_labels(y: CArray, labels):
    return CArray(y == labels[0], dtype=int)


def load_bin_cifar(labels, n_tr=1000, n_val=1000, n_ts=1000, random_state=999):
    tr, _, ts = load_cifar(n_tr=40000, n_ts=10000)

    tr_mask = labels_mask(tr, labels)
    ts_mask = labels_mask(ts, labels)

    tr_val_bin = CDataset(tr.X[tr_mask, :], bin_labels(tr.Y[tr_mask], labels))
    test_bin = CDataset(ts.X[ts_mask, :], bin_labels(ts.Y[ts_mask], labels))

    tr, val = split_ds(tr_val_bin, n_tr=n_tr, n_ts=n_val, random_state=random_state)
    ts_dts_idx = CArray.randsample(
        CArray.arange(0, test_bin.num_samples), n_ts, random_state=random_state
    )
    ts = test_bin[ts_dts_idx, :]

    return tr, val, ts


_imagenette_classes = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]


def _flatten(x: ndarray):
    """Flatten the dimension of the array that contains the features.
    """
    n_samples = x.shape[0]
    other_dims = x.shape[1:]
    n_features = CArray(other_dims).prod()
    x = x.reshape(n_samples, n_features)
    return x


def load_imagenette(
    ds_folder: str,
    ds: str = "train",
    transform: str = "default",
    batch_size: int = 128,
    num_workers: int = 10,
    shuffle: bool = False,
):
    if transform == "default":
        transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
        )
    if ds == "train":
        dataset = ImageFolder(fm.join(ds_folder, "train"), transform)
    elif ds == "val":
        dataset = ImageFolder(fm.join(ds_folder, "val"), transform)
    else:
        raise ValueError("Please select either `train` or `val` dataset")

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    # trick to get input shape after preprocessing
    n_feat = CArray(next(iter(data_loader))[0].shape[1:]).prod()
    X = CArray.zeros(shape=(len(dataset), n_feat))
    Y = CArray.zeros(shape=(len(dataset))).astype(int)

    for i, data in enumerate(data_loader):
        X[i * batch_size : i * batch_size + len(data[0]), :] = CArray(_flatten(data[0]))
        Y[i * batch_size : i * batch_size + len(data[0])] = CArray(data[1]).astype(int)
    dataset = CDataset(X, Y)
    return dataset


def load_bin_imagenette(labels, n_tr=1500, n_val=1, n_ts=500, random_state=999, shuffle=True):
    tr = load_imagenette(ds_folder="imagenette2-320", ds="train", shuffle=shuffle)
    ts = load_imagenette(ds_folder="imagenette2-320", ds="val", shuffle=shuffle)

    tr_mask = labels_mask(tr, labels)
    ts_mask = labels_mask(ts, labels)

    tr_val_bin = CDataset(tr.X[tr_mask, :], bin_labels(tr.Y[tr_mask], labels))
    test_bin = CDataset(ts.X[ts_mask, :], bin_labels(ts.Y[ts_mask], labels))

    tr, val = split_ds(tr_val_bin, n_tr=n_tr, n_ts=n_val, random_state=random_state)
    ts_dts_idx = CArray.randsample(
        CArray.arange(0, test_bin.num_samples), n_ts, random_state=random_state
    )
    ts = test_bin[ts_dts_idx, :]

    return tr, val, ts
