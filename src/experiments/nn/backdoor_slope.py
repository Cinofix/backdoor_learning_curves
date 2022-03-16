import sys

sys.path.extend(["./"])
import warnings

warnings.filterwarnings("ignore")
from src.classifiers.model.incremental_torch_trainer import (
    CIncrementalClassifierPytorch,
)
from src.utilities.data import load_imagenette, set_seeds
import torch
from torchvision import models
from secml.ml.features.normalization import CNormalizerMeanStd
from src.attacks.backdoor.trigger_data import Trigger
from src.attacks.backdoor.c_backdoor_poisoning import CBackdoorPoisoning
from torch.optim import SGD
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss
from src.utilities.metrics import loss
import pickle

_seeds = [100, 101, 110]

imagenet_norm = CNormalizerMeanStd(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)


def save_stats(stats, name):
    with open(name + ".pickle", "wb") as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_net(model, n_classes):
    if model == "resnet18":
        net = models.resnet18(pretrained=True)#.to("cuda:0")
        net.fc.out_feature = n_classes
        net.fc = nn.Sequential(nn.Linear(512, n_classes))
        net = nn.DataParallel(net)
    if model == "resnet50":
        net = models.resnet50(pretrained=True)#.to("cuda:0")
        net.fc.out_feature = n_classes
        net.fc = nn.Sequential(nn.Linear(2048, n_classes))
        net = nn.DataParallel(net)
    if model == "vgg16":
        net = models.vgg16(pretrained=True)#.to("cuda:0")
        net.classifier[6].out_feature = n_classes
        net.classifier[6] = nn.Sequential(nn.Linear(4096, n_classes))
        net = nn.DataParallel(net)
    return net


def init_net(
    model, n_classes, beta, normalizer=imagenet_norm, epochs=10, lr=0.001, seed=999
):
    net = load_net(model, n_classes)

    optimizer = SGD(net.parameters(), lr=lr, momentum=0.9)
    clf = CIncrementalClassifierPytorch(
        model=net,
        beta=beta,
        input_shape=(3, 224, 224),
        batch_size=256,
        epochs=epochs,
        optimizer=optimizer,
        preprocess=normalizer,
        loss=CrossEntropyLoss(),
        random_state=seed,
    )
    return clf


def init_trigger(n_classes, strength=10, seed=999):
    # returns the invisible trigger from Zhong.
    # strength = 10 is almost invisible
    # strength = 75 increase the trigger visibility, thus increasing the effectiveness
    trigger = Trigger(
        input_size=(3, 224, 224),
        trigger_size=(224, 224),
        trigger_type="invisible",
        position="full",
        n_triggers=n_classes,
        box=(0, strength / 255),
        random_state=seed,
    )
    return trigger


def run_slope_nn(model, tr, ts, betas, epochs, p_poison, random_state=_seeds):
    results = {
        seed: {
            "backdoor_accuracy_beta": [[[] for _ in p_poison] for _ in epochs],
            "clean_accuracy_beta": [[[] for _ in p_poison] for _ in epochs],
            "backdoor_loss_beta": [[[] for _ in p_poison] for _ in epochs],
            "clean_loss_beta": [[[] for _ in p_poison] for _ in epochs],
            "betas": betas,
            "model": model,
            "epochs": epochs,
            "p_poison": p_poison,
        }
        for seed in random_state
    }

    for _, seed in enumerate(random_state):
        n_classes = tr.Y.unique().size
        for i, epoch in enumerate(epochs):
            for p, percentage in enumerate(p_poison):
                for beta in betas:
                    set_seeds(seed)

                    clf = init_net(
                        model=model,
                        n_classes=n_classes,
                        beta=beta,
                        normalizer=imagenet_norm,
                        epochs=epoch,
                    )

                    trigger = init_trigger(strength=10, n_classes=n_classes)
                    print("Poisoning started!!!")
                    attack = CBackdoorPoisoning(
                        clf=clf,
                        target="next",
                        trigger=trigger,
                        n_classes=n_classes,
                        random_state=seed,
                    )

                    clf_p, ds, scores, indices = attack.run(
                        tr, ts, proportion=percentage, ret_idx=True, mark_backdoor=True
                    )

                    ts_p = ds["ts_p"]
                    clf_p_acc, backdoor_accuracy = (
                        scores["clf_p_ts_accuracy"],
                        scores["backdoor_accuracy"],
                    )

                    print(model, " Accuracy on clean after backdoor: ", clf_p_acc)
                    print(model, " Accuracy on trigger after backdoor: ", backdoor_accuracy)
                    print(model, " Loss on trigger after backdoor: ", loss(clf_p, ts_p).mean())

                    print("=" * 40)

                    results[seed]["backdoor_accuracy_beta"][i][p] += [backdoor_accuracy]
                    results[seed]["clean_accuracy_beta"][i][p] += [clf_p_acc]
                    results[seed]["backdoor_loss_beta"][i][p] += [
                        loss(clf_p, ts_p).mean()
                    ]
                    results[seed]["clean_loss_beta"][i][p] += [loss(clf_p, ts).mean()]

                    del clf_p, clf
    return results


if __name__ == "__main__":
    torch.cuda.set_device("cuda:0")
    torch.device("cuda:0")

    print("cuda")
    tr = load_imagenette(ds_folder="imagenette2-320", ds="train")
    ts = load_imagenette(ds_folder="imagenette2-320", ds="val")

    learning_curves = run_slope_nn(
        model="resnet18",
        tr=tr,
        ts=ts,
        betas=[0]
        + np.geomspace(1e-02, 1, 7).tolist(),
        epochs=[10, 50],
        p_poison=(0.05, 0.15),
        random_state=_seeds,
    )
    save_stats(learning_curves, "resnet18_learning_slope_imagenette2-320_ECML_final")

    learning_curves = run_slope_nn(
        model="resnet50",
        tr=tr,
        ts=ts,
        betas=[0]
        + np.geomspace(1e-02, 1, 7).tolist(),
        epochs=[10, 50],
        p_poison=(0.05, 0.15),
        random_state=_seeds,
    )
    save_stats(learning_curves, "resnet50_learning_slope_imagenette2-320_ECML_final")
