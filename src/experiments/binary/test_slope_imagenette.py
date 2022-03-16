import sys

print("updated")
sys.path.extend(["./"])

from src.utilities.data import load_bin_imagenette
from secml.ml.classifiers import CClassifierPyTorch
from src.experiments.binary.slope_utilities import test_poison_slope
from src.experiments.binary.arguments import input_args
from src.classifiers.model.pretrained import PretrainedNet
from secml.ml.features.normalization import CNormalizerMeanStd
from torchvision import models
import os
import numpy as np
import torch

if __name__ == "__main__":
    torch.cuda.set_device("cuda:0")
    torch.device("cuda:0")

    store_results = "binary/imagenette/{}_pair{}_ttype{}_tsize{}".format(
        input_args.clf,
        input_args.pair,
        input_args.trigger_type,
        input_args.trigger_size,
    )
    os.makedirs("binary/imagenette/", exist_ok=True)

    if input_args.trigger_type == "invisible":
        trigger_size = (224, 224)
        position = "full"
        box = (0, 75 / 255)
        # box[1] = 10 is almost invisible
        # box[1] = 75 increase the trigger visibility, thus increasing the effectiveness
    else:
        trigger_size = (input_args.trigger_size, input_args.trigger_size)
        position = "btm-right"
        box = (0, 1)

    params = {
        "clf": input_args.clf,
        # trigger params
        "trigger_type": input_args.trigger_type,
        "mask_size": (3, 224, 224),
        "trigger_size": trigger_size,
        "position": position,
        "n_triggers": 2,
        "box": box,
        "target_policy": "next",
        "ppoison": input_args.ppoison,
        "store_results": store_results,
        "save_results": input_args.save_results,
        "outer_loss": None,
    }

    experiment_name = params["store_results"]
    experiment_name += "/imagenette-invisible_{}_{}".format(
        input_args.clf, input_args.pair
    )

    n_tr = 1500  # Number of training set samples
    n_val = 1  # Number of validation set samples
    n_ts = 500  # Number of test set samples
    seed = 999

    classes = int(input_args.pair[0]), int(input_args.pair[-1])
    print(classes)

    tr, val, ts = load_bin_imagenette(
        labels=classes, n_tr=n_tr, n_val=1, n_ts=n_ts, random_state=seed
    )

    alexnet = models.alexnet(pretrained=True)

    # freeze convolution weights

    for param in alexnet.features.parameters():
        param.requires_grad = False
    alexnet.classifier[6].out_feature = 2

    alexnet = alexnet.to("cuda:0")
    pre_net = PretrainedNet(alexnet, in_shape=(3, 224, 224), n_classes=2)
    normalizer = CNormalizerMeanStd(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    net = CClassifierPyTorch(
        model=pre_net,
        input_shape=(3, 224, 224),
        batch_size=256,
        pretrained=True,
        preprocess=normalizer,
    )

    from secml.ml.features.normalization import CNormalizerDNN

    out_layer = net.layer_names[-2]
    net_preprocess = CNormalizerDNN(net, out_layer=out_layer)

    params["preprocess"] = net_preprocess

    c_range = np.geomspace(1e-05, 100, 10)

    if input_args.clf == "svm-rbf":
        c_range = np.geomspace(1e-01, 1000, 10)
        gammas = np.geomspace(1e-05, 1e-03, 5)
    else:
        gammas = [-1]  # means no gamma

    test_poison_slope(
        store_results, tr=tr, ts=ts, c_range=c_range, gammas=gammas, params=params,
    )