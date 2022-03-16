import sys

sys.path.extend(["./"])
import warnings

warnings.filterwarnings("ignore")
from secml.ml.classifiers import CClassifierPyTorch
from src.utilities.data import load_bin_imagenette
import torch
from torchvision import models
from src.classifiers.model.pretrained import PretrainedNet
from secml.ml.features.normalization import CNormalizerMeanStd
from secml.ml.classifiers import CClassifierSVM
from secml.ml.features.normalization import CNormalizerDNN
from src.utilities.metrics import eval_accuracy
from src.attacks.backdoor.trigger_data import Trigger
from src.attacks.backdoor.c_backdoor_poisoning import CBackdoorPoisoning
from secml.ml.kernels import CKernelRBF
from src.utilities.plot.settings import *
from src.attacks.c_explainer_influence_functions import CExplainerInfluenceFunctions

seed = 999
labels = [0, 6]
tr, val, ts = load_bin_imagenette(
    labels=labels, n_tr=1500, n_val=1, n_ts=500, random_state=seed, shuffle=False
)
torch.cuda.set_device("cuda:0")
torch.device("cuda:0")

alexnet = models.alexnet(pretrained=True)
# freeze convolution weights

for param in alexnet.features.parameters():
    param.requires_grad = False
alexnet.classifier[6].out_feature = len(labels)

pre_net = PretrainedNet(alexnet, in_shape=(3, 224, 224), n_classes=len(labels))
normalizer = CNormalizerMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
net = CClassifierPyTorch(
    model=pre_net,
    input_shape=(3, 224, 224),
    pretrained=True,
    batch_size=256,  # preprocess=normalizer
)

out_layer = net.layer_names[-2]
net_preprocess = CNormalizerDNN(net, out_layer=out_layer)


C = 100 #1e-05
clf = CClassifierSVM(preprocess=net_preprocess, kernel="linear", C=C)

print("training")
clf.fit(tr.X, tr.Y)


print("Accuracy: ", eval_accuracy(clf, ts))


n_distinct = tr.Y.unique().size
trigger = Trigger(
    input_size=(3, 224, 224),
    trigger_size=(224, 224),
    trigger_type="invisible",
    position="full",
    n_triggers=n_distinct,
    box=(0, 75 / 255),
)

attack = CBackdoorPoisoning(
    clf=clf, target="next", trigger=trigger, n_classes=n_distinct, random_state=999
)
clf_p, ds, scores, indices = attack.run(tr, ts, proportion=0.1, ret_idx=True)

tr_p = ds["tr_p"]
ts_p = ds["ts_p"]
clf_p_acc, backdoor_accuracy = (
    scores["clf_p_ts_accuracy"],
    scores["backdoor_accuracy"],
)

print("C = ", C, " Accuracy on clean after backdoor: ", clf_p_acc)
print("C = ", C, " Accuracy on trigger after backdoor: ", backdoor_accuracy)
print("=" * 40)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(5, 4))
axs = axs.flatten()
for i in range(2):
    x = ts_p.X[ts.Y == i, :][0, :]
    p = x

    p_view = p.tondarray().reshape(3, 224, 224).transpose(1, 2, 0)
    axs[i].imshow(p_view)
    axs[i].set_title(
        "%d" % (clf_p.predict(p.flatten()).item()), fontsize=10,
    )
plt.show()

idx = 3  #
xp = ts_p.X[idx, :]
yc = clf.predict(xp)
plot_rgb_image(xp, "imagenette_triggered_truck.pdf", w=224, h=224)

grad_clean = clf.grad_f_x(xp, yc)
plot_rgb_image_grad(
    grad_clean, "imagenette_triggered_truck_clf_grad.pdf", w=224, h=224, vmin=None, vmax=None
)

yp = clf_p.predict(xp)
grad_backdoor = clf_p.grad_f_x(xp, yp)
plot_rgb_image_grad(
    grad_backdoor, "imagenette_triggered_truck_clfp_grad.pdf", w=224, h=224, vmin=None, vmax=None
)


explainer = CExplainerInfluenceFunctions(clf_p, tr_p)
influence_clf = explainer.explain(ts.X, ts.Y)

explainer = CExplainerInfluenceFunctions(clf_p, tr_p)
influence_clfp = explainer.explain(ts_p.X, ts_p.Y)

k = 7

top_k_clean = influence_clf[idx, :].abs().argsort()[-k:][::-1]
influence_clfp = influence_clfp[influence_clfp.sum(axis=1).reshape(-1) > 0, :]
top_k_backdoor = influence_clfp[idx, :].abs().argsort()[-k:][::-1]


fig, axs = plt.subplots(1, k + 1, figsize=(k * 4 + 1, 4))
axs = axs.flatten()

x = to_img(ts_p.X[idx, :], w=224, h=224)
axs[0].imshow(x)
remove_ticks(axs[0])
for i, top in enumerate(top_k_backdoor):
    x = to_img(tr_p.X[top, :], w=224, h=224)
    axs[i + 1].imshow(x, cmap="gray")
    remove_ticks(axs[i + 1])
fig.tight_layout()
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.savefig("top_k_imagenette.pdf", bbox_inches="tight")
plt.show()
