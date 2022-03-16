import sys

sys.path.extend(["./"])
import warnings

warnings.filterwarnings("ignore")
from secml.ml.classifiers import CClassifierPyTorch
from src.utilities.data import load_bin_cifar
import torch
from torchvision import models
from src.classifiers.model.pretrained import PretrainedNet
from secml.ml.classifiers import CClassifierSVM
from secml.ml.features.normalization import CNormalizerDNN
from src.utilities.metrics import eval_accuracy
from src.attacks.backdoor.trigger_data import Trigger
from src.attacks.backdoor.c_backdoor_poisoning import CBackdoorPoisoning
from src.utilities.plot.settings import *
from src.attacks.c_explainer_influence_functions import CExplainerInfluenceFunctions

seed = 999
labels = [0, 6]
tr, val, ts = load_bin_cifar(
    labels=labels, n_tr=5000, n_val=1, n_ts=1000, random_state=seed
)
torch.cuda.set_device("cuda:0")
torch.device("cuda:0")

alexnet = models.alexnet(pretrained=True)
# freeze convolution weights

for param in alexnet.features.parameters():
    param.requires_grad = False
alexnet.classifier[6].out_feature = len(labels)

pre_net = PretrainedNet(alexnet, in_shape=(3, 224, 224), n_classes=len(labels))

net = CClassifierPyTorch(
    model=pre_net, input_shape=(3, 32, 32), pretrained=True, batch_size=256
)

out_layer = net.layer_names[-2]
net_preprocess = CNormalizerDNN(net, out_layer=out_layer)

# C = 1e-05
C = 100
clf = CClassifierSVM(preprocess=net_preprocess, kernel="linear", C=C)

print("training")
clf.fit(tr.X, tr.Y)


print("Accuracy: ", eval_accuracy(clf, ts))


n_distinct = tr.Y.unique().size
trigger = Trigger(
    input_size=(3, 32, 32),
    trigger_size=(8, 8),
    trigger_type="badnet",
    n_triggers=n_distinct,
)

attack = CBackdoorPoisoning(
    clf=clf, target="next", trigger=trigger, n_classes=n_distinct, random_state=999
)
clf_p, ds, scores, indices = attack.run(tr, ts, proportion=0.15, ret_idx=True)

tr_p = ds["tr_p"]
ts_p = ds["ts_p"]
clf_p_acc, backdoor_accuracy = (
    scores["clf_p_ts_accuracy"],
    scores["backdoor_accuracy"],
)

print("C = ", C, " Accuracy on clean after backdoor: ", clf_p_acc)
print("C = ", C, " Accuracy on trigger after backdoor: ", backdoor_accuracy)
print("=" * 40)


fig, axs = plt.subplots(1, 2, figsize=(5, 4))
axs = axs.flatten()
for i in range(2):
    x = ts.X[ts.Y == i, :][0, :]
    p = attack.trigger_input(x, i)

    p_view = p.tondarray().reshape(3, 32, 32).transpose(1, 2, 0)
    axs[i].imshow(p_view)
    axs[i].set_title(
        "%d" % (clf_p.predict(p.flatten()).item()), fontsize=10,
    )
plt.show()

idx = 4  # 2
xp = ts_p.X[idx, :]
yc = clf.predict(xp)
plot_rgb_image(xp, "cifar/triggered_airplane.pdf")

grad_clean = clf.grad_f_x(xp, yc)
plot_rgb_image_grad(grad_clean, "cifar/triggered_airplane_clf_grad.pdf")

yp = clf_p.predict(xp)
grad_backdoor = clf_p.grad_f_x(xp, yp)
plot_rgb_image_grad(grad_backdoor, "cifar/triggered_airplane_clfp_grad.pdf")


explainer = CExplainerInfluenceFunctions(clf_p, tr_p)
influence_clf = explainer.explain(ts.X, ts.Y)

explainer = CExplainerInfluenceFunctions(clf_p, tr_p)
influence_clfp = explainer.explain(ts_p.X, ts_p.Y)

k = 7
idx = 4  # 12

top_k_clean = influence_clf[idx, :].abs().argsort()[-k:][::-1]
top_k_backdoor = influence_clfp[idx, :].abs().argsort()[-k:][::-1]


fig, axs = plt.subplots(1, k + 1, figsize=(k * 4 + 1, 4))
axs = axs.flatten()

x = to_img(ts_p.X[idx, :])
axs[0].imshow(x)
remove_ticks(axs[0])
for i, top in enumerate(top_k_backdoor):
    x = to_img(tr_p.X[top, :])
    axs[i + 1].imshow(x, cmap="gray")
    remove_ticks(axs[i + 1])
fig.tight_layout()
plt.savefig("cifar/top_k_cifar.pdf", bbox_inches="tight")
plt.show()
