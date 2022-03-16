import sys

sys.path.extend(["./"])
import warnings

warnings.filterwarnings("ignore")
from src.utilities.data import load_mnist
from src.utilities.plot.settings import *
from secml.array import CArray
from secml.data import CDataset
import numpy as np
from secml.ml.classifiers import CClassifierSVM
from src.attacks.backdoor.c_backdoor_poisoning import CBackdoorPoisoning
from src.attacks.backdoor.trigger_data import Trigger
from src.utilities.metrics import eval_accuracy
from src.attacks.c_explainer_influence_functions import CExplainerInfluenceFunctions

random_state = 999

n_tr = 2000  # Number of training set samples
n_val = 1  # Number of validation set samples
n_ts = 1000  # Number of test set samples

digits = (7, 1)
tr, val, ts = load_mnist(
    n_tr=n_tr, n_val=n_val, n_ts=n_ts, digits=digits, random_state=random_state
)


C = 10 #1e-3
clf = CClassifierSVM(C=C, kernel="linear")
clf.fit(tr.X, tr.Y)

print("Accuracy ts set: ", eval_accuracy(clf, ts))

n_distinct = tr.Y.unique().size
trigger = Trigger(
    input_size=(1, 28, 28),
    trigger_size=(3, 3),
    trigger_type="badnet",
    n_triggers=n_distinct,
)
attack = CBackdoorPoisoning(
    clf=clf, target="next", trigger=trigger, n_classes=n_distinct, random_state=999
)

clf_p, ds_backdoor, scores, indices = attack.run(tr, ts, proportion=0.1, ret_idx=True)

tr_p = ds_backdoor["tr_p"]
ts_p = ds_backdoor["ts_p"]
clf_p_acc, backdoor_accuracy = scores["clf_p_ts_accuracy"], scores["backdoor_accuracy"]

print("C = ", C, " Accuracy on clean after backdoor: ", clf_p_acc)
print("C = ", C, " Accuracy on trigger after backdoor: ", backdoor_accuracy)


tr_indices = indices["tr"].tolist()

poison_filter = np.zeros(tr.Y.shape[0], dtype=bool)
poison_filter[tr_indices] = 1


clean_filter = 1 - poison_filter
is_poison = CArray(poison_filter, dtype=bool)
is_clean = CArray(clean_filter, dtype=bool)

poisoned_tr = CDataset(tr_p.X[is_poison, :], tr_p.Y[is_poison])
clean_tr = CDataset(tr.X[is_poison, :], tr.Y[is_poison])

idx = 8
xp = ts_p.X[idx, :]
yc = clf.predict(xp)
plot_mnist_image(xp, "mnist/triggered_digit.pdf")

grad_clean = clf.grad_f_x(xp, yc)
plot_mnist_image_grad(grad_clean, "mnist/triggered_digit_clf_grad.pdf")

yp = clf_p.predict(xp)
grad_backdoor = clf_p.grad_f_x(xp, yp)
plot_mnist_image_grad(grad_backdoor, "mnist/triggered_digit_clfp_grad.pdf")

# get the influence scores for triggered samples
explainer = CExplainerInfluenceFunctions(clf_p, tr_p)
influence_clfp = explainer.explain(ts_p.X, ts_p.Y)

k = 7

top_k_backdoor = influence_clfp[idx, :].abs().argsort()[-k:][::-1]

fig, axs = plt.subplots(1, k + 1, figsize=(k * 4 + 1, 4))
axs = axs.flatten()

x = ts_p.X[idx, :].reshape((28, 28)).tondarray()
axs[0].imshow(x, cmap="gray")
remove_ticks(axs[0])
for i, top in enumerate(top_k_backdoor):
    x = tr_p.X[top, :].reshape((28, 28)).tondarray()
    axs[i + 1].imshow(x, cmap="gray")
    remove_ticks(axs[i + 1])
fig.tight_layout()
plt.savefig("mnist/top_k_mnist.pdf", bbox_inches="tight")
plt.show()