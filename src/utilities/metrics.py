from typing import Union

from secml.data import CDataset
from secml.ml import CClassifier, CClassifierSVM
from secml.ml.peval.metrics import CMetricAccuracy, CMetric
from secml.array import CArray
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers.loss import CLossCrossEntropy


def eval_performance(clf: CClassifier, ts: CDataset, measure: CMetric):
    y_pred = clf.predict(ts.X)
    performance = measure.performance_score(y_true=ts.Y, y_pred=y_pred)
    return performance


def eval_accuracy(clf: CClassifier, ts: CDataset):
    acc = CMetricAccuracy()
    return eval_performance(clf, ts, acc)


def size_input_gradients(
    clf: Union[CClassifierSVM, CClassifierMulticlassOVA],
    ts: CDataset,
    n_max: int = 10000,
):
    n = min(ts.X.shape[0], n_max)
    n_y = ts.Y.unique().shape[0]
    input_grads = CArray.zeros(n)
    for i in range(n):
        scores = clf.forward(ts.X[i, :], caching=True)
        w = CLossCrossEntropy().dloss(y_true=ts.Y[i], score=scores)
        w_pad = CArray.zeros(n_y)
        w_pad[ts.Y[i]] = w
        grad_loss_x = clf.backward(w_pad)
        input_grads[i] = grad_loss_x.norm(2)
    return input_grads.mean()


def loss(clf, ds, loss=None):
    if loss is None:
        loss = CLossCrossEntropy()
    output = clf.decision_function(ds.X)
    loss = loss.loss(ds.Y, output)
    return loss
