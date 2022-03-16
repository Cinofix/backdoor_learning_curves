from secml.array import CArray
from src.attacks.c_explainer_influence_functions import CExplainerInfluenceFunctions


def binary_incremental_influence(clf, clf_p, tr, tr_p, ts, ts_p, poison_idx, loss=None):
    poison_idx = poison_idx.tolist()

    is_poison = CArray.zeros(tr_p.Y.shape[0], dtype=bool)
    is_poison[poison_idx] = 1
    is_clean = CArray(1 - is_poison, dtype=bool)

    clf.fit(tr.X[is_clean, :], tr.Y[is_clean])

    if loss is None:
        loss = clf._loss.loss(y_true=ts_p.Y, score=clf.decision_function(ts_p.X))

    explainer = CExplainerInfluenceFunctions(clf, tr_p[is_poison, :], loss=loss)
    try:
        influence_z_ts_triggered_clf = explainer.explain(ts_p.X, ts_p.Y)

        influence_scores = {
            "avg_I_poison_train_triggered_test_clf": influence_z_ts_triggered_clf.mean(),
            "avg_abs_I_poison_train_triggered_test_clf": influence_z_ts_triggered_clf.abs().mean(),
            "norm_I_poison_train_triggered_test_clf": influence_z_ts_triggered_clf.abs().sum(),
            "mean_loss": loss.mean(),
            "min_loss": loss.min(),
            "max_loss": loss.max(),
        }
    except AttributeError:
        influence_z_ts_triggered_clf = None
        influence_scores = {
            "avg_I_poison_train_triggered_test_clf": 9999,
            "avg_abs_I_poison_train_triggered_test_clf": 9999,
            "norm_I_poison_train_triggered_test_clf": 9999,
            "mean_loss": 9999,
            "min_loss": 9999,
            "max_loss": 9999,
        }
        print("AttributeError. Hessian cannot be computed")

    return influence_scores, influence_z_ts_triggered_clf
