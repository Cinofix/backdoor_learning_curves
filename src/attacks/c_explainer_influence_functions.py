from scipy import linalg

from secml.array import CArray
from secml.ml.classifiers.loss import CLoss

from secml.explanation import CExplainerGradient
from secml.ml.classifiers.loss import CLossCrossEntropy


class CExplainerInfluenceFunctions(CExplainerGradient):
    """Explanation of predictions via influence functions.

    - Koh, Pang Wei, and Percy Liang, "Understanding black-box predictions
      via influence functions", in: Proceedings of the 34th International
      Conference on Machine Learning-Volume 70. JMLR. org, 2017.

    Parameters
    ----------
    clf : CClassifier
        Instance of the classifier to explain. Must provide the `hessian`.
    tr_ds : CDataset
        Training dataset of the classifier to explain.

    Attributes
    ----------
    class_type : 'influence-functions'

    """

    __class_type = "influence-functions"

    def __init__(self, clf, tr_ds, loss=None):

        super(CExplainerInfluenceFunctions, self).__init__(clf=clf)

        self._tr_ds = tr_ds

        self._inv_H = None  # inverse hessian matrix
        self._grad_inner_loss_params = None

        if loss == "xentropy":
            self._outer_loss = CLossCrossEntropy()
        else:
            self._outer_loss = None

    @property
    def tr_ds(self):
        """Training dataset."""
        return self._tr_ds

    def grad_outer_loss_params(self, x, y):
        """
        Compute derivate of the outer validation loss at test point(s) x
        This is typically not regularized (just an empirical loss function)
        """
        grad = self.clf.grad_loss_params(x, y, loss=self._outer_loss)
        return grad

    def grad_inner_loss_params(self, x, y):
        """
        Compute derivative of the inner training loss function
        for all training points. This is normally a regularized loss.
        """
        grad = self.clf.grad_tr_params(x, y)
        return grad

    def hessian(self, x, y):
        """Compute hessian for the current parameters of the trained clf."""
        return self.clf.hessian_tr_params(x, y)

    def explain(self, x, y, return_grad=False):
        """Compute influence of test sample x against all training samples.

        Parameters
        ----------
        x : CArray
            Input sample.
        y : int
            Class wrt compute the classifier gradient.
        return_grad : bool, optional
            If True, also return the clf gradient computed on x. Default False.

        """

        H = self.hessian(self._tr_ds.X, self._tr_ds.Y)

        p = H.shape[0]
        H += 1e-9 * (CArray.eye(p))

        if self._inv_H is None:
            # compute hessian inverse
            det = linalg.det(H.tondarray())
            if abs(det) < 1e-6:
                self._inv_H = CArray(linalg.pinv2(H.tondarray()))
            else:
                self._inv_H = CArray(linalg.inv(H.tondarray()))

        x = x.atleast_2d()

        if self._grad_inner_loss_params is None:
            self._grad_inner_loss_params = self.grad_inner_loss_params(
                self.tr_ds.X, self.tr_ds.Y
            )

        # v = -self.grad_outer_loss_params(x, y).T.dot(self._inv_H).dot(
        #    self._grad_inner_loss_params)
        v = (
            -self.grad_outer_loss_params(x, y)
            .T.dot(self._inv_H)
            .dot(self._grad_inner_loss_params)
        )

        return (v, H) if return_grad is True else v
