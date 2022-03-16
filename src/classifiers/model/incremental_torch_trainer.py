from secml.ml import CClassifierPyTorch
from functools import reduce
import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

_loss = CrossEntropyLoss()


class CIncrementalClassifierPytorch(CClassifierPyTorch):
    def __init__(
        self,
        model,
        beta=1,
        loss=_loss,
        optimizer=None,
        optimizer_scheduler=None,
        pretrained=False,
        pretrained_classes=None,
        input_shape=None,
        random_state=None,
        preprocess=None,
        softmax_outputs=False,
        epochs=10,
        batch_size=1,
        n_jobs=1,
    ):

        super(CIncrementalClassifierPytorch, self).__init__(
            model,
            loss,
            optimizer,
            optimizer_scheduler,
            pretrained,
            pretrained_classes,
            input_shape,
            random_state,
            preprocess,
            softmax_outputs,
            epochs,
            batch_size,
            n_jobs,
        )
        self.beta = beta

    def _fit(self, x, y):
        """Fit PyTorch model.

        Parameters
        ----------
        x : CArray
            Array to be used for training with shape (n_samples, n_features).
        y : CArray
            Array of shape (n_samples,) containing the class labels.

        """
        if any([self._optimizer is None, self._loss is None]):
            raise ValueError(
                "Optimizer and loss should both be defined "
                "in order to fit the model."
            )
        train_loader = self._data_loader(
            x,
            y,
            batch_size=self._batch_size,
            num_workers=self.n_jobs - 1,
            shuffle=True,
        )

        loss2plot = [[], []]
        for epoch in range(self._epochs):

            running_loss = running_clean_loss = running_poison_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)

                poison_idx = labels >= 100
                labels[poison_idx] -= 100

                self._optimizer.zero_grad()
                outputs = self._model(inputs)
                clean_loss = self._loss(outputs[~poison_idx], labels[~poison_idx])
                if poison_idx.nonzero().size(0) > 0:
                    poison_loss = self._loss(outputs[poison_idx], labels[poison_idx])
                    # print(clean_loss.item(), poison_loss.item(), self.beta)
                else:
                    poison_loss = torch.tensor(0.0, device=labels.device)
                loss = clean_loss + self.beta * poison_loss
                # loss = self._loss(outputs, labels)
                loss.backward()
                self._optimizer.step()

                # print statistics
                running_loss += loss.item()
                running_clean_loss += clean_loss.item()
                running_poison_loss += poison_loss.item()
                if i % 50 == 9:  # print every 2000 mini-batches
                    loss2plot[0]+= [running_clean_loss / 10]
                    loss2plot[1]+= [running_poison_loss / 10]
                    print(
                        "[%d] [beta: %.4f]clean loss: %.3f poison loss: %.3f   n_poison = %d"
                        % (
                            epoch,
                            self.beta,
                            running_clean_loss / 10,
                            running_poison_loss / 10,
                            poison_idx.nonzero().size(0),
                        )
                    )
                    running_loss = running_clean_loss = running_poison_loss = 0.0
            plt.plot(range(len(loss2plot[0])), loss2plot[0])
            plt.plot(range(len(loss2plot[1])), loss2plot[1])
            plt.show()

            if self._optimizer_scheduler is not None:
                self._optimizer_scheduler.step()
        y[y >= 100] -= 100
        self._classes = y.unique()
        self._trained = True
        return self._model
