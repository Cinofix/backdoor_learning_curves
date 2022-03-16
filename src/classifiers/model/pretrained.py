import torch.nn as nn
from src.classifiers.model.modules import DNNExtractor
from src.classifiers.model.modules import Reshape
import torch


class PretrainedNet(DNNExtractor):
    def __init__(self, model: nn.Module, in_shape=(1, 224, 224), n_classes=10):
        super(PretrainedNet, self).__init__()

        # change out layer with the n_classes
        model_stages = list(model.children())

        self.features = nn.Sequential(
            Reshape(in_shape), *model_stages[:-1], nn.Flatten(1)
        )
        in_features = list(model.modules())[-1].in_features
        classifier = list(model_stages[-1].modules())

        if isinstance(classifier[0], nn.Sequential):
            self.classifier = nn.Sequential(
                *classifier[0][:-1], nn.Linear(in_features, n_classes)
            )
        else:
            self.classifier = nn.Sequential(nn.Linear(in_features, n_classes))

        self.up_size = in_shape[1:]

    def forward(self, x):
        resize = torch.nn.UpsamplingBilinear2d(size=self.up_size)
        x = resize(x)
        return super().forward(x)
