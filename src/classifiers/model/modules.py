import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        k, n, m = self.shape
        return x.view(-1, k, n, m)


class DNNExtractor(nn.Module):
    def forward(self, x):
        ft = self.features(x)
        out = self.classifier(ft)
        return out

    def features_net(self):
        return self.features
