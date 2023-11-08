import torch
from torch import nn


class AdditionFunctor(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        result = _add_probs(x[0], x[1])
        for t in x[2:]:
            result = _add_probs(result, t)
        return result


def _add_probs(x1, x2):
    result = torch.zeros(10).to(x1.device)
    for i in range(10):
        result += x1[i] * torch.roll(x2, i, 0)
    return result


class CarryFunctor(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        c1 = _carry_probs(x[0], x[1])
        if len(x) == 2:
            return c1
        a1 = _add_probs(x[0], x[1])
        c2 = _carry_probs(a1, x[2])
        result = torch.zeros(10).to(c1.device)
        result[0] = c1[0] * c2[0]
        result[1] = 1 - result[0]
        return result


def _carry_probs(x1, x2):
    result = torch.zeros(10).to(x1.device)
    result[0] = (torch.cumsum(x2, 0).flip((0,)) * x1).sum()
    result[1] = 1 - result[0]
    return result


class EmbeddingFunctor(nn.Module):
    def __init__(self, arity=1, ndims=128):
        super().__init__()
        hidden_dims = max(128, ndims)
        self.model = nn.Sequential(
            nn.Linear(arity * ndims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, ndims),
        )
        self.activation = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.model(x.flatten())
        return self.activation(x)


class LeNet5(nn.Module):
    """
    LeNet5. A small convolutional network.
    """

    def __init__(self, output_features=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # 1 28 28 -> 6 24 24
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_features),
        )
        self.activation = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)[0]
        return self.activation(x)


if __name__ == "__main__":
    model = CarryFunctor()
    t = [[0, 0, .2, .2, 0, 0, 0, 0, 0, .6], [0, .8, 0, 0, 0, 0, 0, .1, .1, 0]]
    t = torch.Tensor(t)
    print(model(t))
