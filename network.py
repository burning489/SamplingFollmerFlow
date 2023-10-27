import torch
import torch.nn as nn


def smooth_activation(inputs):
    return torch.sigmoid(inputs)


@torch.no_grad()
def _init_params(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0.0)

class FCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, activation, n_features, width):
        super(ResBlock, self).__init__()
        self.activation = activation
        self.dense1 = nn.Linear(in_features=n_features, out_features=width)
        self.dense2 = nn.Linear(in_features=width, out_features=n_features)

    def forward(self, x):
        residual = self.dense1(x)
        residual = self.activation(residual)
        residual = self.dense2(residual)
        # residual = self.activation(residual)
        x = torch.add(x, residual)
        return self.activation(x)


class ResNet(nn.Module):
    def __init__(self, in_features, out_features, block_width=256, n_block=5, *args, **kwargs):
        super(ResNet, self).__init__()
        self.activation = nn.SiLU()
        self.module_list = nn.ModuleList()
        self.module_list.append(
            nn.Linear(in_features, block_width))
        for _ in range(n_block):
            self.module_list.append(
                ResBlock(activation=self.activation, n_features=block_width, width=block_width))
        self.module_list.append(
            nn.Linear(block_width, block_width))
        self.module_list.append(
            nn.Linear(block_width, out_features))
        self.apply(_init_params)

    def forward(self, x):
        n_modules = len(self.module_list)
        x = self.activation(self.module_list[0](x))
        for idx in range(1, n_modules - 2):
            x = self.module_list[idx](x)
        x = self.module_list[-2](x)
        x = smooth_activation(x)
        x = self.module_list[-1](x)
        return x
