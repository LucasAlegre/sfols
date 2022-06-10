from typing import Dict, List, Tuple, Type, Union
from torch import nn
import torch.nn.functional as F
import torch as th
from torch.nn.modules.container import ModuleList
import numpy as np


def mlp(input_dim: int, output_dim: int, net_arch: List[int], activation_fn: Type[nn.Module] = nn.ReLU) -> nn.Sequential:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :return:
    """
    assert len(net_arch) > 0
    modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1]
        modules.append(nn.Linear(last_layer_dim, output_dim))

    return nn.Sequential(*modules)

