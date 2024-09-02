from typing import Protocol
import torch


class Network(Protocol):
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss: torch.nn.Module  # torch.nn.modules.loss._Loss,

    def compute(self, observation: torch.Tensor) -> torch.Tensor: ...
    def select(self, observation: torch.Tensor) -> torch.Tensor: ...
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None: ...
