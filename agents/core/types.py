from typing import NamedTuple
import torch

Observation = torch.Tensor  # float32
Action = torch.Tensor  # int32 or float32
Reward = torch.Tensor  # float32
Terminal = torch.Tensor  # bool

# A one-step transition in an environment
Transition = NamedTuple(
    "Transition",
    observation=Observation,
    action=Action,
    next_observation=Observation,
    reward=Reward,
    terminal=Terminal,
)
