import numpy as np
import torch

from gymnasium import Space

from agents.core.types import Observation, Action


class QNetwork:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.Module,  # torch.nn.modules.loss._Loss,
        observation_space: Space,
        action_space: Space,
        epsilon_init: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 5e-4,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

        self.observation_space = observation_space
        self.action_space = action_space

        self.epsilon = epsilon_init
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def compute(self, observation: Observation) -> Action:
        return self.model.forward(observation)

    def select(self, observation: Observation) -> Action:
        if np.random.random() < self.epsilon:
            return torch.tensor(self.action_space.sample())
        else:
            # Get q_values for each action
            Qs = self.model.forward(observation)
            # Get the greedy action from Qs
            # TODO: what if a batch of observations are given? what if actions are multi-dimensional?
            # FIXME: does not work if output is continuous and not multiclass
            return Qs.argmax(axis=-1)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        # FIXME: make sure that prediction and target have the same dtype
        loss = self.loss.forward(prediction, target.to(prediction.dtype))
        loss.backward()
        self.optimizer.step()

        # TODO: Should this be done when learning or when acting?
        # Reduce epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
