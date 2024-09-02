import numpy as np
import torch

from agents.core.types import Observation, Action, Reward, Terminal, Transition
from agents.core import Memory, Network
from agents.utils import unwrap_transitions


class QAgent:
    def __init__(
        self,
        network: Network,
        memory: Memory,
        gamma: float = 0.99,
        batch_size: int = 32,
    ) -> None:
        self.network = network
        self.memory = memory

        self.gamma = gamma
        self.batch_size = batch_size

    def act(self, observation: Observation) -> Action:
        return self.network.select(observation)

    def learn(self) -> None:
        if self.memory.size < self.batch_size:
            return

        transitions_batch = self.memory.sample(self.batch_size)
        observations, actions, next_observations, rewards, terminals = (
            unwrap_transitions(transitions_batch)
        )

        # Get the action values of observations with the taken actions
        # FIXME: does not work if output is continuous and not multiclass
        Q_eval = self.network.compute(observations)[
            np.arange(self.batch_size, dtype=np.int32), actions
        ]

        # Get the action values of next observations
        Q_prime = self.network.compute(next_observations)
        Q_prime[terminals] = 0.0

        # Compute the target value to update the Q values
        Q_targets = rewards + self.gamma * torch.max(Q_prime, dim=1)[0]

        # Update the network
        self.network.update(Q_eval, Q_targets)

    def remember(
        self,
        observation: Observation,
        action: Action,
        next_observation: Observation,
        reward: Reward,
        terminal: Terminal,
    ):
        self.memory.store(
            Transition(observation, action, next_observation, reward, terminal)
        )
