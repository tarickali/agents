import numpy as np
import torch

from agents.core.types import Observation, Action, Reward, Terminal, Transition
from agents.core import Memory
from agents.networks import QNetwork
from agents.utils import unwrap_transitions


class DDQNAgent:
    def __init__(
        self,
        network: QNetwork,
        memory: Memory,
        gamma: float = 0.99,
        tau: int = 0.005,
        batch_size: int = 32,
    ) -> None:
        self.behavior_network = network
        self.memory = memory

        self.target_network = network.clone()

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

    def act(self, observation: Observation) -> Action:
        return self.behavior_network.select(observation)

    def learn(self) -> None:
        if self.memory.size < self.batch_size:
            return

        transitions_batch = self.memory.sample(self.batch_size)
        observations, actions, next_observations, rewards, terminals = (
            unwrap_transitions(transitions_batch)
        )

        # Get the action values of observations with the taken actions
        Q_eval = self.behavior_network.compute(observations)[
            np.arange(self.batch_size, dtype=np.int32), actions
        ]

        # Get the action values of next observations
        with torch.no_grad():
            Q_behavior_prime = self.behavior_network.compute(next_observations)
            Q_target_prime = self.target_network.compute(next_observations)
        Q_target_prime[terminals] = 0.0

        # Compute the target value to update the Q values
        Q_targets = (
            rewards
            + self.gamma
            * Q_target_prime[
                np.arange(self.batch_size, dtype=np.int32),
                torch.argmax(Q_behavior_prime, dim=1),
            ]
        )

        # Ensure that Q_targets and Q_eval have the same dtype
        Q_targets = Q_targets.to(Q_eval.dtype)

        # Update the behavior network
        self.behavior_network.update(Q_eval, Q_targets)

        # Update the target network
        target_network_state_dict = self.target_network.model.state_dict()
        behavior_network_state_dict = self.behavior_network.model.state_dict()

        for key in behavior_network_state_dict:
            target_network_state_dict[key] = behavior_network_state_dict[
                key
            ] * self.tau + target_network_state_dict[key] * (1 - self.tau)
        self.target_network.model.load_state_dict(target_network_state_dict)

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
