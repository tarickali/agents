import torch

from agents.core.types import Observation, Action, Reward, Terminal, Transition


def unwrap_transitions(
    transitions: list[Transition],
) -> tuple[Observation, Action, Observation, Reward, Terminal]:
    observations, actions, next_observations, rewards, terminals = list(
        zip(*transitions)
    )

    observations = torch.stack(observations)
    actions = torch.stack(actions)
    next_observations = torch.stack(next_observations)
    rewards = torch.stack(rewards)
    terminals = torch.stack(terminals)

    return observations, actions, next_observations, rewards, terminals
