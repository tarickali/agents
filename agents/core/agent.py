from typing import Protocol, Any

from .types import Observation, Action


class Agent(Protocol):
    def act(self, observation: Observation) -> Action: ...
    def learn(self) -> None: ...
