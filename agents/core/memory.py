from typing import Protocol

from agents.core.types import Transition


class Memory(Protocol):
    def store(self, transition: Transition) -> None: ...
    def sample(self, k: int = 1) -> list[Transition]: ...

    @property
    def size(self) -> int: ...
