from collections import deque
import random

from agents.core.types import Observation, Action, Reward, Terminal, Transition


class UniformBuffer:
    def __init__(self, capacity: int = int(10e6)) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def store(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, k: int = 1) -> list[Transition]:
        return random.sample(self.buffer, k=k)

    @property
    def size(self) -> int:
        return len(self.buffer)
