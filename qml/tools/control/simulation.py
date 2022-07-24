from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar

S = TypeVar("S")
C = TypeVar("C")


class StateUpdater(ABC, Generic[S, C]):
    @abstractmethod
    def update_state(self, old_state: S, control: C) -> S:
        pass


class Controller(ABC, Generic[S, C]):
    @abstractmethod
    def control(self, state: S) -> C:
        pass


def simulate(
    initial_state: S,
    state_updater: StateUpdater[S, C],
    controller: Controller[S, C],
    num_steps: int,
) -> tuple[Sequence[S], Sequence[C]]:
    states = [initial_state]
    controls = []
    for i in range(num_steps):
        controls.append(controller.control(states[-1]))
        states.append(state_updater.update_state(states[-1], controls[-1]))
    return states, controls
