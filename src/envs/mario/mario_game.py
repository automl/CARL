from abc import ABC, abstractmethod


class MarioGame(ABC):
    @abstractmethod
    def getPort(self) -> int:
        pass

    @abstractmethod
    def initGame(self):
        pass

    @abstractmethod
    def stepGame(self, left: bool, right: bool, down: bool, speed: bool, jump: bool):
        pass

    @abstractmethod
    def resetGame(self, level: str, timer: int, mario_state: int, inertia: float):
        pass

    @abstractmethod
    def computeObservationRGB(self):
        pass

    @abstractmethod
    def computeReward(self) -> float:
        pass

    @abstractmethod
    def computeDone(self) -> bool:
        pass

    @abstractmethod
    def getCompletionPercentage(self) -> float:
        pass

    @abstractmethod
    def getFrameSize(self) -> int:
        pass
