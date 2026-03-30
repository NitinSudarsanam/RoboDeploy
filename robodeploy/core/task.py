from abc import ABC, abstractmethod


class BaseTask(ABC):
    def __init__(self, robot_id: int):
        self.robot_id = robot_id

    @abstractmethod
    def get_observation_spec(self) -> dict:
        """Define what sensors this task needs."""
        # e.g., {"rgb": True, "depth": False, "segmentation": True}
        pass

    @abstractmethod
    def get_instruction(self) -> str:
        """The language goal for this specific task."""
        pass