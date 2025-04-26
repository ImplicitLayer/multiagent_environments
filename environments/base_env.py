from abc import ABC, abstractmethod


class BaseEnvironment(ABC):
    """An abstract class for the agent interaction environment."""

    @abstractmethod
    def reset(self):
        """
        Resets the environment to the initial state.
        :return: (any) Initial state.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Processes the agent action and returns the following state, reward, and completion flag.
        :param action: (any) Agent action.
        :return: (tuple) (next_state, reward, done).
        """
        pass

    @abstractmethod
    def render(self):
        """
        Displays the current state of the environment (if applicable).
        """
        pass

    @abstractmethod
    def sample_action(self):
        """
        Returns a random valid action.
        :return: (any) Action.
        """
        pass
