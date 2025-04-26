import random
import torch
import torch.nn.functional as F
from environments.base_env import BaseEnvironment

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiAgentSentimentAnalysisEnv(BaseEnvironment):
    """An environment for multi-agent analysis of text tone."""

    def __init__(self, dataset, num_agents=2, mode="classification", cooperative=True):
        """
        Initializes the sentiment analysis environment.

        :param dataset: (list) List of (text, tone label) pairs.
        :param num_agents: (int) Number of agents.
        :param mode: (str) "classification" or "regression".
        :param cooperative: (bool) Whether agents cooperate.
        """
        if not isinstance(dataset, list) or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in dataset):
            raise ValueError("Dataset must be a list of (text, label) tuples.")

        if not isinstance(num_agents, int) or num_agents <= 0:
            raise ValueError("num_agents must be a positive integer.")

        if mode not in ["classification", "regression"]:
            raise ValueError("mode must be either 'classification' or 'regression'.")

        if not isinstance(cooperative, bool):
            raise ValueError("cooperative must be a boolean.")

        self.dataset = dataset
        self.num_agents = num_agents
        self.mode = mode
        self.cooperative = cooperative
        self.current_index = 0
        self.done = False
        self.states = [None] * num_agents

    def reset(self):
        """Resets the environment to the starting state."""
        if not self.dataset:
            raise ValueError("Cannot reset environment: dataset is empty.")

        self.current_index = 0
        self.done = False
        text, _ = self.dataset[self.current_index]
        self.states = [text] * self.num_agents
        return self.states

    def step(self, actions):
        """
        Processes agent predictions and computes rewards.

        :param actions: (list) List of agent predictions.
        :return: (tuple) (next states, rewards, done).
        """
        if self.done:
            return [None] * self.num_agents, [0] * self.num_agents, True

        if not isinstance(actions, list) or len(actions) != self.num_agents:
            raise ValueError(f"Actions must be a list of {self.num_agents} elements.")

        if self.mode == "classification" and not all(isinstance(a, int) for a in actions):
            raise ValueError("In classification mode, actions must be integers.")
        if self.mode == "regression" and not all(isinstance(a, (float, int)) for a in actions):
            raise ValueError("In regression mode, actions must be floats or ints.")

        try:
            _, true_label = self.dataset[self.current_index]
        except IndexError:
            raise IndexError(f"Invalid current index {self.current_index} for dataset size {len(self.dataset)}.")

        rewards = []

        if self.mode == "regression":
            try:
                true_tensor = torch.tensor([true_label], device=device, dtype=torch.float32)
            except Exception as e:
                raise RuntimeError(f"Error creating true_label tensor: {e}")

        for action in actions:
            try:
                if self.mode == "classification":
                    reward = 1.0 if action == true_label else -1.0
                elif self.mode == "regression":
                    action_tensor = torch.tensor([action], device=device, dtype=torch.float32)
                    reward = -F.mse_loss(action_tensor, true_tensor).item()
            except Exception as e:
                print(f"Warning: error during reward computation for action {action}: {e}")
                reward = -1.0  # Penalize unexpected error

            rewards.append(reward)

        if self.cooperative:
            avg_reward = sum(rewards) / len(rewards)
            rewards = [avg_reward] * self.num_agents

        self.current_index += 1
        if self.current_index >= len(self.dataset):
            self.done = True
            next_states = [None] * self.num_agents
        else:
            next_text, _ = self.dataset[self.current_index]
            next_states = [next_text] * self.num_agents

        return next_states, rewards, self.done

    def render(self):
        """Display the current text state."""
        if not self.states or self.states[0] is None:
            print("No current text to render.")
        else:
            print(f"Current text: {self.states[0]}")

    def sample_action(self):
        """Randomly samples an action based on the mode."""
        if self.mode == "classification":
            return [random.choice([-1, 0, 1]) for _ in range(self.num_agents)]
        elif self.mode == "regression":
            return [random.uniform(-1.0, 1.0) for _ in range(self.num_agents)]


# Example usage
if __name__ == "__main__":
    try:
        dataset = [("I love this product", 1), ("I hate this product", -1)]

        env_coop = MultiAgentSentimentAnalysisEnv(dataset, num_agents=2, mode="classification", cooperative=True)
        states = env_coop.reset()
        print("Beginning state (cooperation):", states)

        actions = [1, 1]  # Both agents predict positive
        next_states, rewards, done = env_coop.step(actions)
        print("Rewards (cooperation):", rewards)

        env_comp = MultiAgentSentimentAnalysisEnv(dataset, num_agents=2, mode="classification", cooperative=False)
        states = env_comp.reset()
        print("Beginning state (competition):", states)

        actions = [1, 0]  # Agent 1 predicts positive, agent 2 neutral
        next_states, rewards, done = env_comp.step(actions)
        print("Rewards (competition):", rewards)

    except Exception as e:
        print(f"Fatal error: {e}")
