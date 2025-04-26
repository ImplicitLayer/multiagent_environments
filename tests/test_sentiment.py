import unittest
import torch
from environments.sentiment import MultiAgentSentimentAnalysisEnv


class TestMultiAgentSentimentAnalysisEnv(unittest.TestCase):

    def setUp(self):
        self.dataset = [
            ("I love this product", 1),
            ("I hate this product", -1)
        ]
        self.env_coop = MultiAgentSentimentAnalysisEnv(self.dataset, num_agents=2, mode="classification", cooperative=True)
        self.env_comp = MultiAgentSentimentAnalysisEnv(self.dataset, num_agents=2, mode="classification", cooperative=False)

    def test_reset(self):
        states = self.env_coop.reset()
        self.assertEqual(len(states), 2)
        self.assertEqual(states[0], "I love this product")

    def test_step_cooperative(self):
        self.env_coop.reset()
        actions = [1, 1]  # Both agents predict positive sentiment
        next_states, rewards, done = self.env_coop.step(actions)
        self.assertEqual(len(next_states), 2)
        self.assertEqual(len(rewards), 2)
        self.assertIsInstance(rewards[0], float)
        self.assertFalse(done)

    def test_step_competitive(self):
        self.env_comp.reset()
        actions = [1, 0]  # Agent 1 predicts positive, Agent 2 predicts neutral
        next_states, rewards, done = self.env_comp.step(actions)
        self.assertEqual(len(rewards), 2)
        self.assertNotEqual(rewards[0], rewards[1])  # In competition, rewards can differ

    def test_done_flag(self):
        self.env_coop.reset()
        # Simulate completion of the dataset
        for _ in range(len(self.dataset)):
            actions = [1, 1]
            next_states, rewards, done = self.env_coop.step(actions)
        self.assertTrue(done)

    def test_invalid_actions(self):
        self.env_coop.reset()
        with self.assertRaises(ValueError):
            self.env_coop.step([1])  # Only one action, but we need 2

    def test_invalid_dataset_reset(self):
        with self.assertRaises(ValueError):
            env = MultiAgentSentimentAnalysisEnv([], num_agents=2, mode="classification")
            env.reset()

    def test_regression_mode(self):
        # Testing regression mode where labels are floats
        dataset_regression = [("I love this product", 1.5), ("I hate this product", -1.0)]
        env_reg = MultiAgentSentimentAnalysisEnv(dataset_regression, num_agents=2, mode="regression", cooperative=True)
        env_reg.reset()
        actions = [1.5, 1.0]
        next_states, rewards, done = env_reg.step(actions)
        self.assertEqual(len(rewards), 2)
        self.assertTrue(isinstance(rewards[0], float))

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            MultiAgentSentimentAnalysisEnv(self.dataset, num_agents=2, mode="invalid_mode")

    def test_sample_action(self):
        self.env_coop.reset()
        actions = self.env_coop.sample_action()
        self.assertEqual(len(actions), 2)
        for action in actions:
            if self.env_coop.mode == "classification":
                self.assertIn(action, [-1, 0, 1])
            elif self.env_coop.mode == "regression":
                self.assertTrue(-1.0 <= action <= 1.0)


if __name__ == '__main__':
    unittest.main()
