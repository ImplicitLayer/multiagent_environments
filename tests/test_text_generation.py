import unittest
import torch
from environments.text_generation import MultiAgentTextGenerationEnv, get_bert_embeddings, cosine_similarity_batch


class TestMultiAgentTextGenerationEnv(unittest.TestCase):

    def setUp(self):
        self.dataset = [
            ("How are you?", "I'm fine!"),
            ("What's new?", "Nothing.")
        ]
        self.env_coop = MultiAgentTextGenerationEnv(self.dataset, num_agents=2, cooperative=True)
        self.env_comp = MultiAgentTextGenerationEnv(self.dataset, num_agents=2, cooperative=False)

    def test_reset(self):
        states = self.env_coop.reset()
        self.assertEqual(len(states), 2)
        self.assertEqual(states[0], "How are you?")

    def test_step_cooperative(self):
        self.env_coop.reset()
        actions = ["I'm fine!", "I'm fine!"]  # Both agents predict the correct response
        next_states, rewards, done = self.env_coop.step(actions)
        self.assertEqual(len(next_states), 2)
        self.assertEqual(len(rewards), 2)
        self.assertTrue(isinstance(rewards[0], float))
        self.assertFalse(done)

    def test_step_competitive(self):
        self.env_comp.reset()
        actions = ["I'm fine!", "I'm good!"]  # Different responses
        next_states, rewards, done = self.env_comp.step(actions)
        self.assertEqual(len(rewards), 2)
        self.assertNotEqual(rewards[0], rewards[1])  # In competition, rewards can differ

    def test_invalid_actions(self):
        self.env_coop.reset()
        with self.assertRaises(ValueError):
            self.env_coop.step(["I'm fine!"])  # Only one action, need 2

    def test_invalid_text_action(self):
        self.env_coop.reset()
        with self.assertRaises(ValueError):
            self.env_coop.step([123, "I'm fine!"])  # Invalid action type (not string)

    def test_done_flag(self):
        self.env_coop.reset()
        for _ in range(len(self.dataset)):
            actions = ["I'm fine!", "I'm fine!"]
            next_states, rewards, done = self.env_coop.step(actions)
        self.assertTrue(done)

    def test_sample_action(self):
        self.env_coop.reset()
        actions = self.env_coop.sample_action()
        self.assertEqual(len(actions), 2)
        self.assertTrue(isinstance(actions[0], str))  # Check if the generated actions are strings

    def test_get_bert_embeddings(self):
        embeddings = get_bert_embeddings(["How are you?", "What's new?"])
        self.assertEqual(embeddings.shape[0], 2)  # 2 texts, should return 2 embeddings
        self.assertEqual(embeddings.shape[1], 768)  # BERT base has 768 hidden size

    def test_cosine_similarity_batch(self):
        tensor1 = torch.rand((2, 768))
        tensor2 = torch.rand((2, 768))
        similarities = cosine_similarity_batch(tensor1, tensor2)
        self.assertEqual(len(similarities), 2)
        for sim in similarities:
            self.assertTrue(-1 <= sim <= 1)  # Cosine similarity should be in the range [-1, 1]

    def test_invalid_bert_embeddings_input(self):
        with self.assertRaises(ValueError):
            get_bert_embeddings("Invalid input")  # Should raise an error because it's not a list of strings

    def test_invalid_cosine_similarity_input(self):
        tensor1 = torch.rand((2, 768))
        tensor2 = torch.rand((3, 768))  # Mismatched shapes
        with self.assertRaises(ValueError):
            cosine_similarity_batch(tensor1, tensor2)

    def test_invalid_sample_action(self):
        # Testing the exception if `sample_action` is called without proper state
        env = MultiAgentTextGenerationEnv(self.dataset, num_agents=2, cooperative=True)
        with self.assertRaises(ValueError):
            env.sample_action()


if __name__ == '__main__':
    unittest.main()
