import unittest
import torch
from environments.dialogue import MultiAgentDialogueEnv, get_bert_embeddings, cosine_similarity_batch


class TestMultiAgentDialogueEnv(unittest.TestCase):

    def setUp(self):
        self.dataset = [
            ("How are you?", "I'm good, thank you!"),
            ("What's new?", "Nothing much."),
        ]
        self.env_coop = MultiAgentDialogueEnv(self.dataset, num_agents=2, cooperative=True)
        self.env_comp = MultiAgentDialogueEnv(self.dataset, num_agents=2, cooperative=False)

    def test_reset(self):
        states = self.env_coop.reset()
        self.assertEqual(len(states), 2)
        self.assertEqual(states[0], "How are you?")

    def test_step_cooperative(self):
        self.env_coop.reset()
        actions = ["I'm fine, thanks!", "Doing great!"]
        next_states, rewards, done = self.env_coop.step(actions)
        self.assertEqual(len(next_states), 2)
        self.assertEqual(len(rewards), 2)
        self.assertIsInstance(rewards[0], float)
        self.assertFalse(done)

    def test_step_competitive(self):
        self.env_comp.reset()
        actions = ["Nothing new!", "All good!"]
        next_states, rewards, done = self.env_comp.step(actions)
        self.assertEqual(len(rewards), 2)
        self.assertNotEqual(rewards[0], rewards[1])

    def test_done_flag(self):
        self.env_coop.reset()
        # Проходим всю длину датасета
        for _ in range(len(self.dataset)):
            actions = ["Response 1", "Response 2"]
            next_states, rewards, done = self.env_coop.step(actions)
        self.assertTrue(done)

    def test_sample_action(self):
        self.env_coop.reset()
        actions = self.env_coop.sample_action()
        self.assertEqual(len(actions), 2)
        for action in actions:
            self.assertIsInstance(action, str)

    def test_invalid_actions(self):
        self.env_coop.reset()
        with self.assertRaises(ValueError):
            self.env_coop.step(["Only one action"])

    def test_empty_dataset_reset(self):
        with self.assertRaises(ValueError):
            env = MultiAgentDialogueEnv([], num_agents=2)
            env.reset()


class TestUtilityFunctions(unittest.TestCase):

    def test_get_bert_embeddings(self):
        texts = ["Hello world", "Test sentence"]
        embeddings = get_bert_embeddings(texts)
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.shape[0], 2)

    def test_get_bert_embeddings_invalid_input(self):
        with self.assertRaises(ValueError):
            get_bert_embeddings("Not a list")

    def test_cosine_similarity_batch(self):
        tensor1 = torch.randn(3, 768)
        tensor2 = torch.randn(3, 768)
        similarities = cosine_similarity_batch(tensor1, tensor2)
        self.assertIsInstance(similarities, list)
        self.assertEqual(len(similarities), 3)

    def test_cosine_similarity_batch_invalid_inputs(self):
        tensor1 = torch.randn(2, 768)
        tensor2 = torch.randn(3, 768)
        with self.assertRaises(ValueError):
            cosine_similarity_batch(tensor1, tensor2)


if __name__ == '__main__':
    unittest.main()
