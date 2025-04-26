import random
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu
from transformers import BertTokenizer, BertModel
from environments.base_env import BaseEnvironment

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize BERT tokenizer and model
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    bert_model.eval()
except Exception as e:
    raise RuntimeError(f"Error loading BERT models: {e}")

# Download necessary NLTK resources
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: Could not download nltk data. Error: {e}")


def get_bert_embeddings(texts):
    """
    Computes BERT embeddings for input texts.

    :param texts: (list) List of input strings.
    :return: (torch.Tensor) Tensor of shape (len(texts), hidden_size) with CLS embeddings.
    """
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError("Input must be a list of strings.")

    with torch.no_grad():
        try:
            inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(device)
            outputs = bert_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            return cls_embeddings
        except Exception as e:
            raise RuntimeError(f"Error during BERT embedding computation: {e}")


def cosine_similarity_batch(tensor1, tensor2):
    """
    Computes cosine similarity between two tensors.

    :param tensor1: (torch.Tensor)
    :param tensor2: (torch.Tensor)
    :return: (list) Similarity scores.
    """
    if not (isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor)):
        raise ValueError("Inputs must be torch tensors.")
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"Tensors must have the same shape, got {tensor1.shape} and {tensor2.shape}.")

    similarities = torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=-1)
    return similarities.tolist()


class MultiAgentDialogueEnv(BaseEnvironment):
    """
    Multi-agent dialogue environment supporting cooperation and competition.
    """

    def __init__(self, dataset, num_agents=2, cooperative=True):
        """
        Initializes the environment.

        :param dataset: (list) List of (question, expected answer) pairs.
        :param num_agents: (int) Number of agents.
        :param cooperative: (bool) Cooperation flag.
        """
        if not isinstance(dataset, list) or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in dataset):
            raise ValueError("Dataset must be a list of (question, answer) tuples.")

        if not isinstance(num_agents, int) or num_agents <= 0:
            raise ValueError("num_agents must be a positive integer.")

        if not isinstance(cooperative, bool):
            raise ValueError("cooperative must be a boolean.")

        self.dataset = dataset
        self.num_agents = num_agents
        self.cooperative = cooperative
        self.current_index = 0
        self.done = False
        self.states = [None] * num_agents
        self.previous_responses = {i: [] for i in range(num_agents)}

    def reset(self):
        """
        Resets environment to the initial state.

        :return: (list) Initial states.
        """
        if not self.dataset:
            raise ValueError("Cannot reset environment: dataset is empty.")

        self.current_index = 0
        self.done = False
        self.previous_responses = {i: [] for i in range(self.num_agents)}
        self.states = [self.dataset[self.current_index][0]] * self.num_agents
        return self.states

    def step(self, actions):
        """
        Processes actions and computes rewards.

        :param actions: (list) List of responses from agents.
        :return: (tuple) (next_states, rewards, done_flag).
        """
        if self.done:
            return [None] * self.num_agents, [0] * self.num_agents, True

        if not isinstance(actions, list) or len(actions) != self.num_agents or not all(isinstance(a, str) for a in actions):
            raise ValueError(f"Actions must be a list of {self.num_agents} strings.")

        rewards = []
        try:
            expected_response = self.dataset[self.current_index][1]
        except IndexError:
            raise IndexError(f"Invalid current index {self.current_index} in dataset.")

        # BLEU scores
        try:
            reference = nltk.word_tokenize(expected_response.lower())
        except LookupError:
            nltk.download('punkt')
            reference = nltk.word_tokenize(expected_response.lower())

        bleu_scores = []
        for action in actions:
            try:
                candidate = nltk.word_tokenize(action.lower())
                bleu_score = sentence_bleu([reference], candidate)
            except Exception:
                bleu_score = 0.0
            bleu_scores.append(bleu_score)

        # Semantic similarity
        texts = [expected_response] + actions
        try:
            embeddings = get_bert_embeddings(texts)
            emb_expected = embeddings[0].unsqueeze(0).expand(self.num_agents, -1)
            emb_actions = embeddings[1:]
            semantic_scores = cosine_similarity_batch(emb_expected, emb_actions)
        except Exception as e:
            print(f"Warning: semantic similarity failed with error: {e}")
            semantic_scores = [0.0] * self.num_agents

        for i in range(self.num_agents):
            diversity_penalty = -0.5 if actions[i] in self.previous_responses[i] else 0.5
            self.previous_responses[i].append(actions[i])
            reward = (bleu_scores[i] * 2 + semantic_scores[i] * 3 + diversity_penalty) / 3
            rewards.append(reward)

        # Cooperation handling
        if self.cooperative:
            avg_reward = sum(rewards) / len(rewards)
            rewards = [avg_reward] * self.num_agents

        # Advance dataset index
        self.current_index += 1
        if self.current_index >= len(self.dataset):
            self.done = True
            next_states = [None] * self.num_agents
        else:
            next_states = [self.dataset[self.current_index][0]] * self.num_agents

        return next_states, rewards, self.done

    def render(self):
        """
        Renders current state.

        :return: None
        """
        if not self.states or self.states[0] is None:
            print("No active question.")
        else:
            print(f"Current question: {self.states[0]}")

    def sample_action(self):
        """
        Samples random actions.

        :return: (list) Sampled actions.
        """
        if not self.dataset:
            raise ValueError("Cannot sample actions: dataset is empty.")
        return [random.choice([resp for _, resp in self.dataset]) for _ in range(self.num_agents)]


# Example usage
if __name__ == "__main__":
    try:
        dataset = [
            ("How are you?", "I'm good, thank you!"),
            ("What's new?", "Nothing much."),
            ("What's your favorite movie?", "I love Inception."),
            ("Tell me a joke!", "Why don't scientists trust atoms? Because they make up everything!")
        ]

        env_coop = MultiAgentDialogueEnv(dataset, num_agents=2, cooperative=True)

        states = env_coop.reset()
        print("Initial states:", states)

        actions = ["I'm fine too!", "Doing well, thanks!"]
        next_states, rewards, done = env_coop.step(actions)
        print("Rewards (cooperation):", rewards)

        env_comp = MultiAgentDialogueEnv(dataset, num_agents=2, cooperative=False)

        states = env_comp.reset()
        print("Initial states:", states)

        actions = ["I'm good!", "Great!"]
        next_states, rewards, done = env_comp.step(actions)
        print("Rewards (competition):", rewards)

    except Exception as e:
        print(f"Fatal error: {e}")
