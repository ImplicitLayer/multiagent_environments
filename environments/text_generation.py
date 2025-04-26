import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertModel
from environments.base_env import BaseEnvironment
from nltk.translate.bleu_score import sentence_bleu
import nltk

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load GPT-2 for text generation
try:
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
    model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model_gpt2.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load GPT-2 model or tokenizer: {e}")

# Load BERT for semantic similarity
try:
    tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
    model_bert = BertModel.from_pretrained("bert-base-uncased").to(device)
    model_bert.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load BERT model or tokenizer: {e}")


def get_bert_embeddings(texts):
    """
    Computes BERT embeddings for a list of texts.

    :param texts: (list) List of input texts.
    :return: (torch.Tensor) Tensor of CLS embeddings for each text.
    """
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError("Input to get_bert_embeddings must be a list of strings.")

    with torch.no_grad():
        try:
            inputs = tokenizer_bert(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model_bert(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
        except Exception as e:
            raise RuntimeError(f"Failed to compute BERT embeddings: {e}")
    return cls_embeddings


def cosine_similarity_batch(tensor1, tensor2):
    """
    Computes batch-wise cosine similarity between two tensors.

    :param tensor1: (torch.Tensor) Tensor of shape (batch_size, hidden_size).
    :param tensor2: (torch.Tensor) Tensor of shape (batch_size, hidden_size).
    :return: (list) List of cosine similarity scores between corresponding elements.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"Tensors must have the same shape for cosine similarity, got {tensor1.shape} and {tensor2.shape}.")
    return F.cosine_similarity(tensor1, tensor2, dim=-1).tolist()


class MultiAgentTextGenerationEnv(BaseEnvironment):
    """
    Multi-agent environment for text generation with support for cooperation and competition.

    :param dataset: (list) List of (seed_text, expected_response) pairs.
    :param num_agents: (int) Number of agents.
    :param cooperative: (bool) Whether agents cooperate (True) or compete (False).
    """

    def __init__(self, dataset, num_agents=2, cooperative=True):
        """
        Initializes the environment.

        :param dataset: (list) List of (seed_text, expected_response) pairs.
        :param num_agents: (int) Number of agents.
        :param cooperative: (bool) Flag for cooperative (True) or competitive (False) behavior.
        """
        if not isinstance(dataset, list) or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in dataset):
            raise ValueError("Dataset must be a list of (seed_text, expected_response) tuples.")

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
        Resets the environment and starts a new text generation episode.

        :return: (list) Initial states for all agents.
        """
        if not self.dataset:
            raise ValueError("Cannot reset environment: dataset is empty.")

        self.current_index = 0
        self.done = False
        self.previous_responses = {i: [] for i in range(self.num_agents)}
        seed_text, _ = self.dataset[self.current_index]
        self.states = [seed_text] * self.num_agents
        return self.states

    def step(self, actions):
        """
        Processes actions from all agents and computes rewards.

        :param actions: (list) List of generated texts from agents.
        :return: (tuple) (next_states, rewards, done_flag).
        """
        if self.done:
            return [None] * self.num_agents, [0.0] * self.num_agents, True

        if not isinstance(actions, list) or len(actions) != self.num_agents:
            raise ValueError(f"Actions must be a list of {self.num_agents} strings.")

        expected_text = self.dataset[self.current_index][1]
        emb_expected = get_bert_embeddings([expected_text])

        rewards = []
        for i, action in enumerate(actions):
            if not isinstance(action, str):
                raise ValueError("Each action must be a string.")

            try:
                reference = expected_text.lower().split()
                candidate = action.lower().split()
                bleu_score = sentence_bleu([reference], candidate)

                emb_action = get_bert_embeddings([action])
                semantic_score = cosine_similarity_batch(emb_expected, emb_action)[0]

                diversity_penalty = -0.5 if action in self.previous_responses[i] else 0.5
                self.previous_responses[i].append(action)

                reward = (bleu_score * 2 + semantic_score * 3 + diversity_penalty) / 3
            except Exception as e:
                print(f"Warning: Error computing reward for agent {i}: {e}")
                reward = -1.0

            rewards.append(reward)

        if self.cooperative:
            avg_reward = sum(rewards) / self.num_agents
            rewards = [avg_reward] * self.num_agents

        self.current_index += 1
        if self.current_index >= len(self.dataset):
            self.done = True
            next_states = [None] * self.num_agents
        else:
            next_seed, _ = self.dataset[self.current_index]
            next_states = [next_seed] * self.num_agents

        return next_states, rewards, self.done

    def render(self):
        """Displays the current text prompt."""
        if not self.states or self.states[0] is None:
            print("No current text available.")
        else:
            print(f"Current text: {self.states[0]}")

    def sample_action(self):
        """
        Generates random texts for each agent using GPT-2 (for testing).

        :return: (list) List of generated texts.
        """
        if not self.states or self.states[0] is None:
            raise ValueError("Cannot sample action: current state is empty.")

        try:
            prompt = self.states[0]
            inputs = tokenizer_gpt2(prompt, return_tensors="pt").to(device)

            outputs = model_gpt2.generate(
                inputs.input_ids,
                max_length=50,
                num_return_sequences=self.num_agents,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
            generated_texts = tokenizer_gpt2.batch_decode(outputs, skip_special_tokens=True)
            return generated_texts
        except Exception as e:
            raise RuntimeError(f"Error during GPT-2 text generation: {e}")


# Example usage
if __name__ == "__main__":
    try:
        dataset = [("How are you?", "I'm fine!"), ("What's new?", "Nothing.")]

        env_coop = MultiAgentTextGenerationEnv(dataset, num_agents=2, cooperative=True)
        states = env_coop.reset()
        print("Beginning state (cooperation):", states)

        actions = env_coop.sample_action()
        next_states, rewards, done = env_coop.step(actions)
        print("Rewards (cooperation):", rewards)

        env_comp = MultiAgentTextGenerationEnv(dataset, num_agents=2, cooperative=False)
        states = env_comp.reset()
        print("Beginning state (competition):", states)

        actions = env_comp.sample_action()
        next_states, rewards, done = env_comp.step(actions)
        print("Rewards (competition):", rewards)

    except Exception as e:
        print(f"Fatal error during environment run: {e}")
