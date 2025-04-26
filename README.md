# Multi-Agent NLP Environments

This project is a set of environments for multi-agent learning, including sentiment analysis, text generation, and dialog systems. The environments allow modeling agent interaction 
in different contexts using advanced models such as GPT, BERT.

### Description

The project includes several multi-agent learning environments:

* **Sentiment Analysis environment:** Simulates a multi-task sentiment analysis task using text data.
* **Text Generation:** Uses models to generate text based on a given context.
* **Dialog System:** Includes support for dialogs between agents, where each agent can generate responses based on given questions or queries.

Environments support both cooperative and competitive modes, and agent rewards are computed based on the quality of their actions, similarity to real-world answers, and the diversity of their responses.

### Project Structure

1. **Sentiment Analysis:** An environment for multigenic agents that analyze text tone using classification or regression.

2. **Text generation:** An environment for text generation using GPT-2 that supports cooperative and competitive interactions.

3. **Dialog System:** An environment where agents engage in dialog and attempt to generate meaningful responses based on context.

4. **Multi-agent learning:** The ability for multiple agents to interact within the same environment, with the calculation of a reward that depends on the agents' behavior.


There are also examples of environments and unit tests for each environment

### Install dependencies

1. Clone the repository:

```bash
git clone https://github.com/ImplicitLayer/multiagent_environments.git
cd multiagent_environments
```

2. Establish dependencies:

```bash
pip install -r requirements.txt
```

### Example of use

1. Sentiment Analysis Environment 

```python
from multi_agent_sentiment_analysis import MultiAgentSentimentAnalysisEnv

# Example dataset
dataset = [("I love this product", 1), ("I hate this product", -1)]

# Initialize environment with cooperative agents
env_coop = MultiAgentSentimentAnalysisEnv(dataset, num_agents=2, mode="classification", cooperative=True)
states = env_coop. reset()
print("Starting state (cooperation):", states)

# Agents make their predictions
actions = [1, 1] # Both agents predict a positive tone

# Take a step in the environment
next_states, rewards, done = env_coop.step(actions)
print("Rewards (cooperation):", rewards)
```

2. Text generation environment

```python
from multi_agent_text_generation import MultiAgentTextGenerationEnv

# Sample dataset
dataset = [("How are you?", "I'm fine!"), ("What's new?", "Nothing.")]

# Initialize environment with cooperative agents
env_coop = MultiAgentTextGenerationEnv(dataset, num_agents=2, cooperative=True)
states = env_coop. reset()
print("Starting state (cooperation):", states)

# Generate agent actions
actions = env_coop.sample_action()

# Take a step in the environment
next_states, rewards, done = env_coop.step(actions)
print("Rewards (cooperation):", rewards)
```

3. Dialog system

```python
from multi_agent_dialog_system import MultiAgentDialogEnv

# Example dataset
dataset = [("Hello, how are you?", "I'm good, thanks!"), ("What are you doing?", "Just working.")]]

# Initializing the environment
env = MultiAgentDialogEnv(dataset, num_agents=2)

# Resetting the state
states = env.reset()
print("Initial dialog state:", states)

# Generating agent responses
actions = env.sample_action()

# Processing a step
next_states, rewards, done = env.step(actions)
print("Rewards for agents:", rewards)
```

### Testing
Use the following command to run unit tests:

```bash
python -m unittest test_file_name.py
```
The project includes tests for the following functions:

* Checking the `reset()` method: Checking whether the initial initialization of the environment is correct.

* Verification of `step()` method operation in cooperative and competitive modes.

* Verification of random action generation with `sample_action()`.

* Checking the calculation of `embeddings` and `cosine similarity`.

You can also use `pytest` for testing.
