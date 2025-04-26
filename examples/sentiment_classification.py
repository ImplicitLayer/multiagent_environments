from environments.sentiment import MultiAgentSentimentAnalysisEnv

# Example with classification and competition

dataset = [
    ("I love this movie!", 1),  # Positive text (score: 1)
    ("This is the worst experience ever.", -1),  # Negative text (score: -1)
    ("It's okay, not great but not bad.", 0),  # Neutral text (score: 0)
    ("Amazing! I had so much fun.", 1),  # Positive text (score: 1)
    ("I hate this!", -1),  # Negative text (score: -1)
]

# Initialize the environment with classification and competition
env = MultiAgentSentimentAnalysisEnv(dataset, num_agents=3, mode="classification", cooperative=False)

# Resetting the state of the environment
states = env.reset()
print("Initial state:", states)

# Let's go to all the questions
step_counter = 0
while not env.done:
    print(f"\n=== Step {step_counter} ===")

    # Rendering the current question (text)
    env.render()

    # Agents sample their actions (random predictions for classification)
    actions = env.sample_action()
    print("Actions:", actions)

    # Taking a step into the medium and reaping the rewards
    next_states, rewards, done = env.step(actions)
    print("Rewards:", rewards)
    print("Next states:", next_states)

    step_counter += 1
