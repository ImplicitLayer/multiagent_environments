from environments.sentiment import MultiAgentSentimentAnalysisEnv

# Example with regression and cooperation

dataset = [
    ("I love this movie!", 1.0),  # Positive text (score: 1.0)
    ("This is the worst experience ever.", -1.0),  # Negative text (score: -1.0)
    ("It's okay, not great but not bad.", 0.0),  # Neutral text (score: 0.0)
    ("Amazing! I had so much fun.", 1.0),  # Positive text (score: 1.0)
    ("I hate this!", -1.0),  # Negative text (score: -1.0)
]

# Initialize the environment with regression and cooperation
env = MultiAgentSentimentAnalysisEnv(dataset, num_agents=3, mode="regression", cooperative=True)

# Resetting the state of the environment
states = env.reset()
print("Initial state:", states)

# Let's go to all the questions
step_counter = 0
while not env.done:
    print(f"\n=== Step {step_counter} ===")

    # Rendering the current question (text)
    env.render()

    # Agents sample their actions (random predictions for regression)
    actions = env.sample_action()
    print("Actions:", actions)

    # Taking a step into the medium and reaping the rewards
    next_states, rewards, done = env.step(actions)
    print("Rewards:", rewards)
    print("Next states:", next_states)

    step_counter += 1
