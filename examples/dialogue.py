from environments.dialogue import MultiAgentDialogueEnv

dataset = [
    ("How are you?", "I'm doing well, thank you."),
    ("What is your favorite color?", "My favorite color is blue."),
    ("Where do you live?", "I live in the cloud."),
    ("What is your profession?", "I'm an AI language model."),
    ("What do you do in your free time?", "I enjoy learning new things."),
    ("Can you help me with homework?", "Of course! I'll do my best."),
    ("Do you like music?", "Yes, I love listening to classical music."),
    ("What is your favorite food?", "I don't eat, but I hear pizza is popular."),
    ("How old are you?", "I was launched in 2020."),
    ("What languages do you speak?", "I can understand and generate text in multiple languages."),
]

# Creating an environment with 3 agents and competitive mechanics
env = MultiAgentDialogueEnv(dataset, num_agents=3, cooperative=False)

# We'll reset env before the episode begins
states = env.reset()
print("Initial state:", states)

# Going through all the questions
step_counter = 0
while not env.done:
    print(f"\n=== Step {step_counter} ===")

    # Rendering the current question
    env.render()

    # Agents sample their actions (random responses from the dataset)
    actions = env.sample_action()
    print("Actions:", actions)

    # Taking a step into the medium
    next_states, rewards, done = env.step(actions)

    print("Rewards:", rewards)
    print("Next states:", next_states)
    print("Done:", done)

    step_counter += 1


