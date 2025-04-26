from environments.text_generation import MultiAgentTextGenerationEnv

# ==========================
# 1. Create a dataset for text generation
# ==========================

# Dataset: (specified start of text, expected continuation of text)
dataset = [
    ("Once upon a time in a small village,", "there lived a brave young girl who dreamed of exploring the world."),
    ("In the heart of the jungle,", "the mysterious temple stood hidden for centuries."),
    ("The scientist looked at the formula,", "realizing that it could change humanity forever."),
    ("Deep beneath the ocean,", "a secret civilization thrived unseen."),
    ("On the distant planet Zoron,", "strange creatures roamed under three suns."),
    ("Long ago, when magic was real,", "wizards ruled the kingdoms with ancient spells."),
]

# ==========================
# 2. Creating an environment
# ==========================

# Customization: 2 agents, cooperative mode
env_coop = MultiAgentTextGenerationEnv(
    dataset=dataset,
    num_agents=2,
    cooperative=True
)

# Customization: 3 agents, competition mode
env_comp = MultiAgentTextGenerationEnv(
    dataset=dataset,
    num_agents=3,
    cooperative=False
)

# ==========================
# 3. Game cycle: Cooperative mode
# ==========================

print("\n=== Cooperative mode ===\n")

states = env_coop.reset()

done = False
step_count = 0

while not done:
    # Print the current opening sentence
    env_coop.render()

    # Generate random actions of agents
    actions = env_coop.sample_action()

    # Printing actions
    for idx, action in enumerate(actions):
        print(f"Agent {idx} action: {action[:100]}...")  # Let's limit the output to the length of the text

    # Let's do the step
    next_states, rewards, done = env_coop.step(actions)

    # Печатаем награды
    print(f"Rewards: {rewards}\n")

    step_count += 1
    if step_count > 10:  # Protection against infinite cycle
        break

# ==========================
# 4. Game cycle: Competition mode
# ==========================

print("\n=== Competition mode ===\n")

states = env_comp.reset()

done = False
step_count = 0

while not done:
    # Print the current opening sentence
    env_comp.render()

    # Generate random actions of agents
    actions = env_comp.sample_action()

    # Print actions
    for idx, action in enumerate(actions):
        print(f"Agent {idx} action: {action[:100]}...")  # Limit output

    # Execute step
    next_states, rewards, done = env_comp.step(actions)

    # Print rewards
    print(f"Rewards: {rewards}\n")

    step_count += 1
    if step_count > 10:
        break
