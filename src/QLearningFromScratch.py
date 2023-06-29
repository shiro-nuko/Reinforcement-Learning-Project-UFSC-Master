import numpy as np
import matplotlib.pyplot as plt

ROWS = 4
COLUMNS = 12
CLIFF = [(3, col) for col in range(1, 11)]
ACTIONS_NUM = 4
EPISODES = 500

q_values = np.zeros((ROWS * COLUMNS, ACTIONS_NUM))

def convert_coord_to_state_index(row, col):
    return row * COLUMNS + col

def update_q_value(state, next_state, action, reward):
    alpha = 1
    gamma = 1 # Undiscounted episode, as in the book
    q_values[state, action] = q_values[state, action] + alpha * (reward + gamma * np.max(q_values[next_state]) - q_values[state, action])

def get_next_state(state, action):
    current_row = state // COLUMNS
    current_column = state % COLUMNS
    next_state = None
    match action:
        case 0: # Up
            next_state = convert_coord_to_state_index(max(current_row - 1, 0), current_column)
        case 1: # Down
            next_state = convert_coord_to_state_index(min(current_row + 1, ROWS - 1), current_column)
        case 2: # Left
            next_state = convert_coord_to_state_index(current_row, max(current_column - 1, 0))
        case 3: # Right
            next_state = convert_coord_to_state_index(current_row, min(current_column + 1, COLUMNS - 1))
    return next_state

def check_for_goal_or_cliff(next_state):
    next_row = next_state // COLUMNS
    next_column = next_state % COLUMNS
    reward = -1
    episode_ended = False

    if (next_row, next_column) == (ROWS - 1, COLUMNS - 1):
        reward = 1
        episode_ended = True
    elif (next_row, next_column) in CLIFF:
        reward = -100
        episode_ended = True

    return reward, episode_ended

def walk(state, action):
    next_state = get_next_state(state, action)
    reward, episode_ended = check_for_goal_or_cliff(next_state)

    return reward, next_state, episode_ended

def choose_action(state, epsilon):
    action = None
    if np.random.rand() < epsilon:
        action = np.random.randint(ACTIONS_NUM)
    else:
        action = np.argmax(q_values[state])
    return action

def plot_reward_by_episode(reward_by_episode):
    plt.plot(reward_by_episode)
    plt.xlabel('Episode')
    plt.ylabel('Reward Per Episode')
    plt.show()

def main_loop():
    epsilon = 1
    epsilon_decay = 0.99
    reward_by_episode = []
    for episode in range(EPISODES):
        total_reward = 0
        state = convert_coord_to_state_index(3, 0)
        episode_ended = False
        step_counter = 0

        while not episode_ended:
            step_counter += 1

            action = choose_action(state, epsilon)
            current_reward, next_state, episode_ended = walk(state, action)
            update_q_value(state, next_state, action, current_reward)

            total_reward += current_reward
            state = next_state

        epsilon *= epsilon_decay
        print(f"Episode: {episode + 1}, Steps: {step_counter}, Total Reward: {total_reward}")
        reward_by_episode.append(total_reward)
    return reward_by_episode

reward_by_episode = main_loop()
plot_reward_by_episode(reward_by_episode)
