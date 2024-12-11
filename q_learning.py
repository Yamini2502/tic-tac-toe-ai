import time
import sys
import numpy as np
import random
import matplotlib.pyplot as plt

# Initializing Q-values
Q = {}

# Hyperparameters
alpha = 0.02  # Learning rate
gamma = 0.85  # Discount factor
epsilon = 0.8  # Exploration rate
epsilon_min = 0.05  # Minimum epsilon
epsilon_decay = 0.9999  # Slower decay for exploration
temperature = 1.0  # Temperature for Boltzmann exploration
temperature_min = 0.1  # Minimum temperature
temperature_decay = 0.999  # Slower decay of temperature

# Game configuration for dynamic board sizes
board_size = 3  # Can be adjusted (e.g., 5 for a 5x5 board)
actionSpace = [(i, j) for i in range(board_size) for j in range(board_size)]

# Performance metrics
results = {"win": 0, "loss": 0, "draw": 0}
decision_times = []
memory_logs = []
regret = 0
optimal_reward = 1  # Assume a win provides the optimal reward
total_rewards = []  # To store total rewards over episodes

# Reward function
def reward(state):
    # Check rows, columns, and diagonals
    for row in state:
        if sum(row) == board_size: 
            return 1  # Positive reward for win
        if sum(row) == -board_size: 
            return 0  # No reward for loss
    
    for col in zip(*state):
        if sum(col) == board_size: 
            return 1  # Positive reward for win
        if sum(col) == -board_size: 
            return 0  # No reward for loss
    
    if sum(state[i][i] for i in range(board_size)) == board_size: 
        return 1  # Positive reward for win
    if sum(state[i][board_size - 1 - i] for i in range(board_size)) == board_size: 
        return 1  # Positive reward for win
    if sum(state[i][i] for i in range(board_size)) == -board_size: 
        return 0  # No reward for loss
    if sum(state[i][board_size - 1 - i] for i in range(board_size)) == -board_size: 
        return 0  # No reward for loss

    if all(all(cell != 0 for cell in row) for row in state):
        return -0.1 

    return 0  # Ongoing game

# Boltzmann exploration for action selection
def boltzmannPolicy(state, temperature):
    q_values = np.array([Q.get((tuple(map(tuple, state)), action), 0) for action in actionSpace])
    exp_q_values = np.exp(q_values / temperature)
    probabilities = exp_q_values / np.sum(exp_q_values)
    
    action = np.random.choice(len(actionSpace), p=probabilities)
    return actionSpace[action]

# Q-learning update
def qLearning(state, action, nextState, reward, alpha, gamma):
    state = tuple(map(tuple, state))
    nextState = tuple(map(tuple, nextState))
    
    current_q = Q.get((state, action), 0)
    future_q = max([Q.get((nextState, a), 0) for a in actionSpace])
    
    Q[(state, action)] = current_q + alpha * (reward + gamma * future_q - current_q)

# Alpha-Beta Pruning for opponent's turn
def alpha_beta_pruning(state, depth, alpha, beta, maximizing_player):
    winner = reward(state)
    if winner != 0 or depth == 0:
        return winner

    if maximizing_player:
        max_eval = float('-inf')
        for action in actionSpace:
            if state[action[0]][action[1]] == 0:
                state[action[0]][action[1]] = -1  # Opponent's move
                eval = alpha_beta_pruning(state, depth - 1, alpha, beta, False)
                state[action[0]][action[1]] = 0  # Undo move
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        return max_eval
    else:
        min_eval = float('inf')
        for action in actionSpace:
            if state[action[0]][action[1]] == 0:
                state[action[0]][action[1]] = 1  # Agent's move
                eval = alpha_beta_pruning(state, depth - 1, alpha, beta, True)
                state[action[0]][action[1]] = 0  # Undo move
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
        return min_eval

# Opponent's Move (Alpha-Beta)
def opponent_move(state):
    best_value = float('-inf')
    best_move = None

    # Ensure the opponent picks a valid move from the available actions
    valid_actions = [action for action in actionSpace if state[action[0]][action[1]] == 0]
    
    if not valid_actions:
        print("No valid moves left!")
        return None  # In case no valid moves are available (shouldn't happen in a normal game)
    
    for action in valid_actions:
        state[action[0]][action[1]] = -1  # Opponent's move
        move_value = alpha_beta_pruning(state, 3, float('-inf'), float('inf'), False)
        state[action[0]][action[1]] = 0  # Undo move
        print(f"Evaluating move {action} with value {move_value}")  # Debugging output
        if move_value > best_value:
            best_value = move_value
            best_move = action
    
    # Ensure we always return a valid move
    if best_move is None:
        print("Error: No best move found!")  # Debugging output
        return random.choice(valid_actions)  # Fallback to a random valid move
    
    return best_move

# Run one game episode
def run_episode(epsilon):
    state = [[0 for _ in range(board_size)] for _ in range(board_size)]
    episode_reward = 0

    while True:
        # Player's random move (simulated)
        player_action = random.choice([a for a in actionSpace if state[a[0]][a[1]] == 0])
        state[player_action[0]][player_action[1]] = 1
        reward_value = reward(state)
        if reward_value != 0 or all(all(cell != 0 for cell in row) for row in state):
            return reward_value, episode_reward

        # Agent's turn (Q-learning agent)
        start_time = time.time()
        action = boltzmannPolicy(state, temperature)  # Boltzmann exploration
        decision_time = time.time() - start_time
        decision_times.append(decision_time)

        state[action[0]][action[1]] = -1
        next_reward = reward(state)
        episode_reward += next_reward

        # Update Q-values
        qLearning(state, action, state, next_reward, alpha, gamma)

        # Check for win/loss or tie
        if next_reward != 0 or all(all(cell != 0 for cell in row) for row in state):
            return next_reward, episode_reward

        # Opponent's turn (Alpha-Beta Pruning)
        opponent_action = opponent_move(state)
        if opponent_action is None:
            return 0, episode_reward  # If no valid move, game ends unexpectedly

        state[opponent_action[0]][opponent_action[1]] = 1
        next_reward = reward(state)
        episode_reward += next_reward

        if next_reward != 0 or all(all(cell != 0 for cell in row) for row in state):
            return next_reward, episode_reward

# Run multiple episodes
num_episodes = 500

for episode in range(num_episodes):
    result, episode_reward = run_episode(epsilon)
    total_rewards.append(episode_reward)
    memory_logs.append(sys.getsizeof(Q))
    regret += (optimal_reward - episode_reward)

    if result == 1:
        results["win"] += 1
    elif result == -1:
        results["loss"] += 1
    else:
        results["draw"] += 1

    # Gradually reduce epsilon to favor exploitation
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Gradually reduce temperature for Boltzmann exploration
    temperature = max(temperature_min, temperature * temperature_decay)

# Plot metrics
episodes = list(range(num_episodes))

# Learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(episodes, np.cumsum(total_rewards) / (np.arange(num_episodes) + 1))
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("Learning Curve")

# Game outcomes
plt.subplot(1, 2, 2)
plt.bar(results.keys(), results.values(), color=['green', 'red', 'blue'])
plt.title("Game Outcomes")
plt.ylabel("Count")
plt.xlabel("Outcome")

plt.tight_layout()
plt.show()

# Final metrics
print("\n--- Q-Learning Metrics ---")
print(f"Win Rate: {results['win'] / num_episodes:.3f}")
print(f"Loss Rate: {results['loss'] / num_episodes:.3f}")
print(f"Draw Rate: {results['draw'] / num_episodes:.3f}")
print(f"Average Execution Time per Move: {np.mean(decision_times):.6f} seconds")
print(f"Average Memory Usage: {np.mean(memory_logs) / 1024:.3f} KB")
print(f"Total Regret: {regret:.3f}")
print(f"Total Reward: {np.sum(total_rewards):.3f}")
