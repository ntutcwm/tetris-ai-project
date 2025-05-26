import tensorflow as tf
if not tf.config.list_physical_devices('GPU'):
    raise RuntimeError("CUDA GPU is not available. Please ensure CUDA is properly installed and configured.")

from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean
from logs import CustomTensorBoard
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import csv
import os # Needed for os.path.exists and os.rename
        

# Run dqn with Tetris
def dqn():
    env = Tetris()
    episodes = 600 # total number of episodes
    max_steps = None # max number of steps per game (None for infinite)
    epsilon_stop_episode = 450 # at what episode the random exploration stops (600 * 0.75)
    mem_size = 20000 # maximum number of steps stored by the agent (increased from 1000)
    discount = 0.95 # discount in the Q-learning formula (see DQNAgent)
    batch_size = 512 # number of actions to consider in each training (increased from 128)
    epochs = 1 # number of epochs per training
    render_every = 100 # renders the gameplay every x episodes (increased from 50)
    render_delay = None # delay added to render each frame (None for no delay)
    log_every = 10 # logs the current stats every x episodes
    replay_start_size = 2000 # minimum steps stored in the agent required to start training (increased from 1000, should be <= mem_size)
    train_every = 1 # train every x episodes
    n_neurons = [64, 64, 64] # number of neurons for each activation layer (increased from [32,32,32])
    activations = ['relu', 'relu', 'relu', 'linear'] # activation layers
    save_best_model = True # saves the best model so far at "best.keras"
    save_model_every_n_episodes = 100 # Save a checkpoint every N episodes

    # Early stopping parameters
    early_stopping_patience = 50  # Stop if avg score doesn't improve for this many log_every periods (50 * 10 = 500 episodes)
    early_stopping_min_delta = 0.01 # Minimum change in monitored quantity to qualify as an improvement
    best_avg_score_for_early_stopping = -float('inf')
    patience_counter = 0

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    # CSV Logger setup
    csv_log_path = f"{log_dir}/training_metrics.csv"
    csv_header = ["episode", "avg_score", "min_score", "max_score", "epsilon", "total_lines_cleared", "played_steps_episode", "cumulative_episode_score", "episode_reward_debug"]
    with open(csv_log_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(csv_header)

    all_episode_scores = [] # For plotting
    best_score_overall = -float('inf') # For saving best.keras

    for episode in tqdm(range(1, episodes + 1)):
        current_state = env.reset()
        done = False
        steps = 0
        episode_reward = 0

        render_this_episode = render_every and episode % render_every == 0

        # Game
        while not done and (not max_steps or steps < max_steps):
            next_states_dict = env.get_next_states()
            if not next_states_dict: # No possible moves, game might be over or in a stuck state
                done = True
                break
            
            # The original code had a slight bug if next_states_dict was empty.
            # Ensure keys are tuples for agent.best_state if it expects immutable types
            processed_next_states = {tuple(v): k for k, v in next_states_dict.items()}
            
            best_s = agent.best_state(processed_next_states.keys())
            if best_s is None: # Should not happen if next_states_dict is not empty
                done = True # Or handle error appropriately
                break
                
            best_a = processed_next_states[best_s]

            reward, done = env.play(best_a[0], best_a[1], render=render_this_episode,
                                    render_delay=render_delay)
            
            episode_reward += reward
            agent.add_to_memory(current_state, best_s, reward, done) # Use best_s as next_state
            current_state = best_s
            steps += 1
        
        current_episode_final_score = env.get_game_score() # This is the cumulative score from Tetris env
        all_episode_scores.append(current_episode_final_score)

        # Train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)

        # Logs
        if log_every and episode % log_every == 0:
            avg_score = mean(all_episode_scores[-log_every:]) if len(all_episode_scores) >= log_every else mean(all_episode_scores)
            min_score = min(all_episode_scores[-log_every:]) if len(all_episode_scores) >= log_every else min(all_episode_scores)
            max_score = max(all_episode_scores[-log_every:]) if len(all_episode_scores) >= log_every else max(all_episode_scores)

            log.log(episode, avg_score=avg_score, min_score=min_score, max_score=max_score,
                            epsilon=agent.epsilon, total_lines=env.total_lines_cleared, played_steps_ep=env.played_steps,
                            cumulative_score_ep=current_episode_final_score, episode_reward_val=episode_reward) # Added more detailed logging for tensorboard
                    
            # Log to CSV
            with open(csv_log_path, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([episode, avg_score, min_score, max_score, agent.epsilon, env.total_lines_cleared, env.played_steps, current_episode_final_score, episode_reward])
            
            # Early stopping check
            if avg_score > best_avg_score_for_early_stopping + early_stopping_min_delta:
                best_avg_score_for_early_stopping = avg_score
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at episode {episode} due to no improvement in avg_score for {early_stopping_patience * log_every} episodes.")
                break

        # Save best model
        if save_best_model and current_episode_final_score > best_score_overall:
            print(f'Saving a new best model (score={current_episode_final_score}, episode={episode})')
            best_score_overall = current_episode_final_score
            
            old_best_model_path = "best.keras"
            if os.path.exists(old_best_model_path):
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                backup_model_path = f"best_backup_{timestamp}_ep{episode}_score{int(current_episode_final_score)}.keras"
                try:
                    os.rename(old_best_model_path, backup_model_path)
                    print(f"Backed up old best model to {backup_model_path}")
                except OSError as e:
                    print(f"Error backing up old best model: {e}")
            
            agent.save_model("best.keras")
        
        # Save checkpoint model
        if save_model_every_n_episodes and episode % save_model_every_n_episodes == 0:
            print(f'Saving checkpoint model at episode {episode} (score={current_episode_final_score})')
            agent.save_model(f"model_episode_{episode}.keras")

    # After training loop
    # Plot and save reward graph
    plt.figure(figsize=(12, 6))
    plt.plot(all_episode_scores, label='Episode Score')
    # Calculate and plot moving average
    if len(all_episode_scores) >= log_every:
        moving_avg = np.convolve(all_episode_scores, np.ones(log_every)/log_every, mode='valid')
        plt.plot(np.arange(log_every -1, len(all_episode_scores)), moving_avg, label=f'{log_every}-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Score')
    plt.title('Training Progress - Cumulative Score per Episode')
    plt.legend()
    plt.savefig('reward_log.png')
    print("Saved reward_log.png")


if __name__ == "__main__":
    dqn()
