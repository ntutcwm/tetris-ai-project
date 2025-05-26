import sys
import csv
import tensorflow as tf
import datetime

if not tf.config.list_physical_devices('GPU'):
    raise RuntimeError("CUDA GPU is not available. Please ensure CUDA is properly installed and configured.")

if len(sys.argv) < 2:
    exit("Missing model file")

from dqn_agent import DQNAgent
from tetris import Tetris

env = Tetris()
agent = DQNAgent(env.get_state_size(), modelFile=sys.argv[1])
agent.epsilon = 0 # Ensure deterministic behavior for evaluation
done = False

# Initialize variables for tracking stats
total_removed_lines = 0
total_played_steps = 0

while not done:
    next_states = {tuple(v): k for k, v in env.get_next_states().items()}
    if not next_states: # Handle case where no moves are possible
        done = True
        break
    
    best_state = agent.best_state(next_states.keys())
    if best_state is None: # Should not happen if next_states is not empty
        done = True
        break

    best_action = next_states[best_state]
    reward, done = env.play(best_action[0], best_action[1], render=True)

    # Update total stats after each play
    total_removed_lines = env.total_lines_cleared
    total_played_steps = env.played_steps

# After the game is over, write to CSV
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"tetris_custom_metric_submit_{timestamp}.csv"
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['id', 'removed_lines', 'played_steps']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'id': 0, 'removed_lines': total_removed_lines, 'played_steps': total_played_steps})
    writer.writerow({'id': 1, 'removed_lines': total_removed_lines, 'played_steps': total_played_steps})

print(f"Submission file '{output_file}' created successfully.")
print(f"Removed Lines: {total_removed_lines}, Played Steps: {total_played_steps}")
