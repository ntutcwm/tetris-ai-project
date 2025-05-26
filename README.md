# tetris-ai

A bot that plays [Tetris](https://en.wikipedia.org/wiki/Tetris) using deep reinforcement learning. This project is based on the work by Nuno Faria, with a directory structure placing the core AI logic любви `tetris_ai_nuno_faria/`.

## Demo

First 10000 points, after some training (original demo).

![Demo - First 10000 points](tetris_ai_nuno_faria/demo.gif)

## Requirements

The main dependencies are:
- Tensorflow
- Keras
- Opencv-python
- Numpy
- Pillow
- Tqdm
- Matplotlib

A detailed list of packages and specific versions, including notes on NVIDIA GPU setup (CUDA, cuDNN), can be found in the [`requirements.txt`](requirements.txt) file.

## Setup / Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_name>     # Replace <repository_name> with the actual directory name
    ```

2.  **Create and activate a Python virtual environment:**
    It's highly recommended to use a virtual environment. This project was tested with Python 3.8.
    ```bash
    python3 -m venv cwmenv
    source cwmenv/bin/activate
    ```
    (On Windows, activation is `cwmenv\Scripts\activate`)

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Please refer to the comments at the top of [`requirements.txt`](requirements.txt) for important notes on setting up NVIDIA drivers, CUDA, and cuDNN if you plan to use GPU acceleration.

## How to Run

All commands should be run from the root directory of the project.

### Training the Agent

To train the AI agent:
```shell
# Hyperparameters can be configured in tetris_ai_nuno_faria/run.py
python3 tetris_ai_nuno_faria/run.py
```
Training progress, logs, and models (including `best.keras` and periodic checkpoints like `model_episode_XXX.keras`) will be saved in the root directory and the `logs/` directory.

### Playing with a Trained Model

This repository includes a pre-trained model, `best.keras` (located in the root directory), which is the result of the latest training run. 
To play a game using this pre-trained `best.keras` model:
```shell
python3 tetris_ai_nuno_faria/run_model.py best.keras
```

If you train the agent yourself (see "Training the Agent" above), new `best.keras` and checkpoint models (e.g., `model_episode_XXX.keras`) will be generated. You can use them similarly:
```shell
# Example using a checkpoint model
python3 tetris_ai_nuno_faria/run_model.py model_episode_600.keras 
```
(Replace `model_episode_600.keras` with the desired checkpoint model file if you have run your own training.)

The original underlying project by Nuno Faria also mentioned a `sample.keras`. If you have this specific file from the original source, you could place it (e.g., in `tetris_ai_nuno_faria/`) and test it via:
```shell
python3 tetris_ai_nuno_faria/run_model.py tetris_ai_nuno_faria/sample.keras
```

### Viewing Logs with TensorBoard

To view training logs (metrics like score, epsilon, etc.):
```shell
tensorboard --logdir ./logs
```
Navigate to the URL provided by TensorBoard (usually `http://localhost:6006`).

## How does it work

(Content from original `tetris_ai_nuno_faria/README.md` - describes the core DQN algorithm)

#### Reinforcement Learning

At first, the agent will play random moves, saving the states and the given reward in a limited queue (replay memory). At the end of each episode (game), the agent will train itself (using a neural network) with a random sample of the replay memory. As more and more games are played, the agent becomes smarter, achieving higher and higher scores.

Since in reinforcement learning once an agent discovers a good 'path' it will stick with it, it was also considered an exploration variable (that decreases over time), so that the agent picks sometimes a random action instead of the one it considers the best. This way, it can discover new 'paths' to achieve higher scores.

#### Training

The training is based on the [Q Learning algorithm](https://en.wikipedia.org/wiki/Q-learning). Instead of using just the current state and reward obtained to train the network, it is used Q Learning (that considers the transition from the current state to the future one) to find out what is the best possible score of all the given states **considering the future rewards**, i.e., the algorithm is not greedy. This allows for the agent to take some moves that might not give an immediate reward, so it can get a bigger one later on (e.g. waiting to clear multiple lines instead of a single one).

The neural network will be updated with the given data (considering a play with reward *reward* that moves from *state* to *next_state*, the latter having an expected value of *Q_next_state*, found using the prediction from the neural network):

if not terminal state (last round): *Q_state* = *reward* + *discount* × *Q_next_state*
else: *Q_state* = *reward*

#### Best Action

Most of the deep Q Learning strategies used output a vector of values for a certain state. Each position of the vector maps to some action (ex: left, right, ...), and the position with the higher value is selected.

However, the strategy implemented was slightly different. For some round of Tetris, the states for all the possible moves will be collected. Each state will be inserted in the neural network, to predict the score obtained. The action whose state outputs the biggest value will be played.

#### Game State

It was considered several attributes to train the network. After several tests, a conclusion was reached that the following (or a subset) were important:

-   **Number of lines cleared**
-   **Number of holes**
-   **Bumpiness** (sum of the difference between heights of adjacent pairs of columns)
-   **Total Height**
-   (The agent also considers max height for rewards, though it might not be in the state vector by default)

#### Game Score (Reward Basis)

Each block placed yields 1 point. When clearing lines, the given score is $number\_lines\_cleared^2 \times board\_width$. Losing a game subtracts points. Additional penalties and rewards (for holes, bumpiness, height, combos) are defined in [`tetris_ai_nuno_faria/tetris.py`](tetris_ai_nuno_faria/tetris.py).

## Implementation Details

(Content from original `tetris_ai_nuno_faria/README.md`)

The code was implemented using `Python`. For the neural network, it was used the framework `Keras` with a `Tensorflow` backend.

#### Internal Structure (Agent)

The agent is formed by a deep neural network, with variable number of layers, neurons per layer, activation functions, loss function, optimizer, etc. Default configurations can be found in [`tetris_ai_nuno_faria/run.py`](tetris_ai_nuno_faria/run.py).

#### Training Parameters

Default training parameters (episodes, memory size, batch size, epsilon decay, etc.) are also defined in [`tetris_ai_nuno_faria/run.py`](tetris_ai_nuno_faria/run.py).

## Original Project Results (Example)

For 2000 episodes, with epsilon ending at 1500, the agent kept going for too long around episode 1460, so it had to be terminated. Here is a chart with the maximum score every 50 episodes, until episode 1450 (from original README):

![results](tetris_ai_nuno_faria/results.svg)

Note: Decreasing the `epsilon_end_episode` could make the agent achieve better results in a smaller number of episodes.

## Useful Links

(Content from original `tetris_ai_nuno_faria/README.md`)

#### Deep Q Learning
- PythonProgramming - https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/
- Keon - https://keon.io/deep-q-learning/
- Towards Data Science - https://towardsdatascience.com/self-learning-ai-agents-part-ii-deep-q-learning-b5ac60c3f47

#### Tetris AI
- Code My Road - https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/ (uses evolutionary strategies)
