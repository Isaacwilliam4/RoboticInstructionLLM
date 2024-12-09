# RoboticInstructionLLM

First you need to setup an environment, for this project it should be python 3.10 you can do this in conda with the following commands

```
conda create -n robotllm python=3.10 -y
conda activate robotllmenv
conda install -c conda-forge numpy matplotlib gymnasium transformers -y
conda install pytorch torchvision torchaudio -c pytorch
```

Then you need to install the following python packages with pip to get the environment to work correctly

```
cd multigrid/
pip install -e .
```

### Running the RL Agent Training Script

The `train.py` script is designed to train reinforcement learning agents to perform tasks based on instructions. This section provides an example of how to execute the script with various arguments.

#### Example Usage

To run the script, navigate to the project directory and use the following command in your terminal:

```bash
python train.py ./data/instructions.json --num_episodes 20 --max_steps 1000 --agent_view_size 7 --num_agents 3 --seed 42 --train_every_n_iters 1 --prob_fig_path ./log_probs.png
```

#### Explanation of Arguments:
- `./data/instructions.json`: The path to the JSON file containing task instructions for the agents.
- `--num_episodes 20`: Number of episodes for training (default is 10).
- `--max_steps 200`: Maximum number of steps allowed per episode (default is 100).
- `--agent_view_size 7`: The size of the agent's view (default is 7).
- `--num_agents 3`: Number of agents in the environment (default is 3).
- `--seed 42`: Random seed for reproducibility (default is 2).
- `--train_every_n_iters 1`: Specifies the frequency of training iterations (default is 1).
- `--prob_fig_path ./log_probs.png`: Path to save the plot of action probabilities after each episode.

## Attribution
Portions of this project were adapted from [DeepRL-Grounding](https://github.com/devendrachaplot/DeepRL-Grounding) by Chaplot et al, [MultiGrid](https://github.com/ini/multigrid), which are available under the MIT and Apache License.