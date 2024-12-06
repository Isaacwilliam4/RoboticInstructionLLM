import json
import time
import argparse
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
# from multigrid.find_shape import FindShape20x20Env
from multigrid.gym_multigrid.envs.find_shape import FindShape20x20Env
from State_Processing_Module.state_processing import StateProcessingModule
from RL_Block import AgentCoallition

HUMAN_READABLE = True
DEBUGGING = False


def main(args):
    # Read instructions
    with open(args.instructions_path, 'r') as file:
        data = json.load(file)

    # Initialize environment and modules
    env = FindShape20x20Env(
        render_mode='rgb_array' if (HUMAN_READABLE and DEBUGGING) else 'human',
        num_agents=args.num_agents,
        num_balls=random.randint(5, 12),
        num_boxes=random.randint(5, 12),
        view_size=args.agent_view_size,
        seed=args.seed,
    )
    state_processor = StateProcessingModule(num_agents=env.num_agents, view_size=env.agent_view_size)
    hive = AgentCoallition(
        num_agents=env.num_agents,
        agent_view_size=env.agent_view_size,
        state_processor=state_processor,
        action_space_size=env.action_space.n,
        train_every_n_iters=args.train_every_n_iters
    )

    # Train agents
    for task in data:
        for ep in range(args.num_episodes):
            instruction = task['instruction']
            targets = task['targets']
            shapes = [x['shape'] for x in targets]
            colors = [x['color'] for x in targets]

            print(f'Episode {ep}/{args.num_episodes}| instruction : {instruction} | shape : {shapes} | color : {colors}')
            done = True

            while done:
                state = env.reset()
                done = env._task_complete(colors, shapes)

            done = False
            num_steps = 1
            while not done:
                if HUMAN_READABLE and DEBUGGING:
                    img = env.render(mode='rgb_array')
                    # redis_client.lpush('frames', pickle.dumps(img))
                elif HUMAN_READABLE:
                    env.render(mode='human')

                actions, action_probs = hive.get_actions(state, instruction)
                next_state, reward, done, _ = env.step(actions, (shapes, colors), num_steps)
                hive.remember(state, actions, reward, next_state, done, instruction)

                state = next_state
                num_steps += 1
                if num_steps >= args.max_steps:
                    done = True

            hive.train()

            # Plot action probabilities
            log_prob_numpy = np.mean(action_probs.detach().cpu().numpy(), axis=0) 
            plt.clf()
            plt.bar(['still', 'left', 'right', 'forward', 'pickup', 'drop', 'activate', 'done'], log_prob_numpy)
            plt.xlabel('actions')
            plt.ylabel('action probability')
            plt.title(f'Episode {ep}')
            plt.show()
            plt.savefig('./log_probs.png')
            plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run task-based training with RL agents.')
    parser.add_argument('instructions_path', type=str, help='Path to the instructions JSON file')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to train')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--agent_view_size', type=int, default=7, help='Size of the agent\'s view')
    parser.add_argument('--num_agents', type=int, default=3, help='Number of agents in the environment')
    parser.add_argument('--seed', type=int, default=2, help='Random seed for reproducibility')
    parser.add_argument('--train_every_n_iters', type=int, default=1, help='Frequency of training iterations')
    parser.add_argument('--prob_fig_path', type=str, help="Save the plot of the action probabilities to this path")
    args = parser.parse_args()

    main(args)
