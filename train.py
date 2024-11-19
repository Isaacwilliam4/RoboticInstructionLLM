import json
import time
import argparse
# from multigrid.find_shape import FindShape20x20Env
from multigrid.gym_multigrid.envs.find_shape import FindShape20x20Env
from State_Processing_Module.state_processing import StateProcessingModule

def main(args):
    instructions_path = args.instructions_path
    with open(instructions_path, 'r') as file:
        data = json.load(file)

    state_processor = StateProcessingModule()

    for task in data:
        instruction = task['instruction']
        targets = task['targets'][0]
        shape = targets['shape']
        color = targets['color']

        env = FindShape20x20Env()

        nb_agents = len(env.agents)

        while True:
            env.render(mode='human')
            time.sleep(0.1)

            ac = [env.action_space.sample() for _ in range(nb_agents)]

            obs, rewards, done, info = env.step(ac, (shape, color))

            multimodal_fusion = state_processor.multimodal_fusion(instruction, obs)

            if done:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run task-based training with RL agents.')
    parser.add_argument('instructions_path', type=str, help='Path to the instructions JSON file')
    args = parser.parse_args()

    main(args)
