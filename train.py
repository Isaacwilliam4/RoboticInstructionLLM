import json
import time
import argparse
# from multigrid.find_shape import FindShape20x20Env
from multigrid.gym_multigrid.envs.find_shape import FindShape20x20Env
from State_Processing_Module.state_processing import StateProcessingModule
from RL_Block import AgentCoallition
import random
### Debugging ###
# import redis
import pickle

HUMAN_READABLE = True
DEBUGGING = False

# redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

AGENT_VIEW_SIZE = 7
SEED = 2


def main(args):
    # TODO add num_episodes to the args
    num_episodes = 5
    max_steps = 100

    instructions_path = args.instructions_path
    with open(instructions_path, 'r') as file:
        data = json.load(file)


    task = data[0]
    # TODO is this the best way to train?  Or should we loop through all the instructions?
    # task = data[random.randint(0, len(data)-1)]
    instruction = task['instruction']
    targets = task['targets'][0]
    shape = targets['shape']
    color = targets['color']

    env = FindShape20x20Env(
        render_mode='rgb_array' if (HUMAN_READABLE and DEBUGGING) else 'human',
        num_agents=3,
        num_balls=random.randint(5, 12),
        num_boxes=random.randint(5, 12),
        view_size=AGENT_VIEW_SIZE,
        seed=SEED,
    )
    state_processor = StateProcessingModule(num_agents=env.num_agents, view_size=env.agent_view_size)
    # Can we access the size of the state from the env?
    hive = AgentCoallition(
        num_agents=env.num_agents,
        agent_view_size=env.agent_view_size,
        state_processor=state_processor,
        action_space_size=env.action_space.n,
        train_every_n_iters=1
    )
    # Actions:
    # available=['still', 'left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']

    # still = 0
    # # Turn left, turn right, move forward
    # left = 1
    # right = 2
    # forward = 3

    # # Pick up an object
    # pickup = 4
    # # Drop an object
    # drop = 5
    # # Toggle/activate an object
    # toggle = 6

    # # Done completing task
    # done = 7

    for ep in range(num_episodes):
        for task in data:
            instruction = task['instruction']
            targets = task['targets']
            shapes = [x['shape'] for x in targets]
            colors = [x['color'] for x in targets]

            print(f'Episode {ep}/{num_episodes}| instruction : {instruction} | shape : {shapes} | color : {colors}')
            # One iteration of this loop is one episode
            state = env.reset()
            done = False
            num_steps = 1
            while not done:

                if HUMAN_READABLE and DEBUGGING:
                    img = env.render(mode='rgb_array')
                    redis_client.lpush('frames', pickle.dumps(img))
                elif HUMAN_READABLE:
                    env.render(mode='human')
                    # time.sleep(0.1)


                actions = hive.get_actions(state, instruction)
                next_state, reward, done, _ = env.step(actions, (shapes, colors), num_steps)
                hive.remember(state, actions, reward, next_state, done, instruction)

                state = next_state

                num_steps += 1
                if num_steps >= max_steps:
                    done = True

            hive.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run task-based training with RL agents.')
    parser.add_argument('instructions_path', type=str, help='Path to the instructions JSON file')
    args = parser.parse_args()

    main(args)
