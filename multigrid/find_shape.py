import gym
import numpy as np
from gym_multigrid.multigrid import *
from gym import spaces

class FindShapeEnv(MultiGridEnv):
    def __init__(self, size=20, num_balls=10, num_boxes=10, view_size=7):
        self.num_balls = num_balls
        self.num_boxes = num_boxes
        self.colors = ['red', 'blue', 'green', 'yellow']
        self.agent_tasks = [("ball", "red"), ("box", "blue")]  # Tasks for each agent

        # Initialize the world
        self.world = World()

        # Create agents
        agents = [Agent(self.world, i + 1, view_size=view_size) for i in range(len(self.agent_tasks))]

        # Explicitly set agent_view_size before calling the parent class
        self.agent_view_size = view_size

        # Initialize the parent class with the correct view_size
        super().__init__(
            grid_size=size,
            max_steps=500,
            agents=agents,
        )

        # Define observation space
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(view_size, view_size, 12), dtype=np.uint8
        )

        # Initialize variables to track the agents' progress
        self.target_found = [False] * len(self.agent_tasks)  # Track if agent has found their target
        self.first_found = [False] * len(self.agent_tasks)  # Track if agent has found it first

    def _gen_grid(self, width, height):
        # Create grid
        self.grid = Grid(width, height)

        # Generate horizontal walls (top and bottom)
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        # Place balls and boxes randomly
        for _ in range(self.num_balls):
            index = np.random.random_integers(len(self.colors))  # Choose a random index
            self.place_obj(Ball(self.world, index=index))
        for _ in range(self.num_boxes):
            color = np.random.choice(self.colors)
            self.place_obj(Box(self.world, color=color))

        # Place agents randomly
        for agent in self.agents:
            self.place_agent(agent)

    def process_observation(self, obs):
        """
        Flatten or preprocess the observation to make it compatible with VecEnv.
        """
        if isinstance(obs, dict):
            return np.concatenate([v.flatten() for v in obs.values()])
        elif isinstance(obs, np.ndarray):
            return obs.flatten()
        elif isinstance(obs, list):
            # Convert list to a NumPy array and flatten it
            return np.array(obs, dtype=np.float32).flatten()
        else:
            raise ValueError(f"Unsupported observation type: {type(obs)}")

    def reset(self):
        # Reset the environment and the progress tracking variables
        self.target_found = [False] * len(self.agent_tasks)
        self.first_found = [False] * len(self.agent_tasks)
        obs = super().reset()
        obs = self.process_observation(obs)
        return obs.reshape(self.agent_view_size, self.agent_view_size, -1)  # Reshape to match observation space

    def step(self, actions):
        # Ensure actions is an iterable (list or array)
        if isinstance(actions, np.int64):
            actions = [actions]  # Convert single action to a list

        obs, rewards, done, info = super().step(actions)

        # Reward logic for the agents
        for idx, agent in enumerate(self.agents):
            # Get agent's task (e.g., "ball", "red" or "box", "blue")
            task_type, task_color = self.agent_tasks[idx]

            # Check if the agent found the object (e.g., red ball or blue box)
            found_object = False
            agent_pos = agent.pos  # Get agent's position

            # Assuming agent_pos is a numpy array [x, y]
            x, y = agent_pos[0], agent_pos[1]  # Extract x and y coordinates from numpy array

            # Get the object at the agent's current position
            current_cell = self.grid.get(x, y)  # Provide (x, y) coordinates

            # Ensure current_cell is iterable, even if it's a single object
            if not isinstance(current_cell, list):
                current_cell = [current_cell]

            for obj in current_cell:
                if task_type == "ball" and isinstance(obj, Ball) and obj.color == task_color:
                    found_object = True
                    break
                elif task_type == "box" and isinstance(obj, Box) and obj.color == task_color:
                    found_object = True
                    break

            if found_object and not self.target_found[idx]:
                self.target_found[idx] = True
                rewards[idx] += 10  # Reward for finding the target
                if not self.first_found[idx]:
                    self.first_found[idx] = True
                    rewards[idx] += 5  # Additional reward for being the first to find it

        obs = self.process_observation(obs)
        return obs.reshape(self.agent_view_size, self.agent_view_size, -1), rewards, done, info  # Reshape to match observation space


    def render(self, mode="human"):
        """
        Render the grid to the console or use Matplotlib.
        """
        if mode == "human":
            self.grid.render(self.agents)
        else:
            super().render(mode)


class FindShape20x20Env(FindShapeEnv):
    """
    A 20x20 version of the FindShape environment.
    """
    def __init__(self, render_mode='human'):
        # Ensure that view_size is passed down to the parent class properly
        super().__init__(size=20, view_size=7)  # Pass view_size to parent class
        self.render_mode = render_mode

    def render(self, mode="human"):
        super().render(mode=mode)


# Register the environment for Gymnasium
import gymnasium as gym
gym.envs.registration.register(
    id="FindShape20x20-v0",
    entry_point="find_shape:FindShape20x20Env",  # Update path to match your file structure
)
