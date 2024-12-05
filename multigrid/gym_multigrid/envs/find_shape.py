import gym
import numpy as np
from gym_multigrid.multigrid import *
from gym import spaces

COLORS = ['red', 'blue', 'green', 'yellow']
SHAPES = ['ball', 'box']

class FindShapeEnv(MultiGridEnv):
    def __init__(self, size=20, num_balls=10, num_boxes=10, view_size=7, num_agents=3, **kwargs):
        self.num_agents = num_agents
        self.num_balls = num_balls
        self.num_boxes = num_boxes
        self.size = size
        self.prev_loc = [None for _ in range(num_agents)]

        # Initialize the world
        self.world = World()

        # Create agents
        agents = [Agent(self.world, i + 1, view_size=view_size) for i in range(self.num_agents)]

        # Explicitly set agent_view_size before calling the parent class
        self.agent_view_size = view_size

        # Initialize the parent class with the correct view_size
        super().__init__(
            grid_size=size,
            max_steps=500,
            agents=agents,
            **kwargs
        )

        # Define observation space
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(view_size, view_size, 12), dtype=np.uint8
        )

        # Initialize variables to track the agents' progress
        self.target_found = [False] * self.num_agents  # Track if agent has found their target
        self.first_found = [False] * self.num_agents  # Track if agent has found it first

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
            index = np.random.random_integers(len(COLORS)-1)  # Choose a random index
            self.place_obj(Ball(self.world, index=index))
        for _ in range(self.num_boxes):
            color = np.random.choice(COLORS)
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
        self._gen_grid(self.size, self.size)
        self.target_found = [False] * self.num_agents
        self.first_found = [False] * self.num_agents
        obs = super().reset()
        obs = self.process_observation(obs)
        return obs.reshape(self.agent_view_size, self.agent_view_size, -1)  # Reshape to match observation space
    
    def is_right_shape(self, shapes, obj):
        for shape in shapes:
            if shape == "ball" and isinstance(obj, Ball) \
            or shape == "box" and isinstance(obj, Box) \
            or shape == "na":
                return True
        return False

    def is_right_color(self, colors, obj):
        for color in colors:
            if obj.color == color or color == 'na':
                return True 
        return False
    
    def _handle_pickup(self, front_pos, colors, shapes, rewards, agent_idx):
        obj = self.grid.get(*front_pos)
        
        if obj is not None and obj.can_pickup():
            if self.is_right_color(colors, obj):
                if self.is_right_shape(shapes, obj):
                    rewards[agent_idx] += 1000
                    return
                #give some reward for right color
                rewards[agent_idx] += 100
            elif self.is_right_shape(shapes, obj):
                #give some reward for right shape
                rewards[agent_idx] += 100
            else:
                #give small reward for collecting object
                rewards[agent_idx] += 10
            #remove object
            self.grid.set(*front_pos, None)
            self.render()
        # #encourage picking up
        # rewards[agent_idx] += 1
    
    def _task_complete(self, colors, shapes):
        done = True

        for i in range(self.grid.width):
            for j in range(self.grid.height):
                obj = self.grid.get(i,j)
                if obj is not None:
                    for k in range(len(colors)):
                        color = colors[k]
                        shape = shapes[k]
                        if obj.color == color or color == 'na':
                            if shape == "ball" and isinstance(obj, Ball) \
                            or shape == "box" and isinstance(obj, Box)\
                            or shape == 'na':
                                done = False 
                                break
                if not done:
                    break
            if not done:
                break
        return done

    def step(self, actions, task_type, num_steps):
        shapes, colors = task_type
        # Ensure actions is an iterable (list or array)
        if isinstance(actions, np.int64):
            actions = [actions]  # Convert single action to a list

        obs, rewards, _, info = super().step(actions)

        # Reward logic for the agents
        for idx, agent in enumerate(self.agents):

            if self.prev_loc[idx] is not None:
                if not np.array_equal(np.array(self.prev_loc[idx]), np.array(agent.pos)):
                    #give reward for moving
                    rewards[idx] += 1
            else:
                self.prev_loc[idx] = agent.pos

            self.prev_loc[idx] = agent.pos            

            #handle pickup
            if actions[idx] == 4:
                front_pos = agent.front_pos
                self._handle_pickup(front_pos, colors, shapes, rewards, idx)

        done = self._task_complete(colors, shapes)
        obs = self.process_observation(obs)
        return obs.reshape(self.agent_view_size, self.agent_view_size, -1), np.sum(rewards) / num_steps, done, info  # Reshape to match observation space


    def render(self, mode="human"):
        """
        Render the grid to the console or use Matplotlib.
        """
        return super().render(mode)


class FindShape20x20Env(FindShapeEnv):
    """
    A 20x20 version of the FindShape environment.
    """
    def __init__(self, render_mode='human', **kwargs):
        # Ensure that view_size is passed down to the parent class properly
        view_size = kwargs.pop('view_size', 7)  # Default to 7 if not provided
        super().__init__(size=20, view_size=view_size, **kwargs)  # Pass view_size to parent class
        self.render_mode = render_mode
        self.kwargs = kwargs

    def render(self, mode="human"):
        return super().render(mode=mode)


# Register the environment for Gymnasium
import gymnasium as gym
gym.envs.registration.register(
    id="FindShape20x20-v0",
    entry_point="find_shape:FindShape20x20Env",  # Update path to match your file structure
)
