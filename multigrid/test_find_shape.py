import gymnasium as gym
import gym_multigrid
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

# Load the custom environment
from gym_multigrid.envs.find_shape import FindShape20x20Env


def visualize_env(env, model):
    # Reset the environment and get the initial observation
    obs = env.reset()  # This should give a numpy array or dictionary, not a tuple

    # If using VecEnv, `obs` might need to be unwrapped or handled as a batch
    if isinstance(obs, tuple):  # Check if the observation is a tuple (common in VecEnv)
        obs = obs[0]  # Extract the actual observation from the tuple

    done = False
    while not done:
        # Predict the action
        action, _ = model.predict(obs, deterministic=True)  # Predict action

        # Ensure action is a list or numpy array, even if it's scalar
        if isinstance(action, np.ndarray):  # If action is a numpy array
            action = action.tolist()  # Convert numpy array to list
        elif isinstance(action, int):  # If action is an integer
            action = [action]  # Wrap it in a list
        elif isinstance(action, list):  # If action is already a list
            action = np.array(action)  # Convert list to numpy array if needed

        # Debugging: Print the type and contents of the action

        # Make sure the action is an array-like (list or numpy array)
        action = np.array(action)

        # Debugging: Check the type of the action after conversion


        # Ensure that actions are passed as a batch (even if it's just one action)
        # Here we explicitly wrap the action in a list to pass to `env.step()`
        action = [action]  # Wrap the action in a list

        # Debugging: Check the type of the action before passing to `step()`


        # Take a step in the environment and debug the result
        result = env.step(action)  # Pass the action to the environment

        # Debugging: Print the result from env.step()


        # Unpack the result based on the number of values
        # If the result is a tuple or list, unpack it accordingly
        if isinstance(result, tuple):
            obs, reward, done, info = result[:4]  # Handle the first 4 elements
        else:
            obs, reward, done, info = result  # Assuming this returns only 4 values

        env.render()







# Main training function
def main():
    # Create and wrap the environment
    env = make_vec_env(
        FindShape20x20Env,
        n_envs=1,
        env_kwargs={"render_mode": "human"}  # Set render mode if needed
    )
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    # Create PPO model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_find_shape_tensorboard/")

    # Train the model
    print("Training the model...")
    model.learn(total_timesteps=10000)

    # Save the model
    model.save("ppo_find_shape")
    print("Model saved!")

    # Load the model for evaluation
    model = PPO.load("ppo_find_shape")

    # Visualization
    visualize_env(env.envs[0].unwrapped, model)  # Unwrap if needed


if __name__ == "__main__":
    main()
