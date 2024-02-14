import gymnasium as gym
import imageio
import pathlib
import os
import torch

def record_image(agent_env, agent, num_iter, example_path):

    if not os.path.exists(example_path):
        pathlib.Path(example_path).mkdir(parents=True, exist_ok=True)

    env = gym.make(agent_env, render_mode="rgb_array")
    agent.actor.eval()
    state, info = env.reset()
    state = torch.tensor(state)
    done = False
    frames = []
    while not done:
        frames.append(env.render())
        action = agent.actor(state.view(1, -1))[0]
        next_state, reward, terminated, truncated, _ = env.step(action.tolist())
        done = terminated or truncated
        state = torch.tensor(next_state)
    env.close()
    
    imageio.mimsave(os.path.join(example_path, f'run{num_iter}.gif'), frames, fps=30)