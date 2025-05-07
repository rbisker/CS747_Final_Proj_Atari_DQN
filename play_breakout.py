import gymnasium as gym
import ale_py

env = gym.make("ALE/Breakout-v5", render_mode="human")
obs, _ = env.reset()

done = False
while not done:
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    done = terminated or truncated

env.close()