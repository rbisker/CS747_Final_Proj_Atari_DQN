import gymnasium as gym
import ale_py

# Register Atari environments from ale_py
gym.register_envs(ale_py)

# Create the environment (using the updated environment name "ALE/Breakout-v5")
env = gym.make('ALE/Breakout-v5', render_mode='human')

# Reset environment
obs, info = env.reset()

# Track number of lives
lives = info.get('lives', 5)  # Default to 5 lives if not available

done = False
truncated = False
while not done and not truncated and lives > 0:
    # Take random actions and render the game
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    env.render()
    
    # Check for remaining lives (if 'lives' is in the info dict)
    lives = info.get('lives', lives)
    
    # If lives are 0, break the loop and close the environment
    if lives == 0:
        done = True

# Close the environment when finished
env.close()
print("Environment works and has been closed after losing all lives!")
