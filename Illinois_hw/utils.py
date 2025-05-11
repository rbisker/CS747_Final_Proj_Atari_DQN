import random
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from config import *
import torch
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def find_max_lives(env):
    _, info = env.reset()
    return info['lives']
    

def check_live(life, cur_life):
    return life > cur_life

def get_frame(X):
    x = np.uint8(resize(rgb2gray(X), (HEIGHT, WIDTH), mode='reflect') * 255)
    return x

# # As opposed to get_frame, crops out scoreboard at top and black space on bottom
# def new_get_frame(X):
#     # Convert to grayscale
#     gray = rgb2gray(X)
#     # Crop out top 20 and bottom 5 rows
#     gray_cropped = gray[20:-5, :]  # from (210, 160) → (185, 160)
#     # Resize to (HEIGHT, WIDTH)
#     resized = resize(gray_cropped, (HEIGHT, WIDTH), mode='reflect')
#     return np.uint8(resized * 255)

def get_init_state(history, s, history_size):
    frame = get_frame(s)
    for i in range(history_size):
        history[i, :, :] = frame

# def new_get_init_state(history, s, history_size):
#     frame = new_get_frame(s)
#     for i in range(history_size):
#         history[i, :, :] = frame

def do_random_actions(env, num_actions):
    for _ in range(num_actions):
        random_action = random.choice(TRAINABLE_ACTIONS)
        obs, _, _, _, _ = env.step(random_action)
    return obs

def reset_after_life_loss(env, history):
    """
    Resets the environment state after a life is lost, without resetting the entire episode.
    Ensures the ball is fired and the history stack is rebuilt.
    """
    # Take a few random non-FIRE actions to vary paddle position
    num_random_actions = random.randint(5, 25)
    obs = None
    for _ in range(num_random_actions):  
        obs, _, _, _, _ = env.step(random.choice(TRAINABLE_ACTIONS))  #do some random actions before firing to increase variety
        # obs, _, _, _, _ = env.step(0)

    # Try firing the ball, detect success via frame differencing
    max_attempts = 5
    prev_frame = get_frame(obs)
    for i in range(max_attempts):
        obs, _, _, _, _ = env.step(1)  # FIRE
        curr_frame = get_frame(obs)
        if np.abs(curr_frame - prev_frame).sum() > 10:
            break  # Ball is likely launched
        prev_frame = curr_frame

    # If FIRE action failed, restart the episode
    if i == max_attempts - 1:
        print("[DEBUG]: FIRE action failed after", max_attempts, "attempts.  Resetting episode...") 
        return obs, True

    # Rebuild history stack using the current post-FIRE frame
    frame = get_frame(obs)
    history[:] = frame  # fills all history frames with the same post-FIRE frame

    return obs, False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


# def compute_optical_flow(prev: np.ndarray, curr: np.ndarray):
#     """
#     Compute Farneback optical flow between two grayscale frames.
#     Inputs should be shape (84, 84), dtype float32, range [0,1]
#     Returns 2 arrays: dx and dy (both shape 84x84)
#     """
#     flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
#                                         pyr_scale=0.5, levels=3, winsize=15,
#                                         iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
#     dx = flow[..., 0]
#     dy = flow[..., 1]
#     return dx, dy




