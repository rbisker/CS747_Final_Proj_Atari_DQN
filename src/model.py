import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from config import HEIGHT, WIDTH, lstm_seq_length

class DQN(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)  # Replaces .view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.head(x)

class DQN_LSTM(nn.Module):
    def __init__(self, action_size):
        super(DQN_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(256, action_size)
        # Define an LSTM layer

    def forward(self, x, hidden = None, train=True):
        if train==True: # If training, we will merge all the visual states into one. So first two dimensions (batch_size, lstm_seq_length) will be merged into one. This will let us process all these together.
            x = x.view(-1, 1, HEIGHT, WIDTH)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        if train==True: # We will reshape the output to match the original shape. So first dimension will be extended back to (batch_size, lstm_seq_length)
            x = x.view(-1, lstm_seq_length, 512)
        # Pass the state through an LSTM
        ### CODE ###

        return self.head(lstm_output), hidden
    
# class DQN_timediff(nn.Module):
#     def __init__(self, action_size):
#         super(DQN_timediff, self).__init__()
#         self.conv1 = nn.Conv2d(5, 32, kernel_size=8, stride=4)
#         self.bn1 = nn.BatchNorm2d(32)

#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.bn2 = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         self.bn3 = nn.BatchNorm2d(64)

#         self.fc = nn.Linear(3136, 512)
#         self.head = nn.Linear(512, action_size)

#         self.activation = nn.LeakyReLU(negative_slope=0.01)

#     def forward(self, x):
#         x = self.activation(self.bn1(self.conv1(x)))
#         x = self.activation(self.bn2(self.conv2(x)))
#         x = self.activation(self.bn3(self.conv3(x)))
#         x = self.activation(self.fc(x.view(x.size(0), -1)))
#         return self.head(x)
    

class DQN_DualBranch(nn.Module):
    def __init__(self, action_size):
        super().__init__()

        # === Full Frame Path (4 input channels) ===
        self.conv1_full = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=3)  # Output: 21x21
        self.bn1_full = nn.BatchNorm2d(32)
        self.conv2_full = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Maintain 21x21
        self.bn2_full = nn.BatchNorm2d(64)

        # === Diff Frame Path (2 input channels) ===
        self.conv1_diff = nn.Conv2d(2, 16, kernel_size=4, stride=2, padding=1)  # Output: (B, 16, 42, 42)
        self.bn1_diff = nn.BatchNorm2d(16)
        self.conv2_diff = nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1)  # Output: (B, 64, 21, 21)
        self.bn2_diff = nn.BatchNorm2d(64)


        # Combined conv layer to reduce to 10x10
        self.fuse_conv = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_fuse_conv = nn.BatchNorm2d(64)
        self.conv3_combined = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # Down to 11x11
        # self.conv3_combined = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)   #!!!! DEBUG ONLY !!!!
        self.bn3_combined = nn.BatchNorm2d(64)


        # Fully connected layers
        self.fc = nn.Linear(64 * 11 * 11, 512)
        self.head = nn.Linear(512, action_size)

    def forward(self, x):
        full_frames = x[:, :4]    # shape: (batch, 4, 84, 84)
        diff_frames = x[:, 4:]    # shape: (batch, 2, 84, 84)

        # Full path
        x_full = F.relu(self.bn1_full(self.conv1_full(full_frames)))  # 21x21
        x_full = F.relu(self.bn2_full(self.conv2_full(x_full)))       # 21x21

        # Diff path
        x_diff = F.leaky_relu(self.bn1_diff(self.conv1_diff(diff_frames)), negative_slope=0.1)
        x_diff = F.leaky_relu(self.bn2_diff(self.conv2_diff(x_diff)), negative_slope=0.1)


        # if self.training and random.random() < 0.0001:
        #     print(f"[BRANCH DEBUG] x_full mean: {x_full.mean().item():.4f}, std: {x_full.std().item():.4f} | "
        #     f"x_diff mean: {x_diff.mean().item():.4f}, std: {x_diff.std().item():.4f}")


        # Combine and downsample
        x_combined = x_full + x_diff # residual style fusion
        x_combined = F.relu(self.bn_fuse_conv(self.fuse_conv(x_combined))) # (B, 64, 21, 21)
        x_combined = F.relu(self.bn3_combined(self.conv3_combined(x_combined)))  # (B, 64, 11, 11)
        x_combined = torch.flatten(x_combined, start_dim=1)
        x_combined = F.relu(self.fc(x_combined))
        return self.head(x_combined)


