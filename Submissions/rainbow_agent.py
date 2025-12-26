import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os



# Noisy Linear Layer
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            self.reset_noise()
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, 
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)



# Rainbow CNN (Dueling Architecture + Noisy Linear Layers)
class RainbowCNN(nn.Module):
    def __init__(self, state_shape, action_shape, device='cpu'):
        super().__init__()
        self.device = device
        c, h, w = state_shape
        
        # Feature Extractor (CNN)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten())

        self.flatten_dim = 128 * h * w

        # Dueling Architecture (Value and Advantage streams)
        self.value_stream = nn.Sequential(
            NoisyLinear(self.flatten_dim, 512), nn.ReLU(),
            NoisyLinear(512, 1))
        
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.flatten_dim, 512), nn.ReLU(),
            NoisyLinear(512, action_shape))

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        
        # Feature Extraction (CNN)
        features = self.conv(obs)
        
        # Dueling Architecture
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Apply action mask for invalid actions
        mask = None
        if "action_mask" in info:
            mask = info["action_mask"]
            
        q_values = apply_mask_to_logits(q_values, mask, self.device)

        return q_values, state



def apply_mask_to_logits(logits, mask, device):

    if mask is None:
        return logits
    
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.bool, device=device)
    
    huge_negative = torch.tensor(-1e9, device=device)

    return torch.where(mask, logits, huge_negative)

def check_winning_move(board, col, mark, config):
    rows, columns = config.rows, config.columns

    empty_rows = np.where(board[:, col] == 0)[0]
    if len(empty_rows) == 0:
        return False
    row = empty_rows[-1]
        
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 0
        for step in range(-3, 4):
            r_check = row + step * dr
            c_check = col + step * dc
            
            if 0 <= r_check < rows and 0 <= c_check < columns:
                if (r_check == row and c_check == col) or board[r_check][c_check] == mark:
                    count += 1
                    if count >= 4: return True
                else:
                    count = 0
            else:
                count = 0
    return False



TRAINED_MODEL = None
DEVICE = 'cpu'

def load_model():
    global TRAINED_MODEL

    model = RainbowCNN(state_shape=(3, 6, 7), action_shape=7)
    model_path = os.path.join(os.path.dirname(__file__), 'best_rainbow_model.pth')

    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
        return None

def rainbow_agent(observation, configuration):
    global TRAINED_MODEL
    if TRAINED_MODEL is None:
        TRAINED_MODEL = load_model()
    
    board = np.array(observation['board']).reshape(configuration.rows, configuration.columns)
    me = observation.mark
    opponent = 3 - me
    valid_moves = [c for c in range(configuration.columns) if board[0][c] == 0]
    
    # Check for winning move
    for col in valid_moves:
        if check_winning_move(board, col, me, configuration):
            return col

    # Check for opponent winning move
    for col in valid_moves:
        if check_winning_move(board, col, opponent, configuration):
            return col

    # If no winning move, use the trained model to make a move
    net_board = np.copy(board)
    if me == 2:
        net_board[board == 1] = 2
        net_board[board == 2] = 1
    
    layer_me = (net_board == 1)
    layer_opponent = (net_board == 2)
    layer_empty = (net_board == 0)
    state = np.stack([layer_me, layer_opponent, layer_empty])
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    mask_bool = [False] * configuration.columns
    for col in valid_moves:
        mask_bool[col] = True
    mask_tensor = torch.tensor(mask_bool, dtype=torch.bool).to(DEVICE).unsqueeze(0)
    
    with torch.no_grad():
        q_values, _ = TRAINED_MODEL(state_tensor, info={"action_mask": mask_tensor})
        q_values = q_values.squeeze()
        best_move = int(torch.argmax(q_values).item())
    
    if best_move not in valid_moves:
        best_move = int(np.random.choice(valid_moves))

    return best_move       