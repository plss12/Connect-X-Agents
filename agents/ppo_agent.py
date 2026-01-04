import numpy as np
import torch
import torch.nn as nn
import os



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



# Feature Extractor (CNN)
class FeatureExtractor(nn.Module):
    def __init__(self, state_shape, device='cpu'):
        super().__init__()
        self.device = device
        c, h, w = state_shape

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten())

        self.output_dim = 128 * h * w 

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        return self.conv(obs), state



# Actor: Choose action based on policy
class Actor(nn.Module):
    def __init__(self, preprocess_net, action_shape, device='cpu'):
        super().__init__()
        self.preprocess_net = preprocess_net
        self.last_layer = layer_init(nn.Linear(preprocess_net.output_dim, action_shape), std=0.01)
        self.device = device
        
    def forward(self, obs, state=None, info={}):
        features, _ = self.preprocess_net(obs, state)
        logits = self.last_layer(features)

        # Apply action mask for invalid actions
        if info is not None and "action_mask" in info:
            logits = apply_mask_to_logits(logits, info["action_mask"], self.device)

        return logits, state



def apply_mask_to_logits(logits, mask, device):

    if mask is None:
        return logits
    
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.bool, device=device)
    else:
        mask = mask.to(device)
    
    if logits.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1)


    return torch.where(mask, logits, -100)

def check_winning_move(board, col, mark):
    rows, columns = board.shape

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

    kaggle_path = "/kaggle_simulations/agent/model.pth"
    
    if os.path.exists(kaggle_path):
        model_path = kaggle_path
    else:
        model_path = "model.pth"

    base_net = FeatureExtractor(state_shape=(3, 6, 7), device=DEVICE)
    actor = Actor(base_net, action_shape=7, device=DEVICE)
    
    if os.path.exists(model_path):
        try:
            actor.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
            actor.eval()
            return actor
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
        return None

def ppo_agent(observation, configuration):
    global TRAINED_MODEL
    if TRAINED_MODEL is None:
        TRAINED_MODEL = load_model()

    board = np.array(observation.board).reshape(configuration.rows, configuration.columns)
    me = observation.mark
    opponent = 3 - me
    valid_moves = [c for c in range(configuration.columns) if board[0][c] == 0]

    # Check for winning move
    for col in valid_moves:
        if check_winning_move(board, col, me):
            return int(col)

    # Check for opponent winning move
    for col in valid_moves:
        if check_winning_move(board, col, opponent):
            return int(col)
    
    # If no winning move, use policy
    net_board = np.copy(board)
    if me == 2:
        net_board[board==1] = 2
        net_board[board==2] = 1

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
        logits, _ = TRAINED_MODEL(state_tensor, info={"action_mask": mask_tensor})
        logits = logits.squeeze()
        best_move = int(torch.argmax(logits).item())
    
    if best_move not in valid_moves:
        best_move = int(np.random.choice(valid_moves))
    
    return best_move