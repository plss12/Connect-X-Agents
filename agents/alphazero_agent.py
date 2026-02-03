import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time
import math

# --- HELPER ---
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

# --- GAME ---
class Connect4Game:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.win_length = 4

    def get_board_size(self):
        """Board shape: (rows, columns)"""
        return (self.rows, self.cols)

    def get_action_size(self):
        """Number of possible actions (7 columns)"""
        return self.cols

    def get_valid_moves(self, board):
        """Returns a binary vector of valid moves"""
        return (board[0, :] == 0).astype(int)

    def get_next_state(self, board, player, action):
        """
        Applies an action and returns the NEW board and the next player.
        Does not modify the original board (makes a copy).
        """
        # Validate action illegal moves
        if board[0, action] != 0:
            return board, player

        # Find the first empty row in that column
        b = np.copy(board)
        row = np.where(b[:, action] == 0)[0][-1]
        
        # Place piece (We use 1 for P1 and -1 for P2 in the internal logic)
        b[row, action] = player
        
        # Return new board and change turn (-player)
        return b, -player

    def get_game_ended(self, board, player):
        """
        Returns:
         1 if 'player' has won
        -1 if 'player' has lost
         1e-4 if it's a draw (small number different from 0)
         0 if the game has not ended
        """
        # Check win for the current player
        if self._check_win(board, player):
            return 1
        # Check win for the opponent
        if self._check_win(board, -player):
            return -1
        # Check draw (full board)
        if 0 not in board[0, :]:
            return 1e-4
        return 0

    def get_canonical_form(self, board, player):
        """
        Convert the board to the perspective of the current player
        so the neural network always sees the same input format
        """
        return player * board

    def _check_win(self, board, player):
        """Internal logic to check for 4 in a row"""
        # Horizontal
        for c in range(self.cols - 3):
            for r in range(self.rows):
                if board[r][c] == player and board[r][c+1] == player and board[r][c+2] == player and board[r][c+3] == player:
                    return True
        # Vertical
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if board[r][c] == player and board[r+1][c] == player and board[r+2][c] == player and board[r+3][c] == player:
                    return True
        # Diagonal Positive
        for c in range(self.cols - 3):
            for r in range(self.rows - 3):
                if board[r][c] == player and board[r+1][c+1] == player and board[r+2][c+2] == player and board[r+3][c+3] == player:
                    return True
        # Diagonal Negative
        for c in range(self.cols - 3):
            for r in range(3, self.rows):
                if board[r][c] == player and board[r-1][c+1] == player and board[r-2][c+2] == player and board[r-3][c+3] == player:
                    return True
        return False

# --- NEURAL NETWORK ---
class ResidualBlock(nn.Module):
    """
    Fundamental block of AlphaZero.
    Consists of: Conv -> BN -> ReLU -> Conv -> BN -> Residual Connection -> ReLU
    """
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class Connect4NNet(nn.Module):
    def __init__(self, game, args):
        super(Connect4NNet, self).__init__()
        
        # Board dimensions and action size
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

        # Arguments and hyperparameters
        self.args = args
        num_channels = args.get('num_channels', 64)

        # --- BODY ---
        # Initial Block: Converts the 3 input channels to 'num_channels'
        self.conv = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_channels)

        # Residual Tower: 4 Blocks
        self.res1 = ResidualBlock(num_channels)
        self.res2 = ResidualBlock(num_channels)
        self.res3 = ResidualBlock(num_channels)
        self.res4 = ResidualBlock(num_channels)

        # --- POLICY HEAD ---
        # Reduces depth and calculates movement probabilities
        self.p_conv = nn.Conv2d(num_channels, 32, kernel_size=1) 
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * self.board_x * self.board_y, self.action_size)

        # --- VALUE HEAD ---
        # Reduces depth and calculates who wins (-1 to 1)
        self.v_conv = nn.Conv2d(num_channels, 3, kernel_size=1)
        self.v_bn = nn.BatchNorm2d(3)
        self.v_fc1 = nn.Linear(3 * self.board_x * self.board_y, 32)
        self.v_fc2 = nn.Linear(32, 1)

    def forward(self, s):
        # s: input state (Batch, 3, 6, 7)
        
        # 1. Body
        x = F.relu(self.bn(self.conv(s))) # Input
        x = self.res1(x)                    # ResBlock 1
        x = self.res2(x)                    # ResBlock 2
        x = self.res3(x)                    # ResBlock 3
        x = self.res4(x)                    # ResBlock 4

        # 2. Policy Head
        p = F.relu(self.p_bn(self.p_conv(x)))             # Input
        p = p.view(-1, 32 * self.board_x * self.board_y) # Flatten
        p = self.p_fc(p)                                 # Linear
        p = F.log_softmax(p, dim=1)                      # LogSoftmax

        # 3. Value Head
        v = F.relu(self.v_bn(self.v_conv(x)))             # Input
        v = v.view(-1, 3 * self.board_x * self.board_y)   # Flatten
        v = F.relu(self.v_fc1(v))                         # Linear
        v = torch.tanh(self.v_fc2(v))                     # Tanh

        return p, v

    def predict(self, board):
        """
        Fast inference for MCTS
        """
        # Prepare input
        # Convert the board from (6,7) to (3,6,7)
        encoded_board = np.stack([
            (board == 1).astype(np.float32),
            (board == -1).astype(np.float32),
            (board == 0).astype(np.float32)
        ])
        board_tensor = torch.FloatTensor(encoded_board)
        if self.args.cuda: 
            board_tensor = board_tensor.contiguous().cuda()
        board_tensor = board_tensor.view(1, 3, board.shape[0], board.shape[1])
        
        self.eval()
        with torch.no_grad():
            pi, v = self(board_tensor)

        # pi returns LogSoftmax, v returns Tanh
        return torch.exp(pi).detach().cpu().numpy()[0], v.item()

# --- MCTS ---
class MCTS:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        
        # Dictionaries to store tree statistics
        self.Qsa = {}  # Average value -> Q values for (state, action)
        self.Nsa = {}  # Edge visits -> Number of times (state, action) visited
        self.Ns = {}   # Node visits -> Number of times (state) visited
        self.Ps = {}   # Initial policy -> Initial probability (state) returned by nnet
        
        self.Es = {}   # Game ended state cache (state)
        self.Vs = {}   # Valid moves mask (state)

    def getActionProb(self, canonicalBoard, temp=1, time_limit_sec=None):
        """
        Executes simulations until time limit is reached and returns the action probabilities.
        """
        start_time = time.time()
        sims = 0

        # Execute MCTS simulations until time limit is reached
        while True:
            if time.time() - start_time > time_limit_sec:
                break

            self.search(canonicalBoard)
            sims += 1

        s = canonicalBoard.tobytes()
        
        # Get the number of times each action was visited
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.get_action_size())]

        # Apply temperature
        # temp=1: Exploration (Stochastic)
        # temp=0: Competitive (Deterministic)
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs, sims

        # If temp > 0, normalize the visits to obtain a distribution
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs, sims

    def search(self, canonicalBoard):
        """
        Recursive function that goes down the tree, expands leaves and backpropagates values.
        """
        s = canonicalBoard.tobytes()

        # 1. Game ended -> Return the result
        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(canonicalBoard, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        # 2. New leaf -> Expand and backpropagate the nn value 
        if s not in self.Ps:
            pi, v = self.nnet.predict(canonicalBoard)
            
            # Mask for filtering invalid moves
            valid_moves = self.game.get_valid_moves(canonicalBoard)
            pi = pi * valid_moves 
            sum_pi = np.sum(pi)
            
            # Re-normalize and assign uniform probability to valid moves
            if sum_pi > 0:
                pi /= sum_pi 
            else:
                print("All valid moves were masked, making all valid moves equally probable!")
                pi = valid_moves / np.sum(valid_moves)

            self.Ps[s] = pi
            self.Ns[s] = 0
            self.Vs[s] = valid_moves
            
            return -v

        # 3. Known Node -> Selection using PUCT
        valid_moves = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # PUCT Formula: U(s,a) = Q(s,a) + cpuct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        cpuct = self.args.cpuct
        sqrt_Ns = math.sqrt(self.Ns[s])

        for a in np.where(valid_moves)[0]:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + cpuct * self.Ps[s][a] * sqrt_Ns / (1 + self.Nsa[(s, a)])
            else:
                u = cpuct * self.Ps[s][a] * sqrt_Ns + 1e-8 

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        
        # 4. Recursion -> Go down to the next level
        next_s, next_player = self.game.get_next_state(canonicalBoard, 1, a)
        next_s = self.game.get_canonical_form(next_s, next_player)

        v = self.search(next_s)

        # 5. Backpropagation -> Update moving average Q = (N*Q + v) / (N+1) and N
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

# --- HELPER WINNER FUNCTION ---

def check_winning_move(board, player, config):
    """
    Check if player can win in the next move.
    """
    valid_moves = [c for c in range(config.columns) if board[0][c] == 0]
    for col in valid_moves:
        temp_board = board.copy()
        row = np.max(np.where(temp_board[:, col] == 0))
        temp_board[row, col] = player
        
        # Check 4 in a row (Horizontal, Vertical, Diagonal)
        # Horizontal
        for r in range(config.rows):
            for c in range(config.columns - 3):
                if np.all(temp_board[r, c:c+4] == player): return col
        # Vertical
        for r in range(config.rows - 3):
            for c in range(config.columns):
                if np.all(temp_board[r:r+4, c] == player): return col
        # Diagonals
        for r in range(config.rows - 3):
            for c in range(config.columns - 3):
                if np.all([temp_board[r+i, c+i] == player for i in range(4)]): return col
                if np.all([temp_board[r+3-i, c+i] == player for i in range(4)]): return col
    return None

# --- CONFIGURATION ---
MODEL_FILENAME = 'best.pth.tar'
NUM_CHANNELS = 64
DEVICE = 'cpu'
TIME_LIMIT = 1.8

GLOBAL_NET = None
GLOBAL_GAME = None
GLOBAL_ARGS = None

def load_model():
    """
    Load model looking in Kaggle or local paths.
    """
    kaggle_path = os.path.join('/kaggle_simulations/agent/', MODEL_FILENAME)
    
    if os.path.exists(kaggle_path):
        model_path = kaggle_path
    else:
        model_path = MODEL_FILENAME

    game = Connect4Game()
    args = dotdict({'num_channels': NUM_CHANNELS, 'cpuct': 2.0, 'cuda': False})
    nnet = Connect4NNet(game, args)
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            nnet.load_state_dict(checkpoint['state_dict'])
            nnet.to(DEVICE)
            nnet.eval()
            return nnet, game, args
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None, None
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
        return None, None, None

# --- ALPHAZERO ---
def alphazero_agent(observation, configuration):
    global GLOBAL_NET, GLOBAL_GAME, GLOBAL_ARGS

    # 1. Load model
    if GLOBAL_NET is None:
        GLOBAL_NET, GLOBAL_GAME, GLOBAL_ARGS = load_model()

    # 2. Prepare board
    board = np.array(observation.board).reshape(configuration.rows, configuration.columns)
    me = observation.mark
    opponent = 3 - me

    # 3. Quick heuristic
    # Check for winning move
    win_col = check_winning_move(board, me, configuration)
    if win_col is not None:
        return int(win_col)
    
    # Check for opponent winning move
    block_col = check_winning_move(board, opponent, configuration)
    if block_col is not None:
        return int(block_col)
    
    # If no winning move, use model
    # 4. Prepare canonical board
    canonical_board = np.zeros((6, 7), dtype=int)
    canonical_board[board == me] = 1
    canonical_board[board == opponent] = -1

    # 5. MCTS with time limit
    mcts = MCTS(GLOBAL_GAME, GLOBAL_NET, GLOBAL_ARGS)
    probs, sims = mcts.getActionProb(canonical_board, temp=0, time_limit_sec=TIME_LIMIT)
    best_action = int(np.argmax(probs))

    print(f"MCTS: {sims} simulations, best action: {best_action}")

    return best_action