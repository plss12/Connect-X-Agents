import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time
import math

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# --- HELPER ---
class dotdict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

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
        return (board[0, :] == 0).astype(np.int8)

    def get_next_state(self, board, player, action):
        """
        Applies an action and returns the NEW board and the next player.
        Does not modify the original board (makes a copy).
        """
        # Find the first empty row in that column
        b = np.copy(board)
        for r in range(5, -1, -1):
            if b[r, action] == 0:
                b[r, action] = player
                break
        
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
        if self.check_win(board, player):
            return 1
        # Check win for the opponent
        if self.check_win(board, -player):
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
        return (player * board).astype(np.int8)

    def check_win(self, board, player):
        """
        Internal logic to check for 4 in a row.
        """
        t = (board == player)
        # Horizontal
        if (t[:, :-3] & t[:, 1:-2] & t[:, 2:-1] & t[:, 3:]).any():
            return True

        # Vertical
        if (t[:-3, :] & t[1:-2, :] & t[2:-1, :] & t[3:, :]).any():
            return True

        # Diagonals
        if np.any( t[:-3, :-3] & t[1:-2, 1:-2] & t[2:-1, 2:-1] & t[3:, 3:] ):
            return True

        if np.any( t[3:, :-3] & t[2:-1, 1:-2] & t[1:-2, 2:-1] & t[:-3, 3:] ):
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
        num_channels = args.get('num_channels', 32)

        # --- BODY ---
        # Initial Block: Converts the 3 input channels to 'num_channels'
        self.conv = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_channels)

        # Residual Tower: 4 Blocks
        self.res1 = ResidualBlock(num_channels)
        self.res2 = ResidualBlock(num_channels)

        # --- POLICY HEAD ---
        # Reduces depth and calculates movement probabilities
        self.p_conv = nn.Conv2d(num_channels, 16, kernel_size=1) 
        self.p_bn = nn.BatchNorm2d(16)
        self.p_fc = nn.Linear(16 * self.board_x * self.board_y, self.action_size)

        # --- VALUE HEAD ---
        # Reduces depth and calculates who wins (-1 to 1)
        self.v_conv = nn.Conv2d(num_channels, 3, kernel_size=1)
        self.v_bn = nn.BatchNorm2d(3)
        self.v_fc1 = nn.Linear(3 * self.board_x * self.board_y, 32)
        self.v_fc2 = nn.Linear(32, 1)

    def forward(self, s):
        # s: input state (Batch, 3, 6, 7)
        
        # 1. Body
        x = F.relu(self.bn(self.conv(s)))   # Input
        x = self.res1(x)                    # ResBlock 1
        x = self.res2(x)                    # ResBlock 2

        # 2. Policy Head
        p = F.relu(self.p_bn(self.p_conv(x)))            # Input
        p = p.view(-1, 16 * self.board_x * self.board_y) # Flatten
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
        # Convert the board from (6,7) to (3,6,7)
        encoded_board = np.stack([
            (board == 1).astype(np.float32),
            (board == -1).astype(np.float32),
            (board == 0).astype(np.float32)
        ])
        board_tensor = torch.from_numpy(encoded_board).unsqueeze(0)

        with torch.no_grad():            
            pi, v = self(board_tensor)

        # pi returns LogSoftmax, v returns Tanh
        return torch.exp(pi).detach().cpu().numpy()[0], v.item()

class NNetWrapper: 
    """
    Wrapper to use the JIT model compiled in the MCTS.
    """
    def __init__(self, traced_model):
        self.model = traced_model
    
    def predict(self, board):
        """
        Fast inference for MCTS
        """
        # Convert the board from (6,7) to (3,6,7)
        encoded_board = np.stack([
            (board == 1).astype(np.float32),
            (board == -1).astype(np.float32),
            (board == 0).astype(np.float32)
        ])
        board_tensor = torch.from_numpy(encoded_board).unsqueeze(0)
        
        with torch.no_grad():
            pi, v = self.model(board_tensor)

        # pi returns LogSoftmax, v returns Tanh
        return torch.exp(pi).detach().numpy()[0], v.item()

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

        valid_moves = self.game.get_valid_moves(canonicalBoard)

        # Apply temperature
        # Competitive (Deterministic)
        if temp == 0:
            best_counts = [c if valid else -1 for c, valid in zip(counts, valid_moves)]

            bestAs = np.array(np.argwhere(best_counts == np.max(best_counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs, sims

        # Exploration (Stochastic)
        # If temp > 0, normalize the visits to obtain a distribution
        counts = [x ** (1. / temp) for x in counts]
        counts = [c * v for c, v in zip(counts, valid_moves)]
        counts_sum = float(sum(counts))

        if counts_sum == 0:
            total_valid_moves = np.sum(valid_moves)
            if total_valid_moves > 0:
                probs = [1/total_valid_moves if valid else 0 for valid in valid_moves]
                return probs
            else:
                return [1/len(counts)] * len(counts)

        probs = [x / counts_sum for x in counts]
        return probs, sims

    def search(self, canonicalBoard):
        """
        Recursive function that goes down the tree, expands leaves and backpropagates values.
        """
        s = canonicalBoard.tobytes()

        # 1. Game ended -> Return the result
        es = self.Es.get(s)
        if es is not None :
            if es != 0:
                return -es
        else:
            es = self.game.get_game_ended(canonicalBoard, 1)
            self.Es[s] = es
            if es != 0:
                return -es
        
        # 2. Expert knowledge: Attack and Defense
        valid_moves = self.Vs.get(s)
        if valid_moves is None:
            valid_moves = self.game.get_valid_moves(canonicalBoard)
            self.Vs[s] = valid_moves

        # Attack: Check if player can win and stop the search if so
        winning_move = self._check_win_inplace(canonicalBoard, 1, valid_moves)
        if winning_move is not None:
            self._update_stats(s, winning_move, 1)
            return -1 
        
        # Defense: Check if player can block opponent from winning and prune the tree if so
        blocking_move = self._check_win_inplace(canonicalBoard, -1, valid_moves)
        best_act = -1

        if blocking_move is not None:
            best_act = blocking_move

        else:
            # 3. New leaf -> Expand and backpropagate the nn value
            ps = self.Ps.get(s)
            if ps is None:
                pi, v = self.nnet.predict(canonicalBoard)
                
                # Mask for filtering invalid moves
                pi = pi * valid_moves 
                sum_pi = np.sum(pi)
                
                # Re-normalize and assign uniform probability to valid moves
                if sum_pi > 0:
                    pi /= sum_pi 
                else:
                    pi = valid_moves / np.sum(valid_moves)

                self.Ps[s] = pi
                self.Ns[s] = 0
                
                return -v

            # 4. Known Node -> Selection using PUCT
            cur_best = -float('inf')

            # PUCT Formula: U(s,a) = Q(s,a) + cpuct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            cpuct = self.args.cpuct
            sqrt_Ns = math.sqrt(self.Ns[s])

            valid_inds = np.where(valid_moves)[0]
            for a in valid_inds:
                qsa = self.Qsa.get((s, a), a)
                if qsa is not None:
                    nsa = self.Nsa.get((s, a), 0)
                    u = qsa + cpuct * ps[a] * sqrt_Ns / (1 + nsa)
                else:
                    u = cpuct * ps[a] * sqrt_Ns + 1e-8 

                if u > cur_best:
                    cur_best = u
                    best_act = a

            a = best_act

        # 5. Recursion -> Go down to the next level
        a = best_act
        next_s, next_player = self.game.get_next_state(canonicalBoard, 1, a)
        next_s = self.game.get_canonical_form(next_s, next_player)

        v = self.search(next_s)

        # Discounting factor for distant future rewards
        v = self.args.gamma * v

        # 6. Backpropagation -> Update moving average Q = (N*Q + v) / (N+1) and N
        self._update_stats(s, a, v)

        return -v
        
    def _update_stats(self, s, a, v):
        """
        Helper function to perform backpropagation (update Qsa and Nsa).
        """
        key = (s, a)
        nsa = self.Nsa.get(key, 0)
        if nsa > 0:
            # Update moving average Q = (N*Q + v) / (N+1) and N
            self.Qsa[key] = (nsa * self.Qsa[key] + v) / (nsa + 1)
            self.Nsa[key] = nsa + 1
        else:
            self.Qsa[key] = v
            self.Nsa[key] = 1
        
        self.Ns[s] = self.Ns.get(s, 0) + 1
    
    def _check_win_inplace(self, board, player, valid_moves):
        """
        Simulates placing a piece in each valid column and checks for a win.
        """
        valid_cols = np.where(valid_moves)[0]
        
        for col in valid_cols:
            row_idx = -1
            for r in range(5, -1, -1):
                if board[r, col] == 0:
                    row_idx = r
                    break
            
            board[row_idx, col] = player
            is_win = self.game.check_win(board, player)
            board[row_idx, col] = 0
            
            if is_win:
                return col
                
        return None
    
    def _check_win(self, board, player, valid_moves):
        """
        Simulates placing a piece in each valid column and checks for a win.
        """
        valid_cols = np.where(valid_moves)[0]
        for col in valid_cols:
            temp_board, _ = self.game.get_next_state(board, player, col)
            if self.game.check_win_fast(temp_board, player):
                return col
        return None

# --- CONFIGURATION ---
MODEL_FILENAME = 'AlphaZero.pth'
NUM_CHANNELS = 32
DEVICE = 'cpu'
TIME_LIMIT = 1.9
ESTADISTICAS = False

GLOBAL_NET = None
GLOBAL_GAME = None
GLOBAL_ARGS = None
GLOBAL_MCTS = None

def load_model():
    """
    Load model looking in Kaggle or local paths.
    """
    kaggle_path = os.path.join('/kaggle_simulations/agent/', MODEL_FILENAME)
    system_path = os.path.join('models/', MODEL_FILENAME)
    model_path = kaggle_path if os.path.exists(kaggle_path) else system_path

    game = Connect4Game()
    args = dotdict({'num_channels': NUM_CHANNELS, 'cpuct': 2.0, 'gamma': 0.99, 'cuda': False})
    raw_nnet = Connect4NNet(game, args)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        model = torch.load(model_path, map_location=DEVICE)
        raw_nnet.load_state_dict(model)
        raw_nnet.to(DEVICE)
        raw_nnet.eval()
        nnet = torch.quantization.quantize_dynamic(raw_nnet, {nn.Linear}, dtype=torch.qint8, inplace=True)
        dummy_input = torch.randn(1, 3, 6, 7)
        traced_nnet = torch.jit.trace(raw_nnet, dummy_input)
        final_nnet = NNetWrapper(traced_nnet)
        return final_nnet, game, args

    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}") from e

# --- ALPHAZERO ---
def alphazero_agent(observation, configuration):
    global GLOBAL_NET, GLOBAL_GAME, GLOBAL_ARGS, GLOBAL_MCTS

    start_turn_time = time.time()
    current_time_limit = TIME_LIMIT

    # 1. Load model
    if GLOBAL_NET is None:
        GLOBAL_NET, GLOBAL_GAME, GLOBAL_ARGS = load_model()
        GLOBAL_MCTS = MCTS(GLOBAL_GAME, GLOBAL_NET, GLOBAL_ARGS)

        elapsed_loading = time.time() - start_turn_time
        current_time_limit = max(0.1, TIME_LIMIT - elapsed_loading)

    # 2. Prepare board
    board = np.array(observation.board, dtype=np.int8).reshape(configuration.rows, configuration.columns)
    me = observation.mark
    opponent = 3 - me

    # 3. Prepare canonical board
    canonical_board = np.zeros((6, 7), dtype=np.int8)
    canonical_board[board == me] = 1
    canonical_board[board == opponent] = -1

    # 4. Check for first move
    if observation.step == 0 and board[5, 3] == 0:
        if ESTADISTICAS:
            print("First better move: 3")
        return 3

    # 5. Check for immediate win or block
    valid_moves = GLOBAL_GAME.get_valid_moves(canonical_board)

    # Win Instantly?
    win_col = GLOBAL_MCTS._check_win_inplace(canonical_board, 1, valid_moves)
    if win_col is not None: 
        if ESTADISTICAS:
            print("Winning move: ", win_col)
        return int(win_col)

    # Block Instantly?
    block_col = GLOBAL_MCTS._check_win_inplace(canonical_board, -1, valid_moves)
    if block_col is not None:
        if ESTADISTICAS:
            print("Blocking move: ", block_col)
        return int(block_col)

    # 6. MCTS with time limit
    probs, sims = GLOBAL_MCTS.getActionProb(canonical_board, temp=0, time_limit_sec=current_time_limit)
    best_action = int(np.argmax(probs))

    if ESTADISTICAS:
        print(f"MCTS: {sims} simulations, best action: {best_action}")

    return best_action