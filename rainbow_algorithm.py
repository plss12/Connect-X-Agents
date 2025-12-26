import os
import math
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from connectXgym import ConnectXGym, apply_mask_to_logits, check_winning_move
import pyplAI
from pyplAI_algorithms import ConnectXState


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



# Rainbow agent with rules for instant win or loss
class RainbowAgent:
    def __init__(self, policy, env_rows=6, env_columns=7, device='cuda'):
        self.net = policy.model
        self.net.eval()
        self.env_rows = env_rows
        self.env_columns = env_columns
        self.device = device

    def __call__(self, observation, configuration):
        board = np.array(observation['board']).reshape(self.env_rows, self.env_columns)

        me = observation.mark
        opponent = 3 - me
        valid_moves = [c for c in range(self.env_columns) if board[0][c] == 0]

        # Check for winning move
        for col in valid_moves:
            if self.check_winning_move(board, col, me):
                return int(col)

        # Check for opponent winning move
        for col in valid_moves:
            if self.check_winning_move(board, col, opponent):
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
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        mask_bool = [False] * configuration.columns
        for col in valid_moves:
            mask_bool[col] = True
        mask_tensor = torch.tensor(mask_bool, dtype=torch.bool).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values, _ = self.net(state_tensor, info={"action_mask": mask_tensor})
            q_values = q_values.squeeze()

            best_move = int(torch.argmax(q_values).item())
        
        if best_move not in valid_moves:
            best_move = int(np.random.choice(valid_moves))
        
        return best_move

def minimax_lite_agent(observation, configuration):
    depth = 2

    current_board = observation.board
    current_player = observation.mark
        
    pyplai_state = ConnectXState(current_board, current_player)

    valid_moves = pyplai_state.get_moves()
    best_move = valid_moves[len(valid_moves)//2] if valid_moves else None
        
    minimax_solver = pyplAI.Minimax(
                    ConnectXState.apply_move,
                    ConnectXState.get_moves, 
                    ConnectXState.is_final_state, 
                    ConnectXState.wins_player, 
                    ConnectXState.heuristic,
                    2, 
                    depth)
            
    recommended_move = minimax_solver.ejecuta(pyplai_state)

    if recommended_move is not None:
        best_move = recommended_move

    return best_move



# Training function with self-play
def train_rainbow_self_play():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    log_path = "files_rainbow_self_play/logs"
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    
    model_path = "files_rainbow_self_play/models"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path)
    
    TOTAL_EPOCHS = 100
    UPDATE_OPPONENT_FREQ = 5

    train_envs = DummyVectorEnv([lambda: ConnectXGym(opponent='random', apply_symmetry=True),
                                 lambda: ConnectXGym(opponent='random', apply_symmetry=True),

                                 lambda: ConnectXGym(opponent='negamax', apply_symmetry=True),
                                 lambda: ConnectXGym(opponent='negamax', apply_symmetry=True),

                                 lambda: ConnectXGym(opponent=minimax_lite_agent, apply_symmetry=True),
                                 lambda: ConnectXGym(opponent=minimax_lite_agent, apply_symmetry=True),

                                 lambda: ConnectXGym(opponent='random', apply_symmetry=True),
                                 lambda: ConnectXGym(opponent='random', apply_symmetry=True),
                                 lambda: ConnectXGym(opponent='random', apply_symmetry=True),
                                 lambda: ConnectXGym(opponent='random', apply_symmetry=True)])

    test_envs_p1 = [lambda: ConnectXGym(opponent='negamax', switch_prob=0.0, apply_symmetry=False) for _ in range(3)]
    test_envs_p2 = [lambda: ConnectXGym(opponent='negamax', switch_prob=1.0, apply_symmetry=False) for _ in range(3)]
    test_envs_p3 = [lambda: ConnectXGym(opponent=minimax_lite_agent, switch_prob=0.0, apply_symmetry=False) for _ in range(2)]
    test_envs_p4 = [lambda: ConnectXGym(opponent=minimax_lite_agent, switch_prob=1.0, apply_symmetry=False) for _ in range(2)]
    test_envs = DummyVectorEnv(test_envs_p1 + test_envs_p2 + test_envs_p3 + test_envs_p4)

    net = RainbowCNN((3, 6, 7), 7, device).to(device)

    policy = DiscreteQLearningPolicy(model=net, action_space=train_envs.action_space[0],
                                    eps_training=0.0, eps_inference=0.0).to(device)

    algorithm = DQN(policy=policy, optim=AdamOptimizerFactory(lr=1e-4), gamma=0.99,
                    n_step_return_horizon=3, is_double=True,
                    target_update_freq=100, huber_loss_delta=1.0)

    buffer = PrioritizedVectorReplayBuffer(total_size=500000, buffer_num=len(train_envs), alpha=0.6, beta=0.4)

    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    print("Prefilling buffer...\n")
    train_collector.collect(n_step=20000, random=True, reset_before_collect=True)

    train_self_play.last_updated_epoch = 0

    def train_fn(epoch, env_step):
        if (epoch > 1 and epoch % UPDATE_OPPONENT_FREQ == 1 and epoch != train_self_play.last_updated_epoch):
            tqdm.write("\nUpdating opponent...\n")
            
            train_self_play.last_updated_epoch = epoch
            
            torch.save(policy.state_dict(), os.path.join(model_path, "temp_opponent.pth"))

            new_net = RainbowCNN((3, 6, 7), action_shape=7, device=device).to(device)
            new_policy = DiscreteQLearningPolicy(model=new_net, action_space=train_envs.action_space[0], eps_training=0.0, eps_inference=0.0).to(device)
            new_policy.load_state_dict(torch.load(os.path.join(model_path, "temp_opponent.pth")), strict=False)
            new_policy.eval()
            
            new_opponent_bot = RainbowAgent(new_policy)

            for i in range(6, len(train_envs.workers)):
                train_envs.workers[i].env.set_opponent(new_opponent_bot)
    
    def test_fn(epoch, env_step):
        print(f"\r{' ' * shutil.get_terminal_size().columns}\r", end='')

    def save_dual_checkpoint(algorithm):
        torch.save(algorithm.policy.model.state_dict(), os.path.join(log_path, "best_rainbow_model.pth"))
        torch.save(algorithm.policy.state_dict(), os.path.join(log_path, "best_rainbow_agent.pth"))

    logger = TensorboardLogger(SummaryWriter(log_path))
 
    result = algorithm.run_training(OffPolicyTrainerParams(
                                        training_collector=train_collector, test_collector=test_collector, test_step_num_episodes=20,
                                        max_epochs=TOTAL_EPOCHS, epoch_num_steps=20000, batch_size=64, logger=logger,
                                        collection_step_num_env_steps=2000, update_step_num_gradient_steps_per_sample=0.1,
                                        training_fn=train_fn, test_fn=test_fn,
                                        save_best_fn=save_dual_checkpoint, stop_fn=None))
    
    print("\n\nTraining finished")

if __name__ == "__main__":
    train_rainbow_self_play()