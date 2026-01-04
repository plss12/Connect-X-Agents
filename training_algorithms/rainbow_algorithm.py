import os
import math
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

from tianshou.algorithm import RainbowDQN
from tianshou.algorithm.modelfree.c51 import C51Policy
from tianshou.algorithm.optim import AdamOptimizerFactory, LRSchedulerFactoryLinear
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from connectXgym import ConnectXGym, check_winning_move, apply_mask_to_logits
from pyplAI_algorithms import minimax_lite_agent



# Noisy Linear Layer
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("epsilon_in", torch.empty(in_features))
        self.register_buffer("epsilon_out", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)     
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        self.epsilon_in.copy_(self._scale_noise(self.in_features))
        self.epsilon_out.copy_(self._scale_noise(self.out_features))

    def forward(self, x):
        if self.training:
            self.reset_noise()

        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.epsilon_out.ger(self.epsilon_in)
            bias = self.bias_mu + self.bias_sigma * self.epsilon_out
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)



# Rainbow CNN (Dueling Architecture + Noisy Linear Layers)
class RainbowCNN(nn.Module):
    def __init__(self, state_shape, action_shape, num_atoms=51, device='cpu'):
        super().__init__()
        self.device = device
        self.action_shape = action_shape
        self.num_atoms = num_atoms
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
            NoisyLinear(512, num_atoms))
        
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.flatten_dim, 512), nn.ReLU(),
            NoisyLinear(512, action_shape * num_atoms))

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        
        # Feature Extraction (CNN)
        features = self.conv(obs)
        batch_size = features.size(0)
        
        # Dueling Architecture
        value = self.value_stream(features).view(batch_size, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(batch_size, self.action_shape, self.num_atoms)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Apply action mask for invalid actions
        if info is not None and "action_mask" in info:
            logits = apply_mask_to_logits(logits, info["action_mask"], self.device)

        probs = logits.softmax(dim=2)

        return probs, state



# Rainbow agent with rules for instant win or loss
class RainbowAgent:
    def __init__(self, model, num_atoms=51, v_min=-10, v_max=10, device='cuda'):
        self.model = model
        self.model.eval()
        self.device = device

        self.num_atoms = num_atoms
        self.support = torch.linspace(v_min, v_max, num_atoms).to(device)

    def __call__(self, observation, configuration):
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
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        mask_bool = [False] * configuration.columns
        for col in valid_moves:
            mask_bool[col] = True
        mask_tensor = torch.tensor(mask_bool, dtype=torch.bool).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = self.model(state_tensor, info={"action_mask": mask_tensor})
            probs = F.softmax(logits, dim=2)
            q_values = (probs * self.support).sum(dim=2)
            best_move = int(torch.argmax(q_values).item())
        
        if best_move not in valid_moves:
            best_move = int(np.random.choice(valid_moves))
        
        return best_move



# Training function with self-play
def train_rainbow_self_play():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    log_path = "files_rainbow/logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    model_path = "files_rainbow/models"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    TOTAL_EPOCHS = 100
    STEP_PER_EPOCH = 20000
    TOTAL_STEPS = TOTAL_EPOCHS * STEP_PER_EPOCH
    STEP_PER_COLLECT = 2000
    UPDATE_PER_COLLECT = 0.25
    BATCH_SIZE = 512
    BUFFER_SIZE = 500000
    UPDATE_OPPONENT_FREQ = 2
    TEST_EPISODES = 20
    LR = 2.5e-4

    NUM_ATOMS = 101
    V_MIN = -30
    V_MAX = 30
    NOISY_STD = 0.5

    ALPHA = 0.6
    BETA_START = 0.4
    BETA_END = 1.0

    train_envs = DummyVectorEnv([lambda: ConnectXGym(opponent='random', apply_symmetry=True),
                                lambda: ConnectXGym(opponent='random', apply_symmetry=True),
                                lambda: ConnectXGym(opponent='random', apply_symmetry=True),

                                lambda: ConnectXGym(opponent=minimax_lite_agent, apply_symmetry=True),
                                lambda: ConnectXGym(opponent=minimax_lite_agent, apply_symmetry=True),
                                lambda: ConnectXGym(opponent=minimax_lite_agent, apply_symmetry=True),

                                lambda: ConnectXGym(opponent='random', apply_symmetry=True),
                                lambda: ConnectXGym(opponent='random', apply_symmetry=True),
                                lambda: ConnectXGym(opponent='random', apply_symmetry=True),
                                lambda: ConnectXGym(opponent='random', apply_symmetry=True)])

    test_envs = DummyVectorEnv([lambda: ConnectXGym(opponent='negamax', switch_prob=0.0, apply_symmetry=False),
                                lambda: ConnectXGym(opponent='negamax', switch_prob=1.0, apply_symmetry=False),
                                lambda: ConnectXGym(opponent=minimax_lite_agent, switch_prob=0.0, apply_symmetry=False),
                                lambda: ConnectXGym(opponent=minimax_lite_agent, switch_prob=1.0, apply_symmetry=False)])

    net = RainbowCNN((3, 6, 7), 7, num_atoms=NUM_ATOMS, device=device).to(device)

    optim_factory = AdamOptimizerFactory(lr=LR, eps=1e-5)
    optim_factory.with_lr_scheduler_factory(
        LRSchedulerFactoryLinear(
            max_epochs=TOTAL_EPOCHS,
            epoch_num_steps=STEP_PER_EPOCH,
            collection_step_num_env_steps=STEP_PER_COLLECT))

    policy = C51Policy(model=net, action_space=train_envs.action_space[0], num_atoms=NUM_ATOMS,
                       v_min=V_MIN, v_max=V_MAX, eps_inference=0.0, eps_training=0.0).to(device)

    algorithm = RainbowDQN(policy=policy, optim=optim_factory, gamma=0.99,
                    n_step_return_horizon=3, target_update_freq=400)
    
    if os.path.exists(os.path.join(model_path, "best_rainbow_agent.pth")):
        algorithm.load_state_dict(torch.load(os.path.join(model_path, "best_rainbow_agent.pth")))
        print("Loaded best agent from checkpoint")
    else:
        print("No checkpoint found, starting training from scratch")
    
    buffer = PrioritizedVectorReplayBuffer(total_size=BUFFER_SIZE, buffer_num=len(train_envs), 
                                            alpha=ALPHA, beta=BETA_START, weight_norm=True)

    train_collector = Collector(algorithm, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(algorithm, test_envs, exploration_noise=False)

    print("Prefilling buffer...\n")
    train_collector.collect(n_step=STEP_PER_EPOCH, reset_before_collect=True)

    train_rainbow_self_play.last_updated_epoch = 0
    train_rainbow_self_play.opponent_version = 0

    def train_fn(epoch, env_step):

        if env_step <= TOTAL_STEPS:
            beta = BETA_START + (BETA_END - BETA_START) * (env_step / TOTAL_STEPS)
        else:
            beta = BETA_END
        buffer.set_beta(beta)

        if (epoch > 1 and epoch % UPDATE_OPPONENT_FREQ == 1 and epoch != train_rainbow_self_play.last_updated_epoch):
            tqdm.write("\nUpdating opponent...\n")
            train_rainbow_self_play.last_updated_epoch = epoch
            train_rainbow_self_play.opponent_version += 1
            
            torch.save(algorithm.policy.model.state_dict(), os.path.join(model_path, "temp_opponent.pth"))

            new_net = RainbowCNN((3, 6, 7), action_shape=7, num_atoms=NUM_ATOMS, device=device).to(device)
            new_net.load_state_dict(torch.load(os.path.join(model_path, "temp_opponent.pth")))
            new_opponent_bot = RainbowAgent(new_net, num_atoms=NUM_ATOMS, v_min=V_MIN, v_max=V_MAX, device=device)

            self_play_indexes = [6, 7, 8, 9]
            index_to_update = self_play_indexes[train_rainbow_self_play.opponent_version % len(self_play_indexes)]
            train_envs.workers[index_to_update].env.set_opponent(new_opponent_bot)

        if env_step % STEP_PER_COLLECT == 0:
            current_lr = algorithm.optim._optim.param_groups[0]['lr']
            logger.write("training/env_step", env_step, {"training/beta": beta, "training/opponent_version": train_rainbow_self_play.opponent_version, "training/lr": current_lr})
    
    def test_fn(epoch, env_step):
        print(f"\r{' ' * shutil.get_terminal_size().columns}\r", end='')

    def save_dual_checkpoint(algorithm):
        torch.save(algorithm.policy.model.state_dict(), os.path.join(model_path, "best_rainbow_model.pth"))
        torch.save(algorithm.state_dict(), os.path.join(model_path, "best_rainbow_agent.pth"))

    logger = TensorboardLogger(SummaryWriter(log_path))
 
    result = algorithm.run_training(OffPolicyTrainerParams(
                                        training_collector=train_collector, test_collector=test_collector, test_step_num_episodes=TEST_EPISODES,
                                        max_epochs=TOTAL_EPOCHS, epoch_num_steps=STEP_PER_EPOCH, batch_size=BATCH_SIZE, logger=logger,
                                        collection_step_num_env_steps=STEP_PER_COLLECT, update_step_num_gradient_steps_per_sample=UPDATE_PER_COLLECT,
                                        training_fn=train_fn, test_fn=test_fn, save_best_fn=save_dual_checkpoint, stop_fn=None))
    
    torch.save(algorithm.policy.model.state_dict(), os.path.join(model_path, "final_rainbow_model.pth"))
    torch.save(algorithm.state_dict(), os.path.join(model_path, "final_rainbow_agent.pth"))
    print("\n\nTraining finished")

if __name__ == "__main__":
    train_rainbow_self_play()