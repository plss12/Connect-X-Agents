import os
import math
import numpy as np
import gymnasium as gym
from kaggle_environments import make
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

# Adapting ConnectX to gymnasium for Tianshou
class ConnectXGym(gym.Env):
    def __init__(self, switch_prob=0.5, opponent="negamax", apply_symmetry=True):
        self.env = make("connectx", debug=False)
        self.switch_prob = switch_prob
        self.opponent = opponent
        self.apply_symmetry = apply_symmetry

        self.pair = [None, self.opponent]
        self.trainer = self.env.train(self.pair)

        self.rows = self.env.configuration.rows
        self.columns = self.env.configuration.columns

        self.action_space = gym.spaces.Discrete(self.columns)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,self.rows, self.columns), dtype=np.float32)

        self.is_mirrored = False
    
    def set_opponent(self, opponent):
        self.opponent = opponent
        self.pair = [None, self.opponent]
        self.trainer = self.env.train(self.pair)
    
    def _process_observation(self, observation):
        board = np.array(observation['board']).reshape(self.rows, self.columns)

        if observation.mark == 2:
            new_board = np.copy(board)
            new_board[board==1] = 2
            new_board[board==2] = 1
            board = new_board
        
        if self.is_mirrored:
            board = np.fliplr(board)

        layer_me = (board == 1)
        layer_opponent = (board == 2)
        layer_empty = (board == 0)

        return np.stack([layer_me, layer_opponent, layer_empty])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.apply_symmetry and self.np_random.random() < 0.5:
            self.is_mirrored = True
        else:
            self.is_mirrored = False

        if np.random.random() < self.switch_prob:
            self.pair = self.pair[::-1]
            self.trainer = self.env.train(self.pair)

        observation = self.trainer.reset()
        return self._process_observation(observation), {}

    def step(self, action):
        real_action = action
        if self.is_mirrored:
            real_action = self.columns - 1 - action

        observation, reward, done, info = self.trainer.step(int(real_action))

        processed_obs = self._process_observation(observation)
        
        if done:
            if reward == 1:
                reward = 100
            elif reward == -1:
                reward = -100
            else:
                reward = 0
        else: 
            reward = 1
        
        return self._process_observation(observation), reward, done, False, info



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
            nn.Conv2d(in_channels=c, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Flatten()
            )

        self.flatten_dim = 128 * h * w

        # Dueling Architecture (Value and Advantage streams)
        self.value_stream = nn.Sequential(
            NoisyLinear(self.flatten_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.flatten_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, action_shape)
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        
        # Feature Extraction (CNN)
        features = self.conv(obs)
        
        # Dueling Architecture
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values, state



# Opponent agent class for using Tianshou policies in self-play
class TrainedAgentOpponent:
    def __init__(self, policy, env_rows=6, env_columns=7):
        self.policy = policy
        self.policy.eval()
        self.env_rows = env_rows
        self.env_columns = env_columns

    def __call__(self, observation, configuration):
        board = np.array(observation['board']).reshape(self.env_rows, self.env_columns)

        if observation.mark == 2:
            new_board = np.copy(board)
            new_board[board==1] = 2
            new_board[board==2] = 1
            board = new_board

        layer_me = (board == 1)
        layer_opponent = (board == 2)
        layer_empty = (board == 0)
        state = np.stack([layer_me, layer_opponent, layer_empty])

        batch_obs = Batch(obs=np.array([state]), info={})

        with torch.no_grad():
            result = self.policy(batch_obs)
        
        return int(result.act[0])



# Training function with self-play
def train_self_play():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    log_path = "logs_rainbow_self_play"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    TOTAL_EPOCHS = 100
    UPDATE_OPPONENT_FREQ = 5

    train_envs = DummyVectorEnv([lambda: ConnectXGym(opponent='random', apply_symmetry=True) for _ in range(10)])
    test_envs = DummyVectorEnv([lambda: ConnectXGym(opponent='negamax', apply_symmetry=False) for _ in range(10)])

    net = RainbowCNN((3, 6, 7), 7, device).to(device)

    policy = DiscreteQLearningPolicy(model=net, action_space=train_envs.action_space[0],
                                    eps_training=0.0, eps_inference=0.0).to(device)

    algorithm = DQN(policy=policy, optim=AdamOptimizerFactory(lr=1e-4), gamma=0.99,
                    n_step_return_horizon=3, is_double=True,
                    target_update_freq=100, huber_loss_delta=1.0)

    buffer = PrioritizedVectorReplayBuffer(total_size=500000, buffer_num=len(train_envs), alpha=0.6, beta=0.4)

    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    print("Prefilling buffer")
    train_collector.collect(n_step=20000, random=True, reset_before_collect=True)

    logger = TensorboardLogger(SummaryWriter(log_path))

    best_reward = -float('inf')
    for i in range(1, TOTAL_EPOCHS+1):
        print(f"\n=== Epoch {i}/{TOTAL_EPOCHS} ===")

        if i > 1 and i % UPDATE_OPPONENT_FREQ==1:
            print("Updating opponent")
            
            current_policy_copy = copy.deepcopy(policy)
            current_policy_copy.eval()

            new_opponent_bot = TrainedAgentOpponent(current_policy_copy)

            train_envs = DummyVecEnv([lambda: ConnectXGym(opponent=new_opponent_bot, apply_symmetry=True) for _ in range(10)])
            train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
            train_collector.reset()

        result = algorithm.run_training(OffPolicyTrainerParams(
                                                training_collector=train_collector, test_collector=None,
                                                max_epochs=1, epoch_num_steps=20000, batch_size=64, logger=logger,
                                                collection_step_num_env_steps=20, update_step_num_gradient_steps_per_sample=0.1
                                            )
        )
        
        print(f"Testing agent after training epoch {i}/{TOTAL_EPOCHS}")
        test_collector.reset()
        test_result = test_collector.collect(n_episode=20)
        
        current_reward = test_result.returns.mean()
        current_std = test_result.returns.std()

        print(f"Current reward: {current_reward:.2f} ± {current_std:.2f}")
        if current_reward > best_reward:
            best_reward = current_reward
            torch.save(policy.state_dict(), os.path.join(log_path, "best_rainbow_agent.pth"))
            print(f"Training agent saved with the new best reward ({best_reward})")
    
    print("Training finished")

if __name__ == "__main__":
    train_self_play()