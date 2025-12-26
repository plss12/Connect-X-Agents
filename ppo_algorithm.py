import os
import math
import shutil
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from kaggle_environments import make
import matplotlib
matplotlib.use('Agg')

from tianshou.policy import PPOPolicy
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.algorithm.PPO import PPO
from tianshou.trainer import OnPolicyTrainer

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
        self.center_col = self.columns // 2

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
        self.is_mirrored = (self.apply_symmetry and self.np_random.random() < 0.5)
        self.pair = [None, self.opponent]

        if np.random.random() < self.switch_prob:
            self.pair = self.pair[::-1]
            self.trainer = self.env.train(self.pair)
        else:
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
                reward = 20
            elif reward == -1:
                reward = -20
            else:
                reward = 0
        else:
            reward = -0.1
            if int(real_action) == self.center_col:
                reward = 0.2
            elif int(real_action) in [self.center_col-1, self.center_col+1]:
                reward = 0.1
        
        return self._process_observation(observation), reward, done, False, info



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
def train_ppo_self_play():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    log_path = "logs_ppo_self_play"
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    
    model_path = "models_ppo_self_play"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path)
    
    TOTAL_EPOCHS = 100
    UPDATE_OPPONENT_FREQ = 5

    train_envs = DummyVectorEnv([lambda: ConnectXGym(opponent='random', apply_symmetry=True) for _ in range(10)])

    test_envs_p1 = [lambda: ConnectXGym(opponent='negamax', switch_prob=0.0, apply_symmetry=False) for _ in range(5)]
    test_envs_p2 = [lambda: ConnectXGym(opponent='negamax', switch_prob=1.0, apply_symmetry=False) for _ in range(5)]
    test_envs = DummyVectorEnv(test_envs_p1 + test_envs_p2)

    net_base = FeatureExtractor((3, 6, 7), device).to(device)
    
    actor = Actor(preprocess_net=net_base, action_shape=7, device=device).to(device)
    critic = Critic(preprocess_net=net_base, device=device).to(device)
    
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-4)

    def dist_fn(logits):
        return torch.distributions.Categorical(logits=logits)

    policy = PPOPolicy(
        actor=actor, critic=critic, optim=optim, dist_fn=dist_fn, action_scaling=False,
        action_space=train_envs.action_space[0], eps_clip=0.2, dual_clip=None, value_clip=0.5,
        advantage_normalization=True, recompute_advantage=True, vf_coef=0.5, ent_coef=0.01,
        max_grad_norm=0.5, gae_lambda=0.95, discount_factor=0.99, reward_normalization=True).to(device)

    buffer = VectorReplayBuffer(20000, len(train_envs))

    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    print("Prefilling buffer...\n")
    train_collector.collect(n_step=2000, random=True, reset_before_collect=True)

    train_ppo_self_play.last_updated_epoch = 0

    def train_fn(epoch, env_step):
        if (epoch > 1 and epoch % UPDATE_OPPONENT_FREQ == 1 and epoch != train_ppo_self_play.last_updated_epoch):
            tqdm.write("\nUpdating opponent...\n")
            
            train_ppo_self_play.last_updated_epoch = epoch
            
            torch.save(policy.state_dict(), os.path.join(model_path, "temp_opponent.pth"))

            new_base = FeatureExtractor((3, 6, 7), device).to(device)
            new_actor = Actor(new_base, 7, device=device).to(device)
            new_critic = Critic(new_base, device=device).to(device)
            new_optim = torch.optim.Adam(list(new_actor.parameters()) + list(new_critic.parameters()), lr=1e-4)
            
            new_policy = PPOPolicy(actor=new_actor, critic=new_critic, optim=new_optim, dist_fn=dist_fn, action_space=train_envs.action_space[0]).to(device)
            new_policy.load_state_dict(torch.load(os.path.join(model_path, "temp_opponent.pth")), strict=False)
            new_policy.eval()
            
            new_opponent_bot = TrainedAgentOpponent(new_policy)

            for i in range(3, len(train_envs.workers)):
                worker = train_envs.workers[i]
                worker.env.set_opponent(new_opponent_bot)
    
    def test_fn(epoch, env_step):
        print(f"\r{' ' * shutil.get_terminal_size().columns}\r", end='')

    def save_dual_checkpoint(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "best_ppo_agent.pth"))

    logger = TensorboardLogger(SummaryWriter(log_path))
 
    result = OnPolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=TOTAL_EPOCHS,
        step_per_epoch=20000,
        repeat_per_collect=10,
        episode_per_test=20,
        batch_size=64,
        step_per_collect=2000,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=None,
        save_best_fn=save_dual_checkpoint,
        logger=logger).run()
    
    print("\n\nTraining finished")

if __name__ == "__main__":
    train_ppo_self_play()