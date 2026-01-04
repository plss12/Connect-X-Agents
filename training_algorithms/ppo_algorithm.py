import os
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.modelfree.reinforce import DiscreteActorPolicy
from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic
from tianshou.algorithm.optim import AdamOptimizerFactory, LRSchedulerFactoryLinear
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils import TensorboardLogger

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from connectXgym import ConnectXGym, apply_mask_to_logits, check_winning_move
from pyplAI_algorithms import minimax_lite_agent



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
        logits = torch.clamp(logits, min=-10, max=10)

        # Apply action mask for invalid actions
        if info is not None and "action_mask" in info:
            logits = apply_mask_to_logits(logits, info["action_mask"], self.device)

        return logits, state



# Critic: Estimate value of state
class Critic(nn.Module):
    def __init__(self, preprocess_net, device='cpu'):
        super().__init__()
        self.preprocess_net = preprocess_net
        self.last_layer = layer_init(nn.Linear(preprocess_net.output_dim, 1), std=1.0)
        self.device = device
        
    def forward(self, obs, state=None, info={}):
        features, _ = self.preprocess_net(obs, state)
        value = self.last_layer(features)
        return value



# PPO agent with rules for instant win or loss
class PPOAgent:
    def __init__(self, actor, device='cuda'):
        self.actor = actor
        self.actor.eval()
        self.device = device
    
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
            logits, _ = self.actor(state_tensor, info={"action_mask": mask_tensor})
            logits = logits.squeeze()
            best_move = int(torch.argmax(logits).item())
        
        if best_move not in valid_moves:
            best_move = int(np.random.choice(valid_moves))
        
        return best_move



# Training function with self-play
def train_ppo_self_play():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    log_path = "files_ppo/logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    model_path = "files_ppo/models"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    TOTAL_EPOCHS = 100
    STEP_PER_EPOCH = 20000
    STEP_PER_COLLECT = 2000
    UPDATE_PER_COLLECT = 10
    BATCH_SIZE = 500
    UPDATE_OPPONENT_FREQ = 2
    TEST_EPISODES = 20
    LR = 2.5e-4

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

    test_envs_p1 = [lambda: ConnectXGym(opponent='negamax', switch_prob=0.0, apply_symmetry=False) for _ in range(3)]
    test_envs_p2 = [lambda: ConnectXGym(opponent='negamax', switch_prob=1.0, apply_symmetry=False) for _ in range(3)]
    test_envs_p3 = [lambda: ConnectXGym(opponent=minimax_lite_agent, switch_prob=0.0, apply_symmetry=False) for _ in range(2)]
    test_envs_p4 = [lambda: ConnectXGym(opponent=minimax_lite_agent, switch_prob=1.0, apply_symmetry=False) for _ in range(2)]
    test_envs = DummyVectorEnv(test_envs_p1 + test_envs_p2 + test_envs_p3 + test_envs_p4)

    net_actor = FeatureExtractor((3, 6, 7), device).to(device)
    net_critic = FeatureExtractor((3, 6, 7), device).to(device)
    actor = Actor(preprocess_net=net_actor, action_shape=7, device=device).to(device)
    critic = Critic(preprocess_net=net_critic, device=device).to(device)

    optim_factory = AdamOptimizerFactory(lr=LR, eps=1e-5)
    optim_factory.with_lr_scheduler_factory(
        LRSchedulerFactoryLinear(
            max_epochs=TOTAL_EPOCHS,
            epoch_num_steps=STEP_PER_EPOCH,
            collection_step_num_env_steps=STEP_PER_COLLECT))

    policy = DiscreteActorPolicy(actor=actor, action_space=train_envs.action_space[0], 
                                deterministic_eval=True).to(device)
    
    algorithm = PPO(policy=policy, critic=critic, optim=optim_factory, eps_clip=0.2, 
                    value_clip=0.5, vf_coef=0.5, ent_coef=0.02, gae_lambda=0.95, max_grad_norm=0.5, 
                    gamma=0.99, advantage_normalization=True, return_scaling=True).to(device)

    if os.path.exists(os.path.join(model_path, "best_ppo_agent.pth")):
        algorithm.load_state_dict(torch.load(os.path.join(model_path, "best_ppo_agent.pth"), weights_only=False))
        print("Loaded best agent from checkpoint")
    else:
        print("No checkpoint found, starting training from scratch")

    buffer = VectorReplayBuffer(total_size=STEP_PER_COLLECT * len(train_envs), buffer_num=len(train_envs))

    train_collector = Collector(algorithm, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(algorithm, test_envs, exploration_noise=False)

    train_ppo_self_play.last_updated_epoch = 0
    train_ppo_self_play.opponent_version = 0

    def train_fn(epoch, env_step):
        if (epoch > 1 and epoch % UPDATE_OPPONENT_FREQ == 1 and epoch != train_ppo_self_play.last_updated_epoch):
            tqdm.write("\nUpdating opponent...\n")
            train_ppo_self_play.last_updated_epoch = epoch
            train_ppo_self_play.opponent_version += 1
            
            torch.save(algorithm.policy.actor.state_dict(), os.path.join(model_path, "temp_opponent.pth"))

            base_net = FeatureExtractor((3, 6, 7), device).to(device)
            new_actor = Actor(base_net, 7, device=device).to(device)
            new_actor.load_state_dict(torch.load(os.path.join(model_path, "temp_opponent.pth")))
            new_opponent_bot = PPOAgent(new_actor, device=device)

            self_play_indexes = [6, 7, 8, 9]
            index_to_update = self_play_indexes[train_ppo_self_play.opponent_version % len(self_play_indexes)]
            train_envs.workers[index_to_update].env.set_opponent(new_opponent_bot)
        
        if env_step % STEP_PER_COLLECT == 0:
            current_lr = algorithm.optim._optim.param_groups[0]['lr']
            logger.write("training/env_step", env_step, {"training/lr": current_lr, "training/opponent_version": train_ppo_self_play.opponent_version})
    
    def test_fn(epoch, env_step):
        print(f"\r{' ' * shutil.get_terminal_size().columns}\r", end='')

    def save_dual_checkpoint(algorithm):
        torch.save(algorithm.policy.actor.state_dict(), os.path.join(model_path, "best_ppo_actor.pth"))
        torch.save(algorithm.state_dict(), os.path.join(model_path, "best_ppo_agent.pth"))

    logger = TensorboardLogger(SummaryWriter(log_path))
 
    result = algorithm.run_training(OnPolicyTrainerParams(
                    training_collector=train_collector, test_collector=test_collector,
                    max_epochs=TOTAL_EPOCHS, epoch_num_steps=STEP_PER_EPOCH, collection_step_num_env_steps=STEP_PER_COLLECT, 
                    update_step_num_repetitions=UPDATE_PER_COLLECT, test_step_num_episodes=TEST_EPISODES, logger=logger, 
                    batch_size=BATCH_SIZE, training_fn=train_fn, test_fn=test_fn, save_best_fn=save_dual_checkpoint, stop_fn=None))
    
    torch.save(algorithm.policy.actor.state_dict(), os.path.join(model_path, "final_ppo_actor.pth"))
    torch.save(algorithm.state_dict(), os.path.join(model_path, "final_ppo_agent.pth"))
    print("\n\nTraining finished")

if __name__ == "__main__":
    train_ppo_self_play()