import os
import sys
from torch.utils.tensorboard import SummaryWriter
import time
import torch
torch.set_float32_matmul_precision('medium')
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from env.connectXalphazero import Connect4Game
from alphazero.NNet import NNetWrapper as nn
from alphazero.Trainer import Trainer

# Helper class to use dot notation (args.variable)
class dotdict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

# --- HYPERPARAMETER CONFIGURATION ---
args = dotdict({
    # 1. Training Configuration (Cycle)
    'num_iters': 200,            # Number of total generations (Iterations)
    'num_eps': 100,              # Episodes (games) of Self-Play per iteration
    'tempThreshold': 10,         # Turns with exploration (temperature=1) before playing seriously
    'updateThreshold': 0.55,     # % of wins necessary for the new network to replace the old one
    'maxlenOfQueue': 100000,     # Maximum number of training examples stored in memory
    'numItersForTrainExamplesHistory': 25, # Number of past iterations remembered for training
    'arenaCompare': 50,          # Evaluation games (Arena) New vs Old

    # 2. MCTS Configuration (Brain)
    'num_mcts_sims': 500,        # Simulations per move. MORE = Better game, but SLOWER.
    'cpuct': 1.0,                # Exploration constant
    'gamma': 0.99,               # Discounting factor for distant future rewards

    # 3. Neural Network Configuration (Learning)
    'max_lr': 0.005,             # Initial Learning Rate
    'min_lr': 0.0001,            # Final Learning Rate
    'weight_decay': 1e-4,        # L2 Regularization (Weight Decay)
    # 'dropout': 0.3,              # Dropout to prevent overfitting
    'epochs': 10,                # Number of training epochs for the network in each iteration
    'batch_size': 512,           # Batch size
    'cuda': True,                # Use GPU (Change to False if you don't have NVIDIA)
    'num_channels': 32,          # Network size (convolutional filters)

    # 4. Checkpoint Configuration
    'load_model': False,                    # Load previous model? (Set True to resume)
    'checkpoint_folder': './alphazero/files/checkpoints/', # Folder to save checkpoints
    'checkpoint_file': 'best.pth.tar'       # File to load if load_model=True
})

def main():
    print("Starting AlphaZero for Connect4...")
    
    # 1. Create Game and Neural Network
    g = Connect4Game()
    nnet = nn(g, args)

    # 2. Load Checkpoint (If we want to resume)
    if args.load_model:
        print(f"Loading checkpoint from {args.checkpoint_folder}...")
        try:
            nnet.load_checkpoint(args.checkpoint_folder, args.checkpoint_file)
            start_iter = nnet.get_latest_checkpoint_iteration(args.checkpoint_folder)
            print(f"Resuming training from iteration {start_iter}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            args.load_model = False
    else:
        print("Starting training from scratch...")

    # 3. Create Summary Writer for TensorBoard
    run_name = f"run_{int(time.time())}"
    print(f"Creating TensorBoard...")
    writer = SummaryWriter(log_dir=f'./alphazero/files/logs/')
    
    # 4. Create Trainer
    c = Trainer(g, nnet, args, writer)

    # 5. Load Training Examples (Memory)
    if args.load_model:
        print("Charging training examples history...")
        c.loadTrainExamples()

    # 6. Training
    print(f"Starting the training loop...")
    if args.load_model:
        c.learn(start_iter)
    else:
        c.learn()

    # 7. Close TensorBoard
    writer.close()

if __name__ == "__main__":
    main()