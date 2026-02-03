import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import glob
import re
import numpy as np

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

# Helper to calculate averages
class AverageMeter(object):
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1): self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count
    
class NNetWrapper:
    def __init__(self, game, args):
        self.nnet = Connect4NNet(game, args)
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples, writer, iteration):
        """
        examples: list of (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        for epoch in range(self.args.epochs):
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            batch_idx = 0

            # Split data into Batches
            while batch_idx < int(len(examples) / self.args.batch_size):
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # Convert the board from (6,7) to (3,6,7)
                # Channel 0: My pieces (1)
                # Channel 1: Rival pieces (-1)
                # Channel 2: Empty slots (0)
                encoded_boards = []
                for board in boards:
                    encoded_board = np.stack([
                        (board == 1).astype(np.float32),
                        (board == -1).astype(np.float32),
                        (board == 0).astype(np.float32)
                    ])
                    encoded_boards.append(encoded_board)
                
                # Convert to Tensors
                boards = torch.FloatTensor(np.array(encoded_boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if self.args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # Predict
                out_pi, out_v = self.nnet(boards)

                # Calculate Loss (CrossEntropy + MeanSquaredError)
                l_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
                l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size()[0]
                total_loss = l_pi + l_v

                # Backpropagation
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += 1
            
            # Update TensorBoard
            global_step = (iteration - 1) * self.args.epochs + epoch
            if writer:
                writer.add_scalar('Loss/Policy', pi_losses.avg, global_step)
                writer.add_scalar('Loss/Value', v_losses.avg, global_step)
                writer.add_scalar('Loss/Total', pi_losses.avg + v_losses.avg, global_step)

            print(f"Epoch {epoch+1}/{self.args.epochs} | Loss_Pi: {pi_losses.avg:.4f} | Loss_V: {v_losses.avg:.4f}")

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
        board_tensor = board_tensor.view(1, 3, self.board_x, self.board_y)
        
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board_tensor)

        # pi returns LogSoftmax, v returns Tanh
        return torch.exp(pi).detach().cpu().numpy()[0], v.item()

    def save_checkpoint(self, folder='checkpoints', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder): os.mkdir(folder)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoints', filename='best.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath): raise ValueError("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
        self.nnet.load_state_dict(checkpoint['state_dict'])

    def get_latest_checkpoint_iteration(self, folder='checkpoints'):
        if not os.path.exists(folder):
            return 1
        
        files = glob.glob(os.path.join(folder, "checkpoint_*.examples"))
        if not files:
            return 1

        iterations = []
        for f in files:
            try:
                basename = os.path.basename(f)
                num = int(re.findall(r'\d+', basename)[0])
                iterations.append(num)
            except:
                pass
                
        return max(iterations) + 1 if iterations else 1