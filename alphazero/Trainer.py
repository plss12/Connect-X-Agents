from collections import deque
from .MCTS import MCTS
import numpy as np
from tqdm import tqdm
import time
import os
import re
from pickle import Pickler, Unpickler

class Trainer:
    def __init__(self, game, nnet, args, writer=None):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(game, args)  # Previous network
        self.args = args
        self.mcts = MCTS(game, nnet, args)
        self.trainExamplesHistory = []  # Experience buffer
        self.writer = writer

    def executeEpisode(self):
        """
        Executes ONE complete game of Self-Play.
        """
        trainExamples = []
        board = self.game.get_init_board()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            # 1. Get canonical board (perspective of the current player)
            canonicalBoard = self.game.get_canonical_form(board, self.curPlayer)
            
            # 2. Calculate temperature
            # At the beginning we explore (temp=1), then we play seriously (temp=0)
            temp = int(episodeStep < self.args.tempThreshold)

            # 3. MCTS finds the best move
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            
            # 4. Save data and symmetries (Data Augmentation)
            sym = self.game.get_symmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            # 5. Choose action
            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.get_next_state(board, self.curPlayer, action)

            # 6. Game ended?
            r = self.game.get_game_ended(board, 1)
            if r != 0:
                # Propagation of the r (result from player 1's perspective) to all examples in the history
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def arena(self, pmcts, nmcts):
        """
        Compares the old network (pmcts) against the new one (nmcts).
        They play 'arenaCompare' games.
        """
        pwins = 0
        nwins = 0
        draws = 0
        
        # Function for them to play
        def play_match(player1_mcts, player2_mcts):
            # temp=0 to play competitively (deterministic)
            board = self.game.get_init_board()
            curPlayer = 1
            step = 0
            while True:
                step += 1
                canonicalBoard = self.game.get_canonical_form(board, curPlayer)
                
                if curPlayer == 1:
                    pi = player1_mcts.getActionProb(canonicalBoard, temp=0)
                else:
                    pi = player2_mcts.getActionProb(canonicalBoard, temp=0)
                
                # Choose the best move (argmax)
                action = np.argmax(pi)
                
                # Validate move
                valids = self.game.get_valid_moves(canonicalBoard)
                if valids[action] == 0:
                    print("ERROR: Invalid move selected in Arena")
                    action = np.random.choice(np.where(valids)[0])

                board, curPlayer = self.game.get_next_state(board, curPlayer, action)
                r = self.game.get_game_ended(board, 1)
                
                if r != 0:
                    return r * curPlayer

        for _ in tqdm(range(self.args.arenaCompare // 2), desc="Arena"):
            # Half of the games start with the Old network, half with the New one
            # Game 1: Old (1) vs New (-1)
            res = play_match(pmcts, nmcts)
            if res == 1: pwins += 1
            elif res == -1: nwins += 1
            else: draws += 1
            
            # Game 2: New (1) vs Old (-1)
            res = play_match(nmcts, pmcts)
            if res == 1: nwins += 1      # If P1 (New) wins
            elif res == -1: pwins += 1   # If P2 (Old) wins
            else: draws += 1
            
        return pwins, nwins, draws

    def saveTrainExamples(self, iteration):
        # Save example
        folder = self.args.checkpoint_folder
        extension = ".examples"
        if not os.path.exists(folder): os.makedirs(folder)
        filename = os.path.join(folder, f"checkpoint_{iteration}{extension}")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        
        # Delete old examples
        files_to_keep = {f"checkpoint_{iteration}{extension}", f"checkpoint_{iteration-1}{extension}"}
        for file in os.listdir(folder):
            if file.endswith(extension) and file not in files_to_keep:
                os.remove(os.path.join(folder, file))
    
    def loadTrainExamples(self):
        # Looking for the examples file
        model_name = self.args.checkpoint_file
        exact_examples_file = os.path.join(self.args.checkpoint_folder, model_name + ".examples")
        
        file_to_load = ""

        if os.path.isfile(exact_examples_file):
            print(f"Found exact examples: {exact_examples_file}")
            file_to_load = exact_examples_file
        else:
            print(f"No exact examples found for {model_name}. Looking for the most recent...")
            
            folder = self.args.checkpoint_folder
            all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".examples")]
            checkpoint_files = [f for f in all_files if "checkpoint_" in os.path.basename(f)]
            
            if len(checkpoint_files) > 0:
                newest_file = max(checkpoint_files, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
                print(f"Using alternative file (newest): {newest_file}")
                file_to_load = newest_file
            else:
                sys.exit("The folder does not contain any checkpoint_*.examples file")

        # Load the chosen file
        with open(file_to_load, "rb") as f:
            self.trainExamplesHistory = Unpickler(f).load()
    
    def saveTrainCheckpoint(self, iteration):
        # Save checkpoint
        folder = self.args.checkpoint_folder
        extension = ".pth.tar"
        self.nnet.save_checkpoint(folder=folder, filename=f'checkpoint_{iteration}{extension}')

        # Delete old checkpoints
        checkpoint_files = [f for f in os.listdir(folder) if f.startswith("checkpoint_") and f.endswith(extension)]
        checkpoint_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        for file in checkpoint_files[:-2]:
            os.remove(os.path.join(folder, file))

    def learn(self, start_iter=1):
        """
        Main Training Loop
        """
        updates = 0
        for i in range(start_iter, self.args.num_iters + 1):
            print(f'------ Iteration {i}/{self.args.num_iters} ------')
            
            # --- PHASE 1: SELF-PLAY ---
            deque_examples = deque([], maxlen=self.args.maxlenOfQueue)
            
            # Play X episodes against itself
            for _ in tqdm(range(self.args.num_eps), desc="Self Play"):
                self.mcts = MCTS(self.game, self.nnet, self.args)  # Reset MCTS
                deque_examples += self.executeEpisode()

            # Add to global history
            self.trainExamplesHistory.append(deque_examples)
            
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print(f"Deleting old data (Keeping {self.args.numItersForTrainExamplesHistory} iters)...")
                self.trainExamplesHistory.pop(0)
            
            # Save backup of examples
            self.saveTrainExamples(i - 1)

            # Flatten list of examples to train
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            
            # Shuffle data
            np.random.shuffle(trainExamples)

            # Save current network in 'temp' to compare later
            self.nnet.save_checkpoint(folder=self.args.checkpoint_folder, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint_folder, filename='temp.pth.tar')
            
            # MCTS for the old network
            pmcts = MCTS(self.game, self.pnet, self.args)

            # --- PHASE 2: TRAINING ---
            print("Training Neural Network...")
            self.nnet.train(trainExamples, self.writer, i)
            
            # MCTS for the new network (trained)
            nmcts = MCTS(self.game, self.nnet, self.args)

            # --- PHASE 3: ARENA (EVALUATION) ---
            print("Facing New Net vs Old Net...")
            pwins, nwins, draws = self.arena(pmcts, nmcts)

            winratenew = nwins / (nwins + pwins) if (nwins + pwins) > 0 else 0
            winrateold = pwins / (nwins + pwins) if (nwins + pwins) > 0 else 0

            # Update TensorBoard
            if self.writer:
                self.writer.add_scalar('Arena/OldNetWins', pwins, i)
                self.writer.add_scalar('Arena/NewNetWins', nwins, i)
                self.writer.add_scalar('Arena/Draws', draws, i)
                self.writer.add_scalar('Arena/WinRateNew', winratenew, i)
                self.writer.add_scalar('Arena/WinRateOld', winrateold, i)

            print(f"RESULTS: NEW={nwins}, OLD={pwins}, DRAW={draws}")

            # If the new one wins enough, we accept it
            if (nwins + pwins) > 0 and winratenew < self.args.updateThreshold:
                print("REJECTED: The new network is not good enough.")
                self.nnet.load_checkpoint(folder=self.args.checkpoint_folder, filename='temp.pth.tar')
                self.writer.add_scalar('Arena/Updates', updates, i)
            else:
                print("ACCEPTED: Saving new best network.")
                self.nnet.save_checkpoint(folder=self.args.checkpoint_folder, filename='best.pth.tar')
                self.saveTrainCheckpoint(i)
                updates += 1
                self.writer.add_scalar('Arena/Updates', updates, i)