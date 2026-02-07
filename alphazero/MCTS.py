import math
import numpy as np
import torch

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

    def getActionProb(self, canonicalBoard, temp=1):
        """
        Executes 'num_mcts_sims' simulations and returns the action probabilities.
        """
        # Execute MCTS simulations
        for i in range(self.args.num_mcts_sims):
            self.search(canonicalBoard)

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
            return probs

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
        return probs

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