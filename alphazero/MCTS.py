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

        # Apply temperature
        # temp=1: Exploration (Stochastic)
        # temp=0: Competitive (Deterministic)
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        # If temp > 0, normalize the visits to obtain a distribution
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        Recursive function that goes down the tree, expands leaves and backpropagates values.
        """
        s = canonicalBoard.tobytes()

        # 1. Game ended -> Return the result
        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(canonicalBoard, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        # 2. New leaf -> Expand and backpropagate the nn value 
        if s not in self.Ps:
            pi, v = self.nnet.predict(canonicalBoard)
            
            # Mask for filtering invalid moves
            valid_moves = self.game.get_valid_moves(canonicalBoard)
            pi = pi * valid_moves 
            sum_pi = np.sum(pi)
            
            # Re-normalize and assign uniform probability to valid moves
            if sum_pi > 0:
                pi /= sum_pi 
            else:
                print("All valid moves were masked, making all valid moves equally probable!")
                pi = valid_moves / np.sum(valid_moves)

            self.Ps[s] = pi
            self.Ns[s] = 0
            self.Vs[s] = valid_moves
            
            return -v

        # 3. Known Node -> Selection using PUCT
        valid_moves = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # PUCT Formula: U(s,a) = Q(s,a) + cpuct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        cpuct = self.args.cpuct
        sqrt_Ns = math.sqrt(self.Ns[s])

        for a in np.where(valid_moves)[0]:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + cpuct * self.Ps[s][a] * sqrt_Ns / (1 + self.Nsa[(s, a)])
            else:
                u = cpuct * self.Ps[s][a] * sqrt_Ns + 1e-8 

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        
        # 4. Recursion -> Go down to the next level
        next_s, next_player = self.game.get_next_state(canonicalBoard, 1, a)
        next_s = self.game.get_canonical_form(next_s, next_player)

        v = self.search(next_s)

        # 5. Backpropagation -> Update moving average Q = (N*Q + v) / (N+1) and N
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v