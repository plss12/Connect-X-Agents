import numpy as np

# Adapting ConnectX for AlphaZero
class Connect4Game:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.win_length = 4

    def get_init_board(self):
        """Returns an empty 6x7 board"""
        return np.zeros((self.rows, self.cols), dtype=int)

    def get_board_size(self):
        """Board shape: (rows, columns)"""
        return (self.rows, self.cols)

    def get_action_size(self):
        """Number of possible actions (7 columns)"""
        return self.cols
    
    def get_valid_moves(self, board):
        """Returns a binary vector of valid moves"""
        return (board[0, :] == 0).astype(int)

    def get_next_state(self, board, player, action):
        """
        Applies an action and returns the NEW board and the next player.
        Does not modify the original board (makes a copy).
        """
        # Validate action illegal moves
        if board[0, action] != 0:
            return board, player

        # Find the first empty row in that column
        b = np.copy(board)
        row = np.where(b[:, action] == 0)[0][-1]
        
        # Place piece (We use 1 for P1 and -1 for P2 in the internal logic)
        b[row, action] = player
        
        # Return new board and change turn (-player)
        return b, -player

    def get_game_ended(self, board, player):
        """
        Returns:
         1 if 'player' has won
        -1 if 'player' has lost
         1e-4 if it's a draw (small number different from 0)
         0 if the game has not ended
        """
        # Check win for the current player
        if self._check_win(board, player):
            return 1
        # Check win for the opponent
        if self._check_win(board, -player):
            return -1
        # Check draw (full board)
        if 0 not in board[0, :]:
            return 1e-4
        return 0

    def get_canonical_form(self, board, player):
        """
        Convert the board to the perspective of the current player
        so the neural network always sees the same input format
        """
        return player * board

    def get_symmetries(self, board, pi):
        """
        Data Augmentation with horizontal flip
        """
        return [(board, pi), (np.fliplr(board), pi[::-1])]

    def _check_win(self, board, player):
        """Internal logic to check for 4 in a row"""
        # Horizontal
        for c in range(self.cols - 3):
            for r in range(self.rows):
                if board[r][c] == player and board[r][c+1] == player and board[r][c+2] == player and board[r][c+3] == player:
                    return True
        # Vertical
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if board[r][c] == player and board[r+1][c] == player and board[r+2][c] == player and board[r+3][c] == player:
                    return True
        # Diagonal Positive
        for c in range(self.cols - 3):
            for r in range(self.rows - 3):
                if board[r][c] == player and board[r+1][c+1] == player and board[r+2][c+2] == player and board[r+3][c+3] == player:
                    return True
        # Diagonal Negative
        for c in range(self.cols - 3):
            for r in range(3, self.rows):
                if board[r][c] == player and board[r-1][c+1] == player and board[r-2][c+2] == player and board[r-3][c+3] == player:
                    return True
        return False