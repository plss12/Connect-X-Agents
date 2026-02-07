import numpy as np

# Adapting ConnectX for AlphaZero
class Connect4Game:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.win_length = 4

    def get_init_board(self):
        """Returns an empty 6x7 board"""
        return np.zeros((self.rows, self.cols), dtype=np.int8)

    def get_board_size(self):
        """Board shape: (rows, columns)"""
        return (self.rows, self.cols)

    def get_action_size(self):
        """Number of possible actions (7 columns)"""
        return self.cols
    
    def get_valid_moves(self, board):
        """Returns a binary vector of valid moves"""
        return (board[0, :] == 0).astype(np.int8)

    def get_next_state(self, board, player, action):
        """
        Applies an action and returns the NEW board and the next player.
        Does not modify the original board (makes a copy).
        """
        # Find the first empty row in that column
        b = np.copy(board)
        for r in range(5, -1, -1):
            if b[r, action] == 0:
                b[r, action] = player
                break
        
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
        if self.check_win(board, player):
            return 1
        # Check win for the opponent
        if self.check_win(board, -player):
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
        return (player * board).astype(np.int8)

    def get_symmetries(self, board, pi):
        """
        Data Augmentation with horizontal flip
        """
        return [(board, pi), (np.fliplr(board), pi[::-1])]

    def check_win(self, board, player):
        """
        Internal logic to check for 4 in a row.
        """
        t = (board == player)
        # Horizontal
        if (t[:, :-3] & t[:, 1:-2] & t[:, 2:-1] & t[:, 3:]).any():
            return True

        # Vertical
        if (t[:-3, :] & t[1:-2, :] & t[2:-1, :] & t[3:, :]).any():
            return True

        # Diagonals
        if np.any( t[:-3, :-3] & t[1:-2, 1:-2] & t[2:-1, 2:-1] & t[3:, 3:] ):
            return True

        if np.any( t[3:, :-3] & t[2:-1, 1:-2] & t[1:-2, 2:-1] & t[:-3, 3:] ):
            return True
            
        return False