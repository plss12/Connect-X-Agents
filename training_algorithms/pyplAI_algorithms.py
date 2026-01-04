import time
import random
import pyplAI

# Adapting ConnectX to pyplAI
class ConnectXState:
    ROWS = 6
    COLUMNS = 7

    def __init__(self, board_data, current_player=1, stop_time=None):
        self.board = list(board_data)
        self.jugadorActual = current_player
        self.stop_time = stop_time

    def get_piece(self, row, col):
        return self.board[row * self.COLUMNS + col]
    
    def get_moves(self):
        movs = []

        for col in range(self.COLUMNS):
            if self.board[col] == 0:
                movs.append(col)

        return movs

    def apply_move(self, movement):
        landing_index = -1

        for row in range(self.ROWS - 1, -1, -1):
            index = row * self.COLUMNS + movement

            if self.board[index] == 0:
                landing_index = index
                break
        
        self.board[landing_index] = self.jugadorActual
        self.jugadorActual = 2 if self.jugadorActual == 1 else 1

        return self

    def is_final_state(self):
        num_movs = len(self.get_moves())
        if num_movs == 0:
            return True
        
        if self.wins_player(1) or self.wins_player(2):
            return True
        
        return False

    def wins_player(self, player):
        for row in range(self.ROWS):
            for col in range(self.COLUMNS):

                if self.get_piece(row, col) == player:
                    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

                    for dr, dc in directions:
                        count = 1
                        
                        for i in range(1, 4):
                            
                            next_row = row + dr * i
                            next_col = col + dc * i

                            if not (0 <= next_row < self.ROWS and 0 <= next_col < self.COLUMNS):
                                break
                            
                            if self.get_piece(next_row, next_col) == player:
                                count += 1
                            else:
                                break
                        
                        if count == 4:
                            return True

    def evaluate_pattern(self, window, player):
        opponent = 2 if player == 1 else 1

        pieces_player = window.count(player)
        pieces_opponent = window.count(opponent)
        empty = window.count(0)

        def score_pattern(pieces, empty):
            if pieces == 4:
                return 1000000
            elif pieces == 3 and empty == 1:
                return 100000
            elif pieces == 2 and empty == 2:
                return 100
            return 0

        if pieces_opponent > 0 and pieces_player > 0:
            return 0

        player_score = score_pattern(pieces_player, empty)
        opponent_score = score_pattern(pieces_opponent, empty)

        return player_score - opponent_score

    def heuristic(self, player):

        if self.stop_time is not None and (self.stop_time != 0 and time.time() > self.stop_time):
            raise TimeoutError("Time limit exceeded")
            
        score = 0

        # Bonus for controlling the center column
        central_col = self.COLUMNS // 2
        for row in range(self.ROWS):
            if self.get_piece(row, central_col) == player:
                score += 10

        # Evaluating horizontal patterns
        for row in range(self.ROWS):
            for col in range(self.COLUMNS - 3):
                window = [self.get_piece(row, col + i) for i in range(4)]
                score += self.evaluate_pattern(window, player)

        # Evaluating vertical patterns
        for col in range(self.COLUMNS):
            for row in range(self.ROWS - 3):
                window = [self.get_piece(row + i, col) for i in range(4)]
                score += self.evaluate_pattern(window, player)

        # Evaluating diagonal patterns \top-left to bottom-right\
        for row in range(self.ROWS - 3):
            for col in range(self.COLUMNS - 3):
                window = [self.get_piece(row + i, col + i) for i in range(4)]
                score += self.evaluate_pattern(window, player)

        # Evaluating diagonal patterns /bottom-left to top-right/
        for row in range(self.ROWS - 3, self.ROWS):
            for col in range(self.COLUMNS - 3):
                window = [self.get_piece(row - i, col + i) for i in range(4)]
                score += self.evaluate_pattern(window, player)

        return score



# pyplAI minimax agent with CONNECT X adapted
def minimax_agent(observation, configuration):
    limit_time = configuration.timeout - 0.1
    turn_deadline = time.time() + limit_time

    current_board = observation.board
    current_player = observation.mark
        
    pyplai_state = ConnectXState(current_board, current_player, total_limit_time)
    
    valid_moves = pyplai_state.get_moves()
    best_move = valid_moves[len(valid_moves)//2] if valid_moves else None

    for depth in range(1, 5):   
        try:
            
            if time.time() > turn_deadline:
                break
            
            minimax_solver = pyplAI.Minimax(
                ConnectXState.apply_move,
                ConnectXState.get_moves, 
                ConnectXState.is_final_state, 
                ConnectXState.wins_player, 
                ConnectXState.heuristic,
                2, 
                depth)
            
            recommended_move = minimax_solver.ejecuta(pyplai_state)

            if recommended_move is not None:
                best_move = recommended_move
            else:
                break

        except TimeoutError:
            break

    return best_move



# pyplAI mcts agent with CONNECT X adapted
def mcts_agent(observation, configuration):
    limit_time = configuration.timeout - 0.1

    current_board = observation.board
    current_player = observation.mark
    
    pyplai_state = ConnectXState(current_board, current_player)
    
    mcts_solver = pyplAI.MCTS(
         ConnectXState.apply_move,
         ConnectXState.get_moves, 
         ConnectXState.is_final_state, 
         ConnectXState.wins_player, 
         2, 
         limit_time)

    recommended_move = mcts_solver.ejecuta(pyplai_state)
    
    return recommended_move



# Lite minimax agent with CONNECT X adapted for training
def minimax_lite_agent(observation, configuration):
    depth = 2

    current_board = observation.board
    current_player = observation.mark
        
    pyplai_state = ConnectXState(current_board, current_player)

    valid_moves = pyplai_state.get_moves()
    best_move = valid_moves[len(valid_moves)//2] if valid_moves else None
        
    minimax_solver = pyplAI.Minimax(
                    ConnectXState.apply_move,
                    ConnectXState.get_moves, 
                    ConnectXState.is_final_state, 
                    ConnectXState.wins_player, 
                    ConnectXState.heuristic,
                    2, 
                    depth)
            
    recommended_move = minimax_solver.ejecuta(pyplai_state)

    if recommended_move is not None:
        best_move = recommended_move

    return best_move