import time
import random
import math
from copy import deepcopy

# Kaggle does not allow to use pyplAI, so we copy minimax code here
class Minimax:
    def __init__(self, aplicar_movimiento, obtener_movimientos, es_estado_final, gana_jugador, heuristica, numeroJugadores, profundidadBusqueda, estadisticas=False):
        self.aplicar_movimiento = staticmethod(aplicar_movimiento)
        self.obtener_movimientos = staticmethod(obtener_movimientos)
        self.es_estado_final = staticmethod(es_estado_final)
        self.gana_jugador = staticmethod(gana_jugador)
        self.heuristica = staticmethod(heuristica)
        self.jugadores = [i+1 for i in range(numeroJugadores)]
        self.profundidadBusqueda = profundidadBusqueda
        self.estadisticas = estadisticas

    def evaluate(self,estado,jugadorMax):
        if(self.es_estado_final(estado)):
            for jugador in self.jugadores:
                if(self.gana_jugador(estado,jugador)):
                    if(jugador==jugadorMax):
                        return math.inf
                    else:
                        return -math.inf
            return 0
        return self.heuristica(estado,jugadorMax)

    def minimax(self, estado, depth, movMax, jugadorMax, profundidadBusqueda, alpha, beta, estadosEvaluados):
        movs = self.obtener_movimientos(estado)
        score = self.evaluate(estado,jugadorMax)
        jugador=estado.jugadorActual
        newEstado=deepcopy(estado)

        if(self.es_estado_final(estado)):
            return ((score - depth), movMax, estadosEvaluados)

        if(depth<=profundidadBusqueda):
            if(jugadorMax==jugador):
                best = -math.inf
                for i in range(len(movs)):
                    s = self.aplicar_movimiento(newEstado,movs[i])
                    value=self.minimax(s, depth+1, movMax, jugadorMax, profundidadBusqueda, alpha, beta, estadosEvaluados+1)
                    if(value[0]>best):
                        movMax=movs[i]
                    best = max(best, value[0])
                    alpha = max(alpha, best)
                    newEstado=deepcopy(estado)
                    estadosEvaluados=value[2]
                    if (beta <= alpha):
                        break
                return (best, movMax, estadosEvaluados)
            
            else:
                best = math.inf
                for i in range(len(movs)):
                    s = self.aplicar_movimiento(newEstado,movs[i])
                    value=self.minimax(s,  depth+1, movMax, jugadorMax, profundidadBusqueda, alpha, beta, estadosEvaluados+1)
                    best = min(best,value[0])
                    beta = min(beta, best)
                    newEstado=deepcopy(estado)
                    estadosEvaluados=value[2]
                    if (beta <= alpha):
                        break
                return (best, movMax, estadosEvaluados)
        else:
            return ((score - depth), movMax, estadosEvaluados)

    def ejecuta(self, estado):
        t0 = time.time()
        alpha=-math.inf #Maximizar
        beta=math.inf #Minimizar
        jugadorMax = estado.jugadorActual
        movs = self.obtener_movimientos(estado)
        newEstado = deepcopy(estado)
        if(len(movs)==1):
            return movs[0]
        elif(len(movs)==0):
            return None
        else:
            valores = self.minimax(newEstado, 0, movs[0], jugadorMax, self.profundidadBusqueda, alpha, beta, 1)
            mov= valores[1]
            estadosEvaluados = valores[2]
        if(self.estadisticas):
            print("\nTiempo de ejecución: ", time.time()-t0)
            print("Número de estados evaluados: ", estadosEvaluados)
        return mov



# Adapting ConnectX to pyplAI
class ConnectXState:
    ROWS = 6
    COLUMNS = 7

    def __init__(self, board_data, current_player=1, time_limit_epoch=None):
        self.board = list(board_data)
        self.jugadorActual = current_player
        self.stop_time = time_limit_epoch

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

# Agent function that uses Minimax to find the best move for the current player
def minimax_agent(observation, configuration):
    limit_time = configuration.timeout - 0.1
    turn_deadline = time.time() + limit_time
    
    current_board = observation.board
    current_player = observation.mark
        
    pyplai_state = ConnectXState(current_board, current_player, turn_deadline)

    valid_moves = pyplai_state.get_moves()
    best_move = valid_moves[len(valid_moves)//2] if valid_moves else None
    
    for depth in range(1, 5):   
        try:
            
            if time.time() > turn_deadline:
                break
            
            minimax_solver = Minimax(
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