import time
import random
import math
from copy import deepcopy

# Kaggle does not allow to use pyplAI, so we copy mcts code here
class MCTS:
    k = 1/math.sqrt(2) #Coeficiente de exploracion

    def __init__(self, aplicar_movimiento, obtener_movimientos, es_estado_final, gana_jugador, numeroJugadores, tiempoEjecucion, estadisticas=False):
        self.aplicar_movimiento = staticmethod(aplicar_movimiento)
        self.obtener_movimientos = staticmethod(obtener_movimientos)
        self.es_estado_final = staticmethod(es_estado_final)
        self.gana_jugador = staticmethod(gana_jugador)
        self.jugadores = [i+1 for i in range(numeroJugadores)]
        self.tiempoEjecucion = tiempoEjecucion
        self.estadisticas = estadisticas

    class nodo:
        def __init__(self,padre,mov=None):
            self.mov=mov #Movimiento que ha generado el nodo
            self.n = 0 #Numero de veces que se ha visitado el nodo
            self.q = [] #Vector de recompensas acumuladas
            self.hijos = [] #Lista de nodos hijos
            self.padre = padre #Nodo padre

    @staticmethod
    def movs_restantes(v,movs):
        res=movs.copy()
        for hijo in v.hijos:
            if(hijo.mov in movs):
                res.remove(hijo.mov)
        return res
        
    @staticmethod
    def nodos_creados(v):
        if(len(v.hijos)==0):
            return 1
        else:
            suma = 1
            for hijo in v.hijos:
                suma = suma + MCTS.nodos_creados(hijo)
            return suma

    def ejecuta(self, s0):
        movimientos=self.obtener_movimientos(s0)
        if(len(movimientos)==1):
            return movimientos[0]
        elif(len(movimientos)==0):
            return None
        else:
            v0 = self.nodo(None)
            vector = [0]*len(self.jugadores)
            v0.q = vector
            t0 = time.time()
            while( time.time() - t0 < self.tiempoEjecucion):
                tree = self.tree_policy(v0,s0,movimientos)
                v1=tree[0]
                s1=tree[1]
                movs=tree[2]
                delta = self.default_policy(s1,movs,self.jugadores)
                self.backup(v1,delta,self.jugadores)
            jugador = s0.jugadorActual-1
            mejorNodo = self.best_child(v0,0,jugador)
            if(self.estadisticas):
                numeroNodosCreados = self.nodos_creados(v0)
                print("\nTiempo de ejecución: ", self.tiempoEjecucion)
                print("Número de nodos creados: ",numeroNodosCreados)
                print("Número de nodos visitados: ",v0.n)
            mov=mejorNodo.mov
            return mov
    
    def tree_policy(self, v,s,movimientos):
        while(self.es_estado_final(s)==False):
            movs_sin_visitar = self.movs_restantes(v,movimientos)
            if(0<len(movs_sin_visitar)):
                return self.expand(v,s,movs_sin_visitar)
            else:
                jugador=s.jugadorActual-1
                mejorHijo = self.best_child(v, MCTS.k ,jugador)
                copiaEstado=deepcopy(s)
                s = self.aplicar_movimiento(copiaEstado,mejorHijo.mov)
                movimientos = self.obtener_movimientos(s)
                v = mejorHijo
        return [v,s,movimientos]

    def expand(self, v,s,movRestantes):
        copiaEstado=deepcopy(s)
        mov = random.choice(movRestantes)
        s = self.aplicar_movimiento(copiaEstado, mov)
        movs = self.obtener_movimientos(s)
        hijo = self.nodo(v,mov)
        vector = [0]*len(v.q)
        hijo.q = vector
        v.hijos.append(hijo)
        return [hijo,s,movs]

    @staticmethod
    def best_child(v,c,jugador):
        indiceMejorHijo=0
        contador=0
        max=-math.inf
        for hijo in v.hijos:
            heuristica = hijo.q[jugador]/hijo.n + (c * math.sqrt((2*math.log(v.n))/hijo.n))
            if(heuristica>max):
                max = heuristica
                indiceMejorHijo=contador
            contador+=1
        return v.hijos[indiceMejorHijo]

    def default_policy(self, s,movs,jugadores):
        #El jugador que queremos comprobar es el anterior al del estado actual, ya que se ha cambiado en tree_policy
        while(self.es_estado_final(s)==False):
            a = random.choice(movs)
            s = self.aplicar_movimiento(s,a)
            movs = self.obtener_movimientos(s)

        #Crea una lista de 0s del tamaño de la lista de jugadores
        res = [0] * len(jugadores)
        #Cambia los valores de la lista de 0s a la recompensa que obtiene cada jugador
        for jugador in jugadores:
            if(self.gana_jugador(s,jugador)):
                res[jugadores.index(jugador)]=1

        #Si no hay ningún ganador, todos los jugadores empatan
        if (1 not in res):
            res = [0.5] * len(jugadores)
        return res

    @staticmethod
    def backup(v,delta,jugadores):
        while(v != None):
            v.n = v.n+1
            #Incrementa la recompensa del nodo para cada jugador
            for jugador in jugadores:
                v.q[jugadores.index(jugador)]+=delta[jugadores.index(jugador)]
            v = v.padre



# Adapting ConnectX to pyplAI style
class ConnectXState:
    ROWS = 6
    COLUMNS = 7

    def __init__(self, board_data, current_player=1):
        self.board = list(board_data)
        self.jugadorActual = current_player

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


# Agent function that uses MCTS to find the best move for the current player
def mcts_agent(observation, configuration):
    limit_time = configuration.timeout - 0.1
    current_board = observation.board
    current_player = observation.mark
    
    pyplai_state = ConnectXState(current_board, current_player)
    
    mcts_solver = MCTS(
         ConnectXState.apply_move,
         ConnectXState.get_moves, 
         ConnectXState.is_final_state, 
         ConnectXState.wins_player, 
         2, 
         limit_time,
         estadisticas=True)

    recommended_move = mcts_solver.ejecuta(pyplai_state)
    
    return recommended_move