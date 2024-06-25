# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 139:
# 102373 Afonso Calinas

import sys
from sys import stdin
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)
from copy import deepcopy
from collections import deque

DIRECTIONS = {
    'FC': [(-1, 0)],  # up
    'FD': [(0, 1)],  # right
    'FB': [(1, 0)],  # down
    'FE': [(0, -1)],  # left

    'BC': [(0, -1), (-1, 0), (0, 1)],  # left, up, right
    'BD': [(-1, 0), (0, 1), (1, 0)],  # up, right, down
    'BB': [(0, -1), (1, 0), (0, 1)],  # left, down, right
    'BE': [(1, 0), (0, -1), (-1, 0)],  # down, left, up

    'VC': [(0, -1), (-1, 0)],  # left, up
    'VD': [(-1, 0), (0, 1)],  # up, right
    'VB': [(0, 1), (1, 0)],  # right, down
    'VE': [(1, 0), (0, -1)],  # down, left

    'LH': [(0, -1), (0, 1)],  # left, right
    'LV': [(-1, 0), (1, 0)],  # up, down
}

# Opposite directions for validation
OPPOSITE_DIRECTIONS = {
    (-1, 0): (1, 0),
    (1, 0): (-1, 0),
    (0, -1): (0, 1),
    (0, 1): (0, -1)
}

def rotate_clockwise(piece):
    """Rotate the piece clockwise."""
    if piece[0] == 'F':
        if piece[1] == 'C':
            return 'FD'
        elif piece[1] == 'D':
            return 'FB'
        elif piece[1] == 'B':
            return 'FE'
        elif piece[1] == 'E':
            return 'FC'
    elif piece[0] == 'B':
        if piece[1] == 'C':
            return 'BD'
        elif piece[1] == 'D':
            return 'BB'
        elif piece[1] == 'B':
            return 'BE'
        elif piece[1] == 'E':
            return 'BC'
    elif piece[0] == 'V':
        if piece[1] == 'C':
            return 'VD'
        elif piece[1] == 'D':
            return 'VB'
        elif piece[1] == 'B':
            return 'VE'
        elif piece[1] == 'E':
            return 'VC'

def rotate_counterclockwise(piece):
    """Rotate the piece counter-clockwise."""
    if piece[0] == 'F':
        if piece[1] == 'C':
            return 'FE'
        elif piece[1] == 'D':
            return 'FC'
        elif piece[1] == 'B':
            return 'FD'
        elif piece[1] == 'E':
            return 'FB'
    elif piece[0] == 'B':
        if piece[1] == 'C':
            return 'BE'
        elif piece[1] == 'D':
            return 'BC'
        elif piece[1] == 'B':
            return 'BD'
        elif piece[1] == 'E':
            return 'BB'
    elif piece[0] == 'V':
        if piece[1] == 'C':
            return 'VE'
        elif piece[1] == 'D':
            return 'VC'
        elif piece[1] == 'B':
            return 'VD'
        elif piece[1] == 'E':
            return 'VB'

def invert(piece):
    """Invert the piece."""
    if piece[0] == 'F':
        if piece[1] == 'C':
            return 'FB'
        elif piece[1] == 'D':
            return 'FE'
        elif piece[1] == 'B':
            return 'FC'
        elif piece[1] == 'E':
            return 'FD'
    elif piece[0] == 'B':
        if piece[1] == 'C':
            return 'BB'
        elif piece[1] == 'D':
            return 'BE'
        elif piece[1] == 'B':
            return 'BC'
        elif piece[1] == 'E':
            return 'BD'
    elif piece[0] == 'V':
        if piece[1] == 'C':
            return 'VB'
        elif piece[1] == 'D':
            return 'VE'
        elif piece[1] == 'B':
            return 'VC'
        elif piece[1] == 'E':
            return 'VD'
    elif piece[0] == 'L':
        if piece[1] == 'H':
            return 'LV'
        elif piece[1] == 'V':
            return 'LH'

def no_action(piece):
    """The piece doesn't move."""
    return piece

def is_valid_move(board, pipe, moves):

    # print("ARGUMENT MOVES: ", moves)

    requesting_connection_sides = []
    result_moves = []
    locked_not_facing = []
    any_lock = 0
    available_directions = 0

    # print("PIECE WERE WORKING WITH", pipe.piece)

    above, below = board.adjacent_vertical_values(pipe.row, pipe.col)
    left, right = board.adjacent_horizontal_values(pipe.row, pipe.col)

    # if above is not None:
        # print("ABOVE IT IS: ", above.piece)
        # print("ABOVE IS LOCK: ", above.lock)
        # print("ABOVE DIREC: ", DIRECTIONS[above.piece])

    piece = pipe.piece
    directions = DIRECTIONS[piece]

    if above is not None:
        available_directions += 1
        if above.lock:
            any_lock = 1
            if (1, 0) in DIRECTIONS.get(above.piece, []):
                requesting_connection_sides.append(OPPOSITE_DIRECTIONS[(1, 0)])
            else:
                locked_not_facing.append(OPPOSITE_DIRECTIONS[(1, 0)])
    if left is not None:
        available_directions += 1
        if left.lock:
            any_lock = 1
            if (0, 1) in DIRECTIONS.get(left.piece, []):
                requesting_connection_sides.append(OPPOSITE_DIRECTIONS[(0, 1)])
            else:
                locked_not_facing.append(OPPOSITE_DIRECTIONS[(0, 1)])
    if right is not None:
        available_directions += 1
        if right.lock:
            any_lock = 1
            if (0, -1) in DIRECTIONS.get(right.piece, []):
                requesting_connection_sides.append(OPPOSITE_DIRECTIONS[(0, -1)])
            else:
                locked_not_facing.append(OPPOSITE_DIRECTIONS[(0, -1)])
    if below is not None:
        available_directions += 1
        if below.lock:
            any_lock = 1
            if (-1, 0) in DIRECTIONS.get(below.piece, []):
                requesting_connection_sides.append(OPPOSITE_DIRECTIONS[(-1, 0)])
            else:
                locked_not_facing.append(OPPOSITE_DIRECTIONS[(-1, 0)])

    # print("Req: ",requesting_connection_sides)
    if any_lock:
        # print("Req in lock: ",requesting_connection_sides)
        # print("Locked not facing: ",locked_not_facing)
        

        if len(requesting_connection_sides) > len(directions):
            return []
        
        # if (available_directions - len(locked_not_facing)) > len(directions):
        #     print(available_directions - len(locked_not_facing))
        #     return []
        
        for move in moves:
            # print("Move: ",move)
            result_piece = apply_action(piece, move[2])
            # print("ResultPiece:" ,result_piece)
            directions = DIRECTIONS[result_piece]

            # print("Directions of result:", directions)
            if set(requesting_connection_sides).issubset(set(directions)) and all(direction not in locked_not_facing for direction in directions):
                result_moves.append(move)

        # print("End result of moves: ", result_moves)
        return result_moves

    else:
        return moves

def apply_action(piece, action):
    """Apply the action to the piece and return the new piece configuration."""
    if action == 1:
        return rotate_clockwise(piece)
    elif action == 2:
        return rotate_counterclockwise(piece)
    elif action == 3:
        return invert(piece)
    elif action == 4:
        return no_action(piece)

class PipeManiaState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1
        # self.queue = set()
        # for row in range(len(self.board.grid)):
        #     for col in range(len(self.board.grid)):
        #         if self.board.grid[row][col].lock == False:
        #             self.queue.add((row, col))
        # for row in self.grid:
        #     row_pieces = [pipe.piece for pipe in row]
        #self.queue = [(board.grid[col], board.grid[col]) for row in board.grid for col in row]
        

    def __lt__(self, other):
        return self.id < other.id

class Board:
    """Representação interna de um tabuleiro de PipeMania."""
    
    def __init__(self, grid):
        """Método criado por mim, para dar à board uma grid
        onde possamos guardar os valores de cada peça"""
        self.grid = grid
        self.size = (len(grid), len(grid[0]))

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.grid[row][col].piece

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        up_row = row - 1
        down_row = row + 1
        if up_row >= 0:
            above = self.grid[up_row][col]
        else:
            above = None
        if down_row < len(self.grid):
            below = self.grid[down_row][col]
        else:
            below = None
        return above, below

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        left_col = col - 1
        right_col = col + 1
        if left_col >= 0:
            left = self.grid[row][left_col]
        else:
            left = None
        if right_col < len(self.grid[row]):
            right = self.grid[row][right_col]
        else:
            right = None
        return left, right            

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 pipe.py < test-01.txt

            > from sys import stdin
            > line = stdin.readline().split()
        """
        from sys import stdin
        lines = stdin.readlines()
        #with open(filename, 'r') as f:
        #    lines = f.readlines()
        #grid = [line.strip().split('\t') for line in lines]
        grid = []
        for row_index, line in enumerate(lines):
            row = line.strip().split('\t')
            pipe_row = []
            for col_index, piece in enumerate(row):
                pipe_row.append(Pipe(piece, row_index, col_index, lock=False))  # Assuming lock status as False
            grid.append(pipe_row)
        
        board_instance = Board(grid)
        board_instance.clean_up()
        board_instance.constraint_propagation()
        return Board(grid)
    
    def print(self):
        """Prints the board with the piece type of each pipe."""

        for row in self.grid:
            row_pieces = [pipe.piece for pipe in row]
            print("\t".join(row_pieces))

    def print_locks(self):
        """Prints the board with the piece type of each pipe."""

        for row in self.grid:
            row_pieces = [str(pipe.lock) for pipe in row]
            print("\t".join(row_pieces))
    
    def clean_up(self):
        """Returns the Board with the pieces that can be fixed right away"""
        grid = self.grid

        # Top-Left
        pipe = self.grid[0][0]
        if pipe.piece.startswith('V'):
            pipe.piece = 'VB'
            pipe.lock = True
        elif pipe.piece.startswith('F'):
            pipe.piece = 'FB'

        # Top-Right
        pipe = self.grid[0][len(grid) - 1]
        if pipe.piece.startswith('V'):
            pipe.piece = 'VE'
            pipe.lock = True
        elif pipe.piece.startswith('F'):
            pipe.piece = 'FB'

        # Bottom-Left
        pipe = self.grid[len(grid) - 1][0]
        if pipe.piece.startswith('V'):
            pipe.piece = 'VD'
            pipe.lock = True
        elif pipe.piece.startswith('F'):
            pipe.piece = 'FC'

        # Bottom-Right
        pipe = self.grid[len(grid) - 1][len(grid) - 1]
        if pipe.piece.startswith('V'):
            pipe.piece = 'VC'
            pipe.lock = True
        elif pipe.piece.startswith('F'):
            pipe.piece = 'FC'
        

        # Sides
        if len(grid) > 2:

            for i in range(1, len(grid) - 1):
            
            # Top-Side
                pipe = self.grid[0][i]

                if pipe.piece.startswith('B'):
                    if pipe.piece != 'BB':
                        pipe.piece = 'BB'
                        pipe.lock = True
                    else:
                        pipe.lock = True
                elif pipe.piece.startswith('L'):
                    if pipe.piece != 'LH':
                        pipe.piece = 'LH'
                        pipe.lock = True
                    else:
                        pipe.lock = True
                elif pipe.piece.startswith('V'):
                    pipe.piece = 'VB'
                elif pipe.piece.startswith('F'):
                    pipe.piece = 'FB'

            # Bottom-Side
                pipe = self.grid[len(grid) - 1][i]

                if pipe.piece.startswith('B'):
                    if pipe.piece != 'BC':
                        pipe.piece = 'BC'
                        pipe.lock = True
                    else:
                        pipe.lock = True
                elif pipe.piece.startswith('L'):
                    if pipe.piece != 'LH':
                        pipe.piece = 'LH'
                        pipe.lock = True
                    else:
                        pipe.lock = True
                elif pipe.piece.startswith('V'):
                    pipe.piece = 'VC'
                elif pipe.piece.startswith('F'):
                    pipe.piece = 'FC'
            
            # Left-Side
                pipe = self.grid[i][0]

                if pipe.piece.startswith('B'):
                    if pipe.piece != 'BD':
                        pipe.piece = 'BD'
                        pipe.lock = True
                    else:
                        pipe.lock = True
                elif pipe.piece.startswith('L'):
                    if pipe.piece != 'LV':
                        pipe.piece = 'LV'
                        pipe.lock = True
                    else:
                        pipe.lock = True
                elif pipe.piece.startswith('V'):
                    pipe.piece = 'VB'
                elif pipe.piece.startswith('F'):
                    pipe.piece = 'FD'

            # Right-Side
                pipe = self.grid[i][len(grid) - 1]

                if pipe.piece.startswith('B'):
                    if pipe.piece != 'BE':
                        pipe.piece = 'BE'
                        pipe.lock = True
                    else:
                        pipe.lock = True
                elif pipe.piece.startswith('L'):
                    if pipe.piece != 'LV':
                        pipe.piece = 'LV'
                        pipe.lock = True
                    else:
                        pipe.lock = True
                elif pipe.piece.startswith('V'):
                    pipe.piece = 'VC'
                elif pipe.piece.startswith('F'):
                    pipe.piece = 'FE'

    def constraint_propagation(self):
        """Returns a board with the pieces that can be moved definitly to the right position"""
        grid = self.grid

        # Adjacent pipes to the ones that come true in parse
        pipes_to_test = set()

        possibilities = []
        act = []

        # Corners

        # Top-Left
        pipe = self.grid[0][0]
        if pipe.lock == True:
            above, below = self.adjacent_vertical_values(pipe.row, pipe.col)
            left, right = self.adjacent_horizontal_values(pipe.row, pipe.col)
            if below.lock == False:
                pipes_to_test.add(below)
            if right.lock == False:
                pipes_to_test.add(right)
        
        # Top-Right
        pipe = self.grid[0][len(grid) - 1]
        if pipe.lock == True:
            above, below = self.adjacent_vertical_values(pipe.row, pipe.col)
            left, right = self.adjacent_horizontal_values(pipe.row, pipe.col)
            if below.lock == False:
                pipes_to_test.add(below)
            if left.lock == False:
                pipes_to_test.add(left)
        
        # Bottom-Left
        pipe = self.grid[len(grid) - 1][0]
        if pipe.lock == True:
            above, below = self.adjacent_vertical_values(pipe.row, pipe.col)
            left, right = self.adjacent_horizontal_values(pipe.row, pipe.col)
            if above.lock == False:
                pipes_to_test.add(above)
            if right.lock == False:
                pipes_to_test.add(right)

        # Bottom-Right
        pipe = self.grid[len(grid) - 1][len(grid) - 1]
        if pipe.lock == True:
            above, below = self.adjacent_vertical_values(pipe.row, pipe.col)
            left, right = self.adjacent_horizontal_values(pipe.row, pipe.col)
            if above.lock == False:
                pipes_to_test.add(above)
            if left.lock == False:
                pipes_to_test.add(left)

            
        # Sides
        if len(grid) > 2:
            for i in range(1, len(grid) - 1):
                
                # Top-Side
                pipe = self.grid[0][i]

                if pipe.lock == True:
                    above, below = self.adjacent_vertical_values(pipe.row, pipe.col)
                    left, right = self.adjacent_horizontal_values(pipe.row, pipe.col)
                    if left.lock == False:
                        pipes_to_test.add(left)
                    if below.lock == False:
                        pipes_to_test.add(below)
                    if right.lock == False:
                        pipes_to_test.add(right)
                
                # Bottom-Side
                pipe = self.grid[len(grid) - 1][i]

                if pipe.lock == True:
                    above, below = self.adjacent_vertical_values(pipe.row, pipe.col)
                    left, right = self.adjacent_horizontal_values(pipe.row, pipe.col)
                    if left.lock == False:
                        pipes_to_test.add(left)
                    if above.lock == False:
                        pipes_to_test.add(above)
                    if right.lock == False:
                        pipes_to_test.add(right)

                # Left-Side
                pipe = self.grid[i][0]

                if pipe.lock == True:
                    above, below = self.adjacent_vertical_values(pipe.row, pipe.col)
                    left, right = self.adjacent_horizontal_values(pipe.row, pipe.col)
                    if above.lock == False:
                        pipes_to_test.add(above)
                    if right.lock == False:
                        pipes_to_test.add(right)
                    if below.lock == False:
                        pipes_to_test.add(below)

                # Right-Side
                pipe = self.grid[i][len(grid) - 1]

                if pipe.lock == True:
                    above, below = self.adjacent_vertical_values(pipe.row, pipe.col)
                    left, right = self.adjacent_horizontal_values(pipe.row, pipe.col)
                    if above.lock == False:
                        pipes_to_test.add(above)
                    if left.lock == False:
                        pipes_to_test.add(left)
                    if below.lock == False:
                        pipes_to_test.add(below)

        while pipes_to_test:

            pipe = pipes_to_test.pop()
            act.clear()
            possibilities.clear()

            # Top-Side
            if pipe.row == 0:
                if pipe.piece.startswith('V'):
                    possibilities.extend([(pipe.row, pipe.col, 1), (pipe.row, pipe.col, 4)])
                elif pipe.piece.startswith('F'):
                    if pipe.col == 0:
                        possibilities.extend([(pipe.row, pipe.col, 2), (pipe.row, pipe.col, 4)])
                    elif pipe.col == len(grid) - 1:
                        possibilities.extend([(pipe.row, pipe.col, 1), (pipe.row, pipe.col, 4)])
                    else:
                        possibilities.extend([(pipe.row, pipe.col, 1), (pipe.row, pipe.col, 2), (pipe.row, pipe.col, 4)])
            
            # Bottom-Side
            elif pipe.row == len(grid) - 1:
                if pipe.piece.startswith('V'):
                    possibilities.extend([(pipe.row, pipe.col, 1), (pipe.row, pipe.col, 4)])
                elif pipe.piece.startswith('F'):
                    if pipe.col == 0:
                        possibilities.extend([(pipe.row, pipe.col, 1), (pipe.row, pipe.col, 4)])
                    elif pipe.col == len(grid) - 1:
                        possibilities.extend([(pipe.row, pipe.col, 2), (pipe.row, pipe.col, 4)])
                    else:
                        possibilities.extend([(pipe.row, pipe.col, 1), (pipe.row, pipe.col, 2), (pipe.row, pipe.col, 4)])
            # Left-Side (except top-left and bottom-left)
            elif pipe.col == 0 and 0 < pipe.row < len(grid) - 1:
                if pipe.piece.startswith('V'):
                    possibilities.extend([(pipe.row, pipe.col, 2), (pipe.row, pipe.col, 4)])
                elif pipe.piece.startswith('F'):
                    possibilities.extend([(pipe.row, pipe.col, 1), (pipe.row, pipe.col, 2), (pipe.row, pipe.col, 4)])
            # Right-Side (except top-right and bottom-right)
            elif pipe.col == len(grid) - 1 and 0 < pipe.row < len(grid) - 1:
                if pipe.piece.startswith('V'):
                    possibilities.extend([(pipe.row, pipe.col, 2), (pipe.row, pipe.col, 4)])
                elif pipe.piece.startswith('F'):
                    possibilities.extend([(pipe.row, pipe.col, 1), (pipe.row, pipe.col, 2), (pipe.row, pipe.col, 4)])
            # Everywhere else
            else:
                if pipe.piece.startswith('L'):
                    possibilities.extend([(pipe.row, pipe.col, 3), (pipe.row, pipe.col, 4)])
                else:
                    possibilities.extend([(pipe.row, pipe.col, 1), (pipe.row, pipe.col, 2), (pipe.row, pipe.col, 3), (pipe.row, pipe.col, 4)])

            act = is_valid_move(self, pipe, possibilities)
        
            if len(act) == 1:
                # print("ACT: ", act)
                row, col, move = act[0]

                result_piece = apply_action(pipe.piece, move)

                self.grid[row][col].piece = result_piece
                self.grid[row][col].lock = True

                above, below = self.adjacent_vertical_values(pipe.row, pipe.col)
                left, right = self.adjacent_horizontal_values(pipe.row, pipe.col)
                
                if above is not None and above.lock == False:
                    pipes_to_test.add(above)
                if below is not None and below.lock == False:
                    pipes_to_test.add(below)
                if left is not None and left.lock == False:
                    pipes_to_test.add(left)
                if right is not None and right.lock == False:
                    pipes_to_test.add(right)

            
                    
                    

            


class Pipe:
    def __init__(self, piece, row, col, lock):
        self.piece = piece
        self.row = row
        self.col = col
        self.lock = lock
    
    def __eq__(self, other):
        if isinstance(other, Pipe):
            return (self.row, self.col) == (other.row, other.col)
        return False

    def __hash__(self):
        return hash((self.row, self.col))

class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = PipeManiaState(board)
        # TODO numero para ir andando por cada peca tipo contador

    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""

        # print("State Queue: ", state.queue)

        actions = []
        potential_moves = []

        def find_first_locked_pipe():
            for i in range(len(state.board.grid)):
                for j in range(len(state.board.grid[i])):
                    if state.board.grid[i][j].lock:
                        return (i, j)
            return None

        first_locked_pipe = find_first_locked_pipe()
        # print(first_locked_pipe)

        if first_locked_pipe:
            start_i, start_j = first_locked_pipe
            grid_size = len(state.board.grid)

            def process_pipe(i, j):
                nonlocal actions, potential_moves

                if actions:
                    return

                pipe = state.board.grid[i][j]

                if not pipe.lock:

                    potential_moves.clear()
                    # Top-Side
                    if i == 0:
                        if pipe.piece.startswith('V'):
                            potential_moves.extend([(i, j, 1), (i, j, 4)])
                        elif pipe.piece.startswith('F'):
                            if j == 0:
                                potential_moves.extend([(i, j, 2), (i, j, 4)])
                            elif j == grid_size - 1:
                                potential_moves.extend([(i, j, 1), (i, j, 4)])
                            else:
                                potential_moves.extend([(i, j, 1), (i, j, 2), (i, j, 4)])
                    # Bottom-Side
                    elif i == grid_size - 1:
                        if pipe.piece.startswith('V'):
                            potential_moves.extend([(i, j, 1), (i, j, 4)])
                        elif pipe.piece.startswith('F'):
                            if j == 0:
                                potential_moves.extend([(i, j, 1), (i, j, 4)])
                            elif j == grid_size - 1:
                                potential_moves.extend([(i, j, 2), (i, j, 4)])
                            else:
                                potential_moves.extend([(i, j, 1), (i, j, 2), (i, j, 4)])
                    # Left-Side (except top-left and bottom-left)
                    elif j == 0 and 0 < i < grid_size - 1:
                        if pipe.piece.startswith('V'):
                            potential_moves.extend([(i, j, 2), (i, j, 4)])
                        elif pipe.piece.startswith('F'):
                            potential_moves.extend([(i, j, 1), (i, j, 2), (i, j, 4)])
                    # Right-Side (except top-right and bottom-right)
                    elif j == grid_size - 1 and 0 < i < grid_size - 1:
                        if pipe.piece.startswith('V'):
                            potential_moves.extend([(i, j, 2), (i, j, 4)])
                        elif pipe.piece.startswith('F'):
                            potential_moves.extend([(i, j, 1), (i, j, 2), (i, j, 4)])
                    # Everywhere else
                    else:
                        if pipe.piece.startswith('L'):
                            potential_moves.extend([(i, j, 3), (i, j, 4)])
                        else:
                            potential_moves.extend([(i, j, 1), (i, j, 2), (i, j, 3), (i, j, 4)])

                    actions = is_valid_move(state.board, pipe, potential_moves)

                    if not actions:
                        return []

            # Process from first locked pipe to the end of the grid
            for i in range(start_i, grid_size):
                for j in range(start_j if i == start_i else 0, grid_size):
                    process_pipe(i, j)
                    if actions:
                        return actions

            # Process from the start of the grid to the first locked pipe
            for i in range(start_i + 1):
                for j in range(0, start_j if i == start_i else grid_size):
                    process_pipe(i, j)
                    if actions:
                        return actions

        return actions



    def result(self, state: PipeManiaState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        new_board = deepcopy(state.board)
        row, col, decision = action
        if 0 <= row < len(new_board.grid) and 0 <= col < len(new_board.grid):
            if decision == 1:
                # Clockwise rotation
                new_piece = rotate_clockwise(new_board.get_value(row, col))
            elif decision == 2:
                # Counter-Clockwise rotation
                new_piece = rotate_counterclockwise(new_board.get_value(row, col))
            elif decision == 3:
                # Invert
                new_piece = invert(new_board.get_value(row, col))
            elif decision == 4:
                # Do nothing
                new_piece = no_action(new_board.get_value(row, col))

            new_board.grid[row][col].piece = new_piece
            new_board.grid[row][col].lock = True
        

        # print("LOCK?: ", new_board.grid[0][2].lock)
        
        # new_board.print()
        # state.board.print_locks()

        # print("Result")
        # [print(j.piece) for i in state.board.grid for j in i]
        # state.board.print()
        # print(action)
        # [print(j.piece) for i in new_board.grid for j in i]
        # new_board.print()
        # new_board.print_locks()

        new_state = PipeManiaState(new_board)
        # if new_state.search_limit > 19:
        #     a
        # if any([j.piece is None for i in new_state.board.grid for j in i]):
        # print(f"Action: {action}")
        # print("Old state")
        # [print(j.piece) for i in state.board.grid for j in i]
        # print("New state")
        # [print(j.piece) for i in new_state.board.grid for j in i]
            

        return new_state

    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""

        def is_valid(next_row, next_col, direction, grid):
            """Function to check if the next cell is valid for water flow"""
            
            if 0 <= next_row < len(grid) and 0 <= next_col < len(grid):
                next_pipe = grid[next_row][next_col]
                # [print(j.piece) for i in grid for j in i]
                # print(next_pipe.piece)
                # print("coordenadas")
                # print(next_pipe.row, next_pipe.col)
                opposite_direction = OPPOSITE_DIRECTIONS[direction]
                return opposite_direction in DIRECTIONS[next_pipe.piece]
            return False

        def bfs_check(grid, start):
            """BFS algorithm to check if the water flows through every pipe without leaks"""
            size = len(grid)
            visited = set()
            queue = deque([start])
            visited.add((start.row, start.col))
            
            while queue:
                current_pipe = queue.popleft()
                row, col = current_pipe.row, current_pipe.col
                piece = current_pipe.piece

                for direction in DIRECTIONS.get(piece, []):
                    next_row, next_col = row + direction[0], col + direction[1]
                
                    if (next_row, next_col) not in visited and is_valid(next_row, next_col, direction, grid):
                        queue.append(grid[next_row][next_col])
                        visited.add((next_row, next_col))  # Mark as visited when enqueued
            
            # Check if all cells were visited
            for row in range(size):
                for col in range(size):
                    if (row, col) not in visited:
                        return False
            
            return True
        
        if bfs_check(state.board.grid, state.board.grid[0][0]):
            return True
        else:
            return False

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        pass

if __name__ == "__main__":

    board = Board.parse_instance()

    problem = PipeMania(board)

    goal_node = depth_first_tree_search(problem)
    
    if goal_node is not None:

        goal_node.state.board.print()

    else:

        print("No solution")