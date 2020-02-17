import argparse
import queue
import copy
from dataclasses import dataclass, field
from typing import Any

# determine position and weight of each queen
def set_queens(input_file):
    queens = []
    with open(input_file) as board_data:
        csv_reader = csv.reader(board_data, delimiter=',')
        row_count = 1
        # iterate through each row of the board
        for row in csv_reader:
            # for each row, locate the queens in that row
            find_queens_in_row(row, row_count, queens)
            row_count += 1
    return queens


# find every queen in a given row
def find_queens_in_row(row, row_count, queens):
    # iterate the columns of a given row
    for col in range(len(row)):
        weight = row[col]   # get the weight
        if weight.isdigit() == True:
            # create a new queen tuple if a queen exists in this position
            new_queen = ((row_count, col+1), int(weight))
            # add the queen to the "queens" list
            queens.append(new_queen)



def create_queens(data, board):
    queens = []
    for i, q in enumerate(data):
        queen = Queen(id=i, position=q[0], weight=q[1])
        queen.determine_initial_attacks(board)
        queens.append(queen)
    for q in queens:
        board.add_attacks(q)
    return queens, board


class Queen:
    def __init__(self, id, position, weight):
        self.position = position
        self.weight = weight
        self.attacking_positions = set()
        self.id = id

    def is_attacking(self, board):
        if board.check_position(self.position):
            return True
        return False

    def move(self, spaces):
        self.position[1] = self.position[1] + spaces

    def find_all_possible_moves(self, board):
        moves = set((self.position[0], i + 1) for i in range(board.size))
        moves.remove(self.position)
        return moves

    def __repr__(self):
        return "Queen: {}; position: {}; weight: {}".format(self.id, self.position, self.weight)

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return other.id == self.id

    def __lt__(self, other):
        if self.weight < other.weight:
            return self
        elif other.weight < self.weight:
            return other
        elif self.weight == other.weight:
            return self

    def determine_initial_attacks(self, board):
        for i in range(1, board.size + 1):
            if i != self.position[0]:
                horizontal = (i, self.position[1])
                self.attacking_positions.add(horizontal)

            # Forward Horizontal Diaganol up
            for_diaganol_up = self.position[1] + (i - self.position[0])
            if for_diaganol_up <= board.size and i > self.position[0]:
                self.attacking_positions.add((i, for_diaganol_up))
            # Forward Horizontal Diaganol down
            for_diaganol_down = self.position[1] - (i - self.position[0])
            if for_diaganol_down > 0 and i > self.position[0] and i < board.size:
                self.attacking_positions.add((i, for_diaganol_down))

            # Reverse Horizontal Diaganol up
            rev_diaganol_up = self.position[1] + (self.position[0] - i)
            if rev_diaganol_up <= board.size and i < self.position[0]:
                self.attacking_positions.add((i, rev_diaganol_up))
            # Reverse Horizontal Diaganol down
            rev_diaganol_down = self.position[1] - (self.position[0] - i)
            if rev_diaganol_down > 0 and i < self.position[0] and i < board.size:
                self.attacking_positions.add((i, rev_diaganol_down))


class Board:
    def __init__(self, n):
        self.size = n
        self._board = self.make_board()

    def make_board(self):
        board = {}
        for x in range(1, self.size + 1):
            for y in range(1, self.size + 1):
                board[(x, y)] = set()
        return board

    def add_attacks(self, queen):
        for pos in queen.attacking_positions:
            self._board[pos].add(queen)

    def remove_attacks(self, queen):
        for pos in queen.attacking_positions:
            self._board[pos].remove(queen)

    def show_board(self):
        print(self._board)

    def check_position(self, pos):
        return self._board[pos]

    def get_queens_attacking(self, queen):
        attackers = self.check_position(queen.position)
        return attackers

    def copy(self):

        new_board = self.__class__(n=self.size)
        setattr(new_board, '_board', copy.deepcopy(self._board))
        return new_board


@dataclass(order=True)
class PrioritizedNode:
    cost: int
    heuristic: int=field(compare=False)
    id: int=field(compare=False)
    sorted_queens: Any=field(compare=False)
    queens: Any=field(compare=False)
    board: Any=field(compare=False)

    def __hash__(self):
        return hash(self.sorted_queens)

    def __eq__(self, other):
        return self.sorted_queens == other.sorted_queens


def move_cost(num_tiles, weight):
    return num_tiles * weight**2


def heuristic_one(queens, board):
    try:
        return min(queen.weight for queen in queens if queen.is_attacking(board))
    except ValueError:
        return 0


def expand_nodes(node, priority):
    board = node.board
    queens = node.queens
    start_cost = node.cost

    for q in queens:

        moves = q.find_all_possible_moves(board)
        for m in moves:

            # Queen stuff
            queens_copy = queens.copy()
            queens_copy.remove(q)
            new_queen = Queen(q.id, m, q.weight)
            new_queen.determine_initial_attacks(board)
            queens_copy.append(new_queen)

            # Board stuff
            new_board = board.copy()
            new_board.remove_attacks(q)
            new_board.add_attacks(new_queen)

            h1 = heuristic_one(queens_copy, new_board)
            cost = move_cost(abs(q.position[1] - m[1]), q.weight)
            total_cost = cost + start_cost

            new_node = PrioritizedNode(cost=total_cost,
                                       queens=queens_copy,
                                       sorted_queens=tuple(sorted([q.position for q in queens_copy], key=lambda q: q[0])),
                                       board=new_board,
                                       id=node.id + 1,
                                       heuristic=h1)

            priority.put(new_node)


def a_star(board, queens):
    sorted_queens = tuple(sorted([q.position for q in queens], key=lambda q: q[0]))
    priority = queue.PriorityQueue()
    node_id = 0
    h1 = heuristic_one(queens, board)
    start_node = PrioritizedNode(cost=0, queens=queens, board=board, sorted_queens=sorted_queens, id=node_id, heuristic=h1)
    priority.put(start_node)

    current_node = None
    history = {start_node: 0}

    depth = 0
    while not priority.empty():
        next_node = priority.get()
        if next_node.id > depth:
            depth += 1
            print("depth: ", depth)

        if next_node.heuristic == 0:
            return next_node

        if next_node in history and history[next_node] < next_node.cost:
            continue
        history[next_node] = next_node.cost
        expand_nodes(next_node, priority)

def get_input():
    my_parser = argparse.ArgumentParser(description='Please add some command line inputs... if you do not, I won"t know how to behave')
    my_parser.add_argument('input_file', help='Name of the file containing the n-queens board')
    my_parser.add_argument('strategy', help='Specifies the search technique')
    my_parser.add_argument('heuristic', help='Specifies the heuristic (H1 or H2)')
    args = my_parser.parse_args()
    return args.input_file, args.strategy, args.heuristic



if __name__ == '__main__':
    # get command line input
    input_file, strategy, heuristic = get_input()
    # get the locations and weights of the queens
    queens = set_queens(input_file)
    board = Board(n=len(queens))
    queens, board = create_queens(queens, board)

    for q in queens:
        print(q.weight)
        print(q.attacking_positions)
        print()
    node = a_star(board=board, queens=queens)
    print("Finished!", node)
