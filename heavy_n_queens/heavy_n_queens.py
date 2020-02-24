import argparse
import datetime
import random
import queue
import copy
import csv
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# determine position and weight of each queen
def set_queens(input_file):
    data = []
    with open(input_file) as board_data:
        csv_reader = csv.reader(board_data, delimiter=',')
        # iterate through each row of the board
        for col, row in enumerate(csv_reader):
            # for each row, locate the queens in that row
            for i, c in enumerate(row):
                try:
                    weight = int(c)
                except:
                    continue
                data.append(((i + 1, col + 1), weight))

    return data


def create_random_queens(n, board):
    queens = []
    for i in range(n):
        pos = (random.randint(1, n), i + 1)
        q = Queen(id=i, position=pos, weight=random.randint(1, 9))
        q.determine_initial_attacks(board)
        queens.append(q)
    for q in queens:
        board.add_attacks(q)
    return queens, board


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


def sort_queens(queens):
    return tuple(sorted([q.position for q in queens], key=lambda q: q[0]))

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

    def is_solution(self):
        for q in self.queens:
            if q.is_attacking(self.board):
                return False
        return True

    @property
    def came_from(self):
        return self._came_from

    @came_from.setter
    def came_from(self, value):
        self._came_from = value

def move_cost(num_tiles, weight):
    return num_tiles * weight**2


class HeuristicOne:
    def __repr__(self):
        return "HeuristicOne"

    def __call__(self, queens, board):
        try:
            return min(queen.weight**2 for queen in queens if queen.is_attacking(board))
        except ValueError:
            return 0


class HeuristicTwo:
    def __repr__(self):
        return "HeuristicTwo"

    def __call__(self, queens, board):
        attacking_pairs = set()
        for q in queens:
            attackers = board.get_queens_attacking(q)
            for a in attackers:
                lq = a if a.position[0] < q.position[0] else q
                gq = a if a.position[0] > q.position[0] else q
                mw = a.weight if a.weight < q.weight else q.weight
                attacking_pairs.add((lq, gq, mw))
        return sum(pair[2]**2 for pair in attacking_pairs)
    
class HeuristicThree:
    def __repr__(self):
        return "HeuristicThree"

    def __call__(self, queens, board):
        try:
            attacking_dict = {}
            set_list = []
            for i, queen in enumerate(queens):
                attacked = board.get_queens_attacking(queen)
                for a_queen in attacked:
                    j = a_queen.id
                    if i not in attacking_dict and j not in attacking_dict:
                        attacking_dict[i] = set()
                        attacking_dict[j] = attacking_dict[i]
                        set_list.append(attacking_dict[i])
                    elif i not in attacking_dict:
                        attacking_dict[i] = attacking_dict[j]
                    elif j not in attacking_dict:
                        attacking_dict[j] = attacking_dict[i]
                    else:
                        attacking_dict[i].update(attacking_dict[j])
                        attacking_dict[j] = attacking_dict[i]
                    attacking_dict[i].add(queens[i])
                    attacking_dict[i].add(queens[j])
            h3 = 0
            for s in set_list:
                minc = min(queen.weight for queen in s)
                h3 += minc**2
            return h3
        except ValueError:
            return 0    

def expand_node(node, heuristic):
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

            h1 = heuristic(queens_copy, new_board)
            cost = move_cost(abs(q.position[1] - m[1]), q.weight)
            total_cost = cost + start_cost

            yield PrioritizedNode(cost=total_cost,
                                  queens=queens_copy,
                                  sorted_queens=sort_queens(queens_copy),
                                  board=new_board,
                                  id=node.id + 1,
                                  heuristic=h1)


def a_star(board, queens, heuristic):
    sorted_queens = sort_queens(queens)
    priority = queue.PriorityQueue()
    node_id = 0
    h1 = heuristic(queens, board)
    start_node = PrioritizedNode(cost=0,
                                 queens=queens,
                                 board=board,
                                 sorted_queens=sorted_queens,
                                 id=node_id,
                                 heuristic=h1)
    print("ASTAR Start Node: ", start_node)
    priority.put(start_node)
    history = {start_node: 0}

    current_node = start_node
    #came_from = {}
    # Solution space search control structure
    depth = 0
    nodes_processed = 0
    while not priority.empty():
        next_node = priority.get()
        #came_from[next_node] = current_node

        # For visualizing solution depth
        if next_node.id > depth:
            depth += 1
            print("depth: ", depth)

        # Check Goal Condition
        if next_node.heuristic == 0:
            return next_node, nodes_processed

        # Don't expand nodes that already exist in the history at a lower cost
        if next_node in history and history[next_node] < next_node.cost:
            continue
        history[next_node] = next_node.cost
        # Nodes are expanded one at a time with a generator and added to the priority queue
        for node in expand_node(next_node, heuristic):
            node.came_from = current_node
            priority.put(node)
        nodes_processed += 1
        current_node = next_node

    return current_node, nodes_processed

def get_next_node(nodes, current_node, history, sideway):
    best_node = None
    # Find best node
    for node in nodes:
        if not best_node or node.heuristic <= best_node.heuristic:
            if node in history and history[node] <= node.cost:
                continue
            best_node = node

    # Decide to return best node
    try:
        if best_node.heuristic < current_node.heuristic:
            return best_node
        elif best_node.heuristic == current_node.heuristic and sideway:
            return best_node
        elif best_node.heuristic > current_node.heuristic:
            return False
    except:
        return False


def greedy_hill_climb_with_restarts(board, queens, heuristic, sideway_moves):
    history = dict()
    start = time.time()
    best_node, nodes_processed = greedy_hill_climb(board, queens, heuristic, sideway_moves, history, start=True)
    total_nodes_process = nodes_processed
    while best_node.heuristic != 0 and (time.time() - start) < 10:
        node, nodes_processed = greedy_hill_climb(board, queens, heuristic, sideway_moves, history)
        total_nodes_process += nodes_processed
        if node.heuristic == 0:
            return node, total_nodes_process
        elif node.heuristic < best_node.heuristic:
            best_node = node
        elif node.heuristic == best_node.heuristic and node.cost < best_node.cost:
            best_node = node
        #print("restarts: ", restarts)
    return best_node, total_nodes_process


def greedy_hill_climb(board, queens, heuristic, sideway_moves, history, start=False):
    h1 = heuristic(queens, board)
    sorted_queens = sort_queens(queens)
    node_id = 0
    start_node = PrioritizedNode(cost=0,
                                 heuristic=h1,
                                 id=node_id,
                                 queens=queens,
                                 sorted_queens=sorted_queens,
                                 board=board)
    if start:
        print("Greedy Hill Climb Start Node: ", start_node)
    current_node = start_node

    depth = 0
    sideways_used = 0
    nodes_processed = 0
    interval_count = 0
    while current_node.heuristic != 0:

        # For visualizing solution depth
        if current_node.id > depth:
            depth += 1
            #print("depth: ", depth)

        nodes = expand_node(current_node, heuristic)
        next_node = get_next_node(nodes, current_node, history, sideway=(sideway_moves > sideways_used))
        if not next_node:
            history[current_node] = current_node.cost
            return current_node, nodes_processed
        next_node.came_from = current_node
        nodes_processed +=1
        history[current_node] = current_node.cost

        if current_node.heuristic == next_node.heuristic:
            sideways_used += 1

        current_node = next_node
        interval_count += 1
        #print('interval count: ', interval_count)

    return current_node, nodes_processed


def run_algorithms(heuristics, algos, board, queens, out_file, sideway_moves=None):
    print("Starting Queens: ", queens)
    n = len(queens)
    results = []
    for heuristic in heuristics:
        if 'astar' in algos:
            algo = 'astar'
            start = time.time()
            result_node, nodes_processed = a_star(board, queens, heuristic)
            end = time.time()
            runtime = end - start
            if result_node.is_solution():
                process_results(algo, heuristic, n, result_node, nodes_processed, runtime, out_file)

        if 'hill_climb' in algos:
            algo = 'hill_climb'
            start = time.time()
            result_node, nodes_processed = greedy_hill_climb_with_restarts(board, queens, heuristic, sideway_moves)
            end = time.time()
            runtime = end - start

            if result_node.is_solution():
                process_results(algo, heuristic, n, result_node, nodes_processed, runtime, out_file)
                continue
            process_negative_results(algo, heuristic, n, runtime, out_file)
        algo_results = {'algo': algo,
                         'runtime': runtime,
                         'node': result_node}
        results.append(algo_results)
    return results


def process_negative_results(algo, heuristic, n, runtime, out_file):
    results = f"\nAlgorithm: {algo}; Heuristic: {heuristic}; True_result: {False}; Number of queens: {n}; Runtime: {runtime};\n"
    with open(out_file, 'a') as f:
        f.write(results)
    print(results)


def process_results(algo, heuristic, n, node, nodes_processed, runtime, fout):
    result = node
    solution = [result]
    finished = False
    while result.id != 0:
        result = result.came_from
        solution.append(result)

    effective_branching = nodes_processed**(1/len(solution))

    solution_format = ''
    for i, q in enumerate(solution):
        solution_format += 'Move Number: ' + str(i) + ': ' + str(q) + '\n'

    results = f"\nAlgorithm: {algo}; Heuristic: {heuristic}; True_result: {True}; Number of queens: {n}; Runtime: {runtime}; Cost: {node.cost}; Heuristic_score: {node.heuristic}; Depth: {len(solution) - 1}; Effective_Branching: {effective_branching}; Nodes_Processed: {nodes_processed}\nSolution_Queens: {node.queens}\nSolution_Order: {solution_format}\n"
    with open(fout, 'a') as f:
        f.write(results)
    print(results)


def test_algos():
    queens = [((1, 1), 9), ((2, 3), 9), ((3, 2), 1), ((4, 4), 1), ((5, 3), 1)]
    board = Board(n=len(queens))
    queens, board = create_queens(queens, board)

    # node = a_star(board=board, queens=queens, heuristic=heuristic_one)
    node = greedy_hill_climb_with_restarts(board=board, queens=queens, sideway_moves=8,
                                           heuristic=HeuristicOne)
    if node.is_solution():
        print("Finished.... Success!", node)
    else:
        print("ooops...", node)


def evaluate_algorithms():
    out_file = "analysis/algorithm_evaluation"
    algos = ['hill_climb', 'astar']
    heuristics = [HeuristicOne(), HeuristicTwo()]
    n = 5
    sideway_moves = 9
    num_trials = 10

    board = Board(n)
    queens, board = create_random_queens(n, board)
    for i in range(num_trials):
        board = Board(n)
        queens, board = create_random_queens(n, board)
        with open(out_file, 'a') as f:
            f.write("--------------Trial----------------")
            print("--------------Trial----------------")
        results = run_algorithms(heuristics, algos, board, queens, out_file, sideway_moves)


def determine_sideways_moves(heuristic, restart_interval, n, sideway_moves, fout):
    """ algo, true_result, sideways_moves, number of queens, h1, Runtime, Cost, Heuristic, Depth, Queens,\n"
        """
    try:
        board = Board(n)
        queens, board = create_random_queens(n, board)
        start = time.time()
        hc_node = greedy_hill_climb_with_restarts(board, queens, heuristic, sideway_moves)
        end = time.time()

        runtime = end - start
        results = f"GREEDY_HILL, {hc_node.is_solution()}, {sideway_moves}, {n}, {hc_node.heuristic}, {runtime}, {hc_node.cost}, {heuristic}, {hc_node.id}, {queens}\n"
        fout.write(results)
        print(results)
    except KeyboardInterrupt:
        end = time.time()
        runtime = end - start
        results = f"GREEDY_HILL, {False}, {sideway_moves}, {n}, {None}, {runtime}, {None}, {heuristic}, {None}, {None}\n"
        fout.write(results)


def run_sideways_eval():
    my_parser = argparse.ArgumentParser(
        description='Please add some command line inputs... if you do not, I won"t know how to behave')
    my_parser.add_argument('n', help='Name of the file containing the n-queens board', type=int)
    my_parser.add_argument('sideway_moves', help='Specifies the search technique', type=int)
    my_parser.add_argument('heuristic', help='Specifies the heuristic (H1 or H2)')
    args = my_parser.parse_args()
    print(args)
    with open("analysis/sideways.txt", "a") as f:
        try:
            determine_sideways_moves(HeuristicOne, n=args.n, sideway_moves=args.sideway_moves, fout=f)
        except KeyboardInterrupt:
            pass


def cli():
    input_file, algorithm, h = get_input()
    data = set_queens(input_file)
    #data = queens = [((1, 2), 9), ((2, 3), 9), ((3, 4), 1), ((4, 2), 1), ((5, 3), 1)]
    board = Board(len(data))
    queens, board = create_queens(data, board)
    print(queens)
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print(t)

    if h.upper() == "H1":
        heuristic = HeuristicOne()
    elif h.upper() == "H2":
        heuristic = HeuristicTwo()
    elif h.upper() == "H3":
        heuristic = HeuristicThree()

    run_algorithms([heuristic], [algorithm.lower()], board, queens, out_file=f"{algorithm}_{h}_results_{t}.txt", sideway_moves=9)


def get_input():
    my_parser = argparse.ArgumentParser(
        description='Please add some command line inputs... if you do not, I won"t know how to behave')
    my_parser.add_argument('input_file', help='Name of the file containing the n-queens board')
    my_parser.add_argument('algorithm', help='Specifies the search technique')
    my_parser.add_argument('heuristic', help='Specifies the heuristic (H1 or H2)')
    args = my_parser.parse_args()
    return args.input_file, args.algorithm, args.heuristic


def analyze_eval():
    import re

    astar1_values = []
    astar2_values = []
    hc1_values = []
    hc2_values = []

    with open("/analysis/algorithm_evaluation", 'r') as f:
        headers = False
        for line in f:
            header_vals = []
            vals = []
            if re.search('^Algorithm: astar; Heuristic: HeuristicOne;', line):
                for i in line.split(';'):
                    key, val = i.split(':')
                    if not headers:
                        header_vals.append(key.strip())
                    vals.append(val.strip())
                if not headers:
                    headers = header_vals
                astar1_values.append(vals)
            elif re.search('^Algorithm: astar; Heuristic: HeuristicTwo;', line):
                for i in line.split(';'):
                    key, val = i.split(':')
                    if not headers:
                        header_vals.append(key.strip())
                    vals.append(val.strip())
                if not headers:
                    headers = header_vals
                astar2_values.append(vals)

            elif re.search('^Algorithm: hill_climb; Heuristic: HeuristicOne;', line):
                for i in line.split(';'):
                    key, val = i.split(':')
                    if not headers:
                        header_vals.append(key.strip())
                    vals.append(val.strip())
                if not headers:
                    headers = header_vals
                hc1_values.append(vals)

            elif re.search('^Algorithm: hill_climb; Heuristic: HeuristicTwo;', line):
                for i in line.split(';'):
                    key, val = i.split(':')
                    if not headers:
                        header_vals.append(key.strip())
                    vals.append(val.strip())
                if not headers:
                    headers = header_vals
                hc2_values.append(vals)

    df_a1 = pd.DataFrame(astar1_values, columns = headers)
    df_a1 = df_a1.sort_values(by=['Depth'], ascending=False)
    df_a1.to_csv("./astar_1_analysis.csv")

    df_a2 = pd.DataFrame(astar2_values, columns=headers)
    df_a2 = df_a2.sort_values(by=['Depth'], ascending=False)
    df_a2.to_csv("./astar_2_analysis.csv")

    df_h1 = pd.DataFrame(hc1_values, columns=headers)
    df_h1 = df_h1.sort_values(by=['Depth'], ascending=False)
    df_h1.to_csv("./hc_1_analysis.csv")

    df_h2 = pd.DataFrame(hc2_values, columns=headers)
    df_h2 = df_h2.sort_values(by=['Depth'], ascending=False)
    df_h2.to_csv("./hc_2_analysis.csv")

def gen_stats():
    paths = ["./astar_2_analysis.csv",
             "./hc_1_analysis.csv",
             "./astar_1_analysis.csv",
             "./hc_2_analysis.csv"]
    for path in paths:
        df = pd.read_csv(path)
        df = df[df.Cost != 0]
        stats = df.describe()
        stats.to_csv(path.replace('analysis', 'stats'))

if __name__ == '__main__':
    cli()
    #evaluate_algorithms()
    # analyze_eval()
    # gen_stats()


