import gymnasium as gym
import pandas as pd
import numpy as np
from gymnasium import spaces

class Sudoku:
    def __init__(self,puzzle_string: str,solution_string: str):
        self.puzzle_string = puzzle_string
        self.board = self.create_board(puzzle_string)
        self.solution = self.create_board(solution_string)


    def print_board(self):
        for i in range(len(self.board)):
            if i % 3 == 0 and i != 0:
                print("---------------------")
            for j in range(len(self.board[0])):
                if j % 3 == 0 and j != 0:
                    print(" | ", end="")
                if j == 8:
                    print(self.board[i][j])
                else:
                    print(str(self.board[i][j]) + " ", end="")
        print("\n")

    def find_empty(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] == 0:
                    return (i, j)
        return None
    
    def find_all_empty(self):
        empty_cells = []
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] == 0:
                    empty_cells.append((i, j))
        return empty_cells
    
    def set_number(self, number, pos):
        self.board[pos[0]][pos[1]] = number
        

    def valid(self, num, pos):
        # Check row
        for i in range(len(self.board[0])):
            if self.board[pos[0]][i] == num and pos[1] != i:
                return False
        # Check column
        for i in range(len(self.board)):
            if self.board[i][pos[1]] == num and pos[0] != i:
                return False
        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3
        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x * 3, box_x*3 + 3):
                if self.board[i][j] == num and (i,j) != pos:
                    return False
        return True
    
    def create_board(self, puzzle_string: str):
        board = []
        for i in range(9):
            board.append([])
            for j in range(9):
                board[i].append(int(puzzle_string[i*9+j]))
        return board
    
    def solve_backtraking(self):
        empty_cell = self.find_empty()

        if not empty_cell:
            return True
        
        for number in range(1, 10):
            if self.valid(number, empty_cell):
                self.set_number(number, empty_cell)
                if self.solve_backtraking():
                    return True
                self.set_number(0, empty_cell)

        return False


class SudokuEnv(gym.Env):
    def __init__(self,sudokus: pd.DataFrame):
        super(SudokuEnv, self).__init__()

        # Define action (row, column, number)
        self.action_space = spaces.Tuple((spaces.Discrete(9), spaces.Discrete(9), spaces.Discrete(9)))
        self.action_space.n = 729
        self.observation_space = spaces.Box(low=0, high=9, shape=(9,9))
        self.sudokus = sudokus
        self.random_sudoku = self.sudokus.sample()

        self.puzzle_string = self.random_sudoku["puzzle"].values[0]
        self.solution_string = self.random_sudoku["solution"].values[0]

        self.sudoku = Sudoku(self.puzzle_string,self.solution_string)
        self.solution = self.sudoku.solution
        self.initial_board = self.sudoku.board.copy()
        self.actual_df_index = 0

    def step(self, action,step=1):
        # action is a tuple of three elements (row, column, number)
        row, column, number = action

        if self.sudoku.valid(number, (row, column)) and self.initial_board[row][column] == 0:
            self.sudoku.set_number(number, (row, column))

            # Check if the board is solved
            done = np.all(self.sudoku.board == self.solution)

            # give reward if the move is valid and the number is correct
            if self.sudoku.board[row][column] == self.solution[row][column]:
                reward = 1
                # extra reward if the board is solved
                if done:
                    reward += 100
            else :
                reward = -1
        else:
            reward = -1
            done = False

        return self.sudoku.board.copy(), reward, done, {}

    def reset(self,random=False):
        if random:
            self.random_sudoku = self.sudokus.sample()
            self.puzzle_string = self.random_sudoku["puzzle"].values[0]
            self.solution_string = self.random_sudoku["solution"].values[0]
        else:   
            self.puzzle_string = self.sudokus["puzzle"].values[self.actual_df_index]
            self.solution_string = self.sudokus["solution"].values[self.actual_df_index]
            self.sudoku = Sudoku(self.puzzle_string,self.solution_string)
        self.solution = self.sudoku.solution
        self.initial_board = self.sudoku.board.copy()
        self.actual_df_index += 1
        return self.sudoku.board.copy()

    def render(self, mode='human'):
        self.sudoku.print_board()

