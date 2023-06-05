# main function for the program

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from enviroment.enviroment import Sudoku


def main():
    # read the sudokus from a csv file (https://www.kaggle.com/datasets/rohanrao/sudoku)
    #load the first 10000 sudokus
    df_sudoku = pd.read_csv('sudoku.csv', nrows=10000)
    #create a sudoku object
    game = Sudoku(df_sudoku['puzzle'][0], df_sudoku['solution'][0])
    game.print_board()
    game.solve_backtraking()
    game.print_board()



if __name__ == "__main__":
    main()

    

    


