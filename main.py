# main function for the program

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from enviroment.enviroment import Sudoku, SudokuEnv


def main():
    # read the sudokus from a csv file (https://www.kaggle.com/datasets/rohanrao/sudoku)
    #load the first 10000 sudokus
    df_sudoku = pd.read_csv('sudoku.csv', nrows=10000)

    env = SudokuEnv(df_sudoku['puzzle'][0], df_sudoku['solution'][0])
    
    done = False
    reward = 0
    observation= env.reset()

    while not done:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        print(f"Action: {action}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        print("")
    
    env.close()





if __name__ == "__main__":
    main()

    

    


