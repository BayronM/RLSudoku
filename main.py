# main function for the program

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from enviroment.enviroment import Sudoku, SudokuEnv
from agents.QLearningAgent import QLearningAgent
from agents.DQNAgent import DQNAgent

import torch


def main():
    # read the sudokus from a csv file (https://www.kaggle.com/datasets/rohanrao/sudoku)
    #load the first 10000 sudokus
    df_sudoku = pd.read_csv('sudoku_sorted.csv', nrows=10000)
    
    env = SudokuEnv(df_sudoku)
    
    agent = DQNAgent(env)
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")
    agent.train(1000)

    agent.test(10)


    
    # # train the agent
    # agent.train(5000)

    # # test the agent
    # state = env.reset()
    # env.render()
    # for step in range(10000):
    #     action = agent.choose_action(state)
    #     new_state, reward, done, info = env.step(action)
    #     state = new_state

    #     if done:
    #         break
    # env.render()




if __name__ == "__main__":
    main()

    

    


