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
from agents.PPOAgent import PPOAgent

import torch


def main():
    # read the sudokus from a csv file (https://www.kaggle.com/datasets/rohanrao/sudoku)
    #load the first 10000 sudokus
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")

    df_sudoku = pd.read_csv('sudoku_sorted_easy.csv')
    
    env = SudokuEnv(df_sudoku)
    
    agent = PPOAgent(env)
    agent.train(100000)
    agent.save_checkpoint('checkpoint_hard.pth.tar')




    
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

    

    


