#!/usr/bin/env python
from __future__ import print_function

import torch
import sys
import math
import game.deep_traffic as game

from itertools import count
from game.utils import  *

import numpy as np
import time
import random

from q_learning import (QModel, get_state)
from deep_q_learning import (DQN, USE_CUDA, get_roi, Variable)

# game constant
GAME = 'Deep Traffic' # the name of the game being played for log files
ACTIONS = 5 # number of valid actions


def combine_model(path_to_dqn='model_cpu'):
    model2 = DQN()
    model2.load_state_dict(torch.load(path_to_dqn))

    model1 = QModel()
    model1.q_function = np.loadtxt('model.csv', delimiter=',')

    new_s = game.GameState()
    start = time.time()
    reward = 0
    speed = 0.0
    t = 0
    image_data, reward, terminate, (x, y) , up, red, _ , pedes= new_s.frame_step(0)
    last_screen = get_roi(x, y, up)
    current_screen = get_roi(x, y, up)

    state2 = current_screen - last_screen
    while not terminate:

        state1 = get_state(image_data, x, y, up, red, pedes)

        action = select_action(model1, model2, state1, state2, algorithm=3)

        image_data, r, terminate, (x, y), up, red, sp, pedes = new_s.frame_step(action)

        last_screen = current_screen
        current_screen = get_roi(x, y, up)
        state2 = current_screen - last_screen

        reward += r

        speed += sp

        t += 1

    cur = time.time()

    print('The game last for {} frames'.format(t))
    print('The game last for {} second'.format(cur-start))
    print('The total award : {}'.format(reward))
    print('The average speed is : {}'.format(speed/t))


def select_action(model1, model2, state1, state2, algorithm=0):
    model1_score = np.array([0.0] * 5)

    # get model1 action 5 scores
    for i in range(ACTIONS):
        model1_score[i] = model1.get_q_value(state1, i)

    # get model2 action 5 scores
    model2_score = model2(Variable(state2)).data.numpy().flatten()
    #print(model2_score.shape)

    # normalize
    model1_score = model1_score / np.linalg.norm(model1_score)
    model2_score = model2_score / np.linalg.norm(model2_score)

    # select

    if algorithm==0: # simple average
        final_score = model1_score + model2_score
        return np.argmax(final_score)

    elif algorithm == 1: # which one is more confident
        action1 = np.argmax(model1_score)
        action2 = np.argmax(model2_score)
        if action1 == action2:
            return action1
        else:
            sort1 = np.sort(model1_score)
            sort2 = np.sort(model2_score)

            assert sort1[-1] == model1_score[action1]
            assert sort2[-1] == model2_score[action2]

            diff1 = sort1[-1] - sort1[-2]
            diff2 = sort2[-1] - sort2[-2]

            if diff1 > diff2:
                return action1
            else:
                return action2

    elif algorithm == 2:
        if random.randint(0, 1) == 0:
            return np.argmax(model1_score)
        else:
            return np.argmax(model2_score)
    else: # a little bit hack
        if np.argmax(model1_score) == 3:
            return 3
        else:
            return np.argmax(model2_score)




if __name__ == '__main__':
    combine_model()
