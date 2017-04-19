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
from deep_q_learning import (DQN, USE_CUDA, get_roi)

# game constant
GAME = 'Deep Traffic' # the name of the game being played for log files
ACTIONS = 5 # number of valid actions


def combine_model(path_to_dqn):
    model1 = DQN()
    model1.load_state_dict(torch.load(path_to_dqn))

    if USE_CUDA:
        model1.cuda()

    model2 = QModel()

    new_s = game.GameState()
    start = time.time()
    reward = 0
    speed = 0.0
    t = 0
    image_data, reward, terminate, (x, y) , up, red, _ , pedes= new_s.frame_step(0)

    while not terminate:

        state = get_state(image_data, x, y, up, red, pedes)

        action = select_action(model1, model2)

        image_data, r, terminate, (x, y) , up, red, sp, pedes = new_s.frame_step(action)

        reward += r

        speed += sp

        t += 1

    cur = time.time()

    print('The game last for {} frames'.format(t))
    print('The game last for {} second'.format(cur-start))
    print('The total award : {}'.format(reward))
    print('The average speed is : {}'.format(speed/t))


def select_action(model1, model2):
    pass
