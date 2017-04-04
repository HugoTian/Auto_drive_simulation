#!/usr/bin/env python
from __future__ import print_function
from collections import  namedtuple
import sys
import math
sys.path.append("game/")
import deep_traffic as game
import random
import numpy as np
import cv2
from itertools import count
from copy import deepcopy
from PIL import Image
from utils import *
from policy import *

# need to install torch
import torch.optim as optim


# game constant
GAME = 'Deep Traffic' # the name of the game being played for log files
ACTIONS = 5 # number of valid actions

#  For training the model
BATCH_SIZE = 128
GAMMA = 0.999


# global variable
steps_done = 0

model = net()
memory = ReplayBuffer(10000)
optimizer = optim.RMSprop(model.parameters())


def train_model():
    # need to sample more
    if len(memory) < BATCH_SIZE:
        return
    # create batch
    batch = Transition(*zip(*transitions))

    

def test_simulator(t_max):
	s = game.GameState()
	do_nothing = np.zeros(ACTIONS)
	do_nothing[0] = 1
	
	t = 0
	while t < t_max:
		image_data , reward , terminate , (x, y) = s.frame_step(do_nothing)
		t = t + 1

        

if __name__ == "__main__":
	
    test_simulator(1000)
