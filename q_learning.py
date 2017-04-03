#!/usr/bin/env python
from __future__ import print_function
from collections import  namedtuple

import sys
sys.path.append("game/")
import deep_traffic as game
import random
import numpy as np
import cv2

# need to install torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.transforms as T


GAME = 'Deep Traffic' # the name of the game being played for log files
ACTIONS = 5 # number of valid actions


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminate'))

class ReplayBuffer():
    # the Buffer to store all the frames
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1 ) % self.capacity    

    def sample(self, batch_size):
        # get minibatch of images
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class net(nn.Module):
    
    def __init__(self):
        pass

    def forward(self, ):
        pass

def get_image():
    # get the region of interests
    pass



def train_model():
    pass

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
