#!/usr/bin/env python
from __future__ import print_function
from collections import  namedtuple

import sys
sys.path.append("game/")
import deep_traffic as game
import random
import numpy as np
import cv2
from itertools import count
from copy import deepcopy
from PIL import Image

# need to install torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.transforms as T

# game constant
GAME = 'Deep Traffic' # the name of the game being played for log files
ACTIONS = 5 # number of valid actions

#  For training the model
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
USE_CUDA = torch.cuda.is_available()

# region of interest, crop screen to find ROI
ROI_WIDTH = 200
ROI_HEIGHT = 100
CONV_SIZE = 80

# named tuple to record state transitions
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminate'))

# use pytorch build in model to convert image to CONV_SIZE  * CONV_SIZE

resize = T.Compose([T.ToPILImage(),
                    T.Scale(CONV_SIZE, interpolation=Image.CUBIC),
                    T.ToTensor()])

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
    # the network to train and test the model
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, ACTIONS)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def get_image():
    # get the region of interests
    screen = pygame.surfarray.array3d(pygame.display.get_surface())
    



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
