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
# need to install torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.transforms as T

ACTIONS = 5
# region of interest, crop screen to find ROI
ROI_WIDTH = 200
ROI_HEIGHT = 100
CONV_SIZE = 80

# learning parameter
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
USE_CUDA = torch.cuda.is_available()

# named tuple to record state transitions
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminate'))

# use pytorch build in model to convert image to CONV_SIZE  * CONV_SIZE

resize = T.Compose([T.ToPILImage(),
                    T.Scale(CONV_SIZE, interpolation=Image.CUBIC),
                    T.ToTensor()])

class Variable(autograd.Variable):
    # torch variable class
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


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

def get_roi(x,y):
    # get the region of interests
    screen = pygame.surfarray.array3d(pygame.display.get_surface())
    
    # hard code
    # need change when cars can go down
    y_min = max(0 , y - 100)
    y_max = min(SCREENHEIGHT, y + 100)
    screen = screen[300:500, y_min:y_max]
    screen = screen.transpose((2, 0, 1))  # transpose into torch order (CHW)

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)

def select_action(state):
    # given state, selection action,
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        # some times use the model to select actions
        # action with max score
        return model(Variable(state, volatile=True)).data.max(1)[1].cpu()
    else:
        # some time just random action
        return torch.LongTensor([[random.randrange(ACTIONS)]])