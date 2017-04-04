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


# global variable
steps_done = 0

model = net()
memory = ReplayBuffer(10000)
optimizer = optim.RMSprop(model.parameters())

def get_roi(x,y):
    # get the region of interests
    screen = pygame.surfarray.array3d(pygame.display.get_surface())
    
    # hard code
    # need change when cars can go down
    y_min = max(0 , y - 100)
    y_max = min(SCREENHEIGHT, y + 100)
    screen = screen[300:500, y_min:y_max]
    screen = cv2.resize(screen,(80,80), interpolation = cv2.INTER_AREA)
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
        ac=  model(Variable(state, volatile=True)).data.max(1)[1].cpu()
        return ac[0][0]
    else:
        # some time just random action
        return random.randint(0, ACTIONS-1)

def optimize_model():
    # need to sample more
    if len(memory) < BATCH_SIZE:
        return
    # create batch
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(
        tuple(map(lambda s: s == False , batch.terminate)))
    if USE_CUDA:
        non_final_mask = non_final_mask.cuda()

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)

    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train_model():
    # initialize the game
    s = game.GameState()
    do_nothing = 0

    image_data , reward , terminate , (x, y) = s.frame_step(do_nothing)

    while True:
        last_screen = get_roi(x,y)
        current_screen = get_roi(x,y)
        
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(state)

            _, reward, done, (x,y) = s.frame_step(action)

            reward = torch.Tensor([reward])

            # Observe new state
            last_screen = current_screen
            current_screen = get_roi(x,y)
            print (last_screen.size(), current_screen.size())

            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward, done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()

            if done:
                break

def test_simulator(t_max):
	s = game.GameState()
	do_nothing = np.zeros(ACTIONS)
	do_nothing[0] = 1
	
	t = 0
	while t < t_max:
		image_data , reward , terminate , (x, y) = s.frame_step(do_nothing)
		t = t + 1

        

if __name__ == "__main__":
	
    train_model()
