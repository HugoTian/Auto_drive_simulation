#!/usr/bin/env python
from __future__ import print_function

import sys
import math
import game.deep_traffic as game

from itertools import count
from game.utils import *

import numpy as np
import time
import random

#####################################
# game interface, function 

# image_data, reward, terminate, (x, y) , up, red, speed , pedes = s.frame_step(do_nothing)

# state
# left : 0 nothing, 1 pedes, 2, car
# right : 0 nothing, 1 pedes, 2 car
# front : 0 nothing, 1 pedes or car, 2 traffic light

# total 27 states

######################################

# game constant
GAME = 'Deep Traffic' # the name of the game being played for log files
ACTIONS = 5 # number of valid actions

#  For training the model
BATCH_SIZE = 1
GAMMA = 0.999
# region of interest, crop screen to find ROI
ROI_WIDTH = 200
ROI_HEIGHT = 100

UP_THRESHOLD = 50000
# learning parameter
EPS_START = 0.9
EPS_END = 0.02
EPS_DECAY = 200

# game
s = game.GameState()

steps_done = 0
# named tuple to record state transitions
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminate'))

State = namedtuple('State', ('left', 'right', 'front'))

final_policy = None

state_table = {

    'left': {
        150: (240, 280, 60),
        240: (None, None, None),
        330: (None, None, None),
        400: (330, 370, 60)
    },
    'right': {
        150: (None, None, None),
        240: (150, 190, 60),
        330: (400, 440, 60),
        400: (None, None, None)
    },
    'front': {
        150: (150, 190, 70),
        240: (240, 280, 70),
        330: (330, 370, -70),
        400: (400, 440, -70)
    }

}


def check_state(target, img, x, y, up, red, pedes):
    #check state

    x1, x2, y_delta = state_table[target][x]

    if not x1 :
        return 2
    if up:
        if target == 'front':
            y_up = min(y+y_delta, SCREENHEIGHT)
            y_up = max(0, y_up)
        else:
            y_up = min(y+y_delta, SCREENHEIGHT)
            y_up = max(0, y_up)

    else:
        if target == 'front':
            y_up = min(y+y_delta+60, SCREENHEIGHT)
            y_up = max(0, y_up)
        else:
            y_up = min(y+y_delta, SCREENHEIGHT)
            y_up = max(0, y_up)

    if y < y_up:
        roi = np.array(img[x1:x2, y:y_up])
    else:
        roi = np.array(img[x1:x2, y_up:y])

    overall = roi.sum()
    print(target, overall, y, y_up)
    # front case
    if target == 'front':
        
        if up:
            if y > RED_STOP_UP and y - RED_STOP_UP <  RED_LIGHT_THRESHOLD and red:
                return 2
            elif overall < UP_THRESHOLD:
                return 0
            else:
                return 1
        else:
            if y < RED_STOP_DOWN and RED_STOP_DOWN - y < RED_LIGHT_THRESHOLD and red:
                return 2
            elif overall < UP_THRESHOLD:
                return 0
            else:
                return 1
    elif target == 'left': #left case

        if pedes:
            x_p, y_p, l_p = pedes.x, pedes.y, pedes.left

            if up and not l_p and x > x_p and x - x_p < 40 and y > y_p and y - y_p < 60:
                return 1
            elif not up and  l_p and x < x_p and x_p - x < 40 and y_p > y and y_p - y < 60:
                return 1
            elif overall ==0:
                return 0
            else:
                return 2 
        else:
            if overall == 0:
                return 0
            else:
                return 2
    else:

        if pedes:
            x_p, y_p, l_p = pedes.x, pedes.y, pedes.left

            if up and l_p and x < x_p and x_p - x < 40 and y > y_p and y - y_p < 60:
                return 1
            elif not up and not l_p and x > x_p and x - x_p < 40 and y_p > y and y_p - y < 60:
                return 1
            elif overall == 0:
                return 0
            else:
                return 2 
        else:
            if overall == 0:
                return 0
            else:
                return 2

def get_state(image_data, x, y, up,  red, pedes):
    # return
    # left : left lane car or not ?
    # right : right lane car or not
    # front : 0 or 1 or 2, 0 : nothing , 1: car, 2:red light
    # terminate

    left = check_state('left', image_data, x, y, up, red, pedes)
    right = check_state('right', image_data,x, y, up, red, pedes)
    front = check_state('front', image_data,x, y, up,  red, pedes)

    assert left in (0,1,2)
    assert right in (0,1,2)
    assert front in (0, 1, 2)
    print(left, right, front)
    return State(left, right, front)


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


class QModel:
    def __init__(self):
        self.num_state = 27
        self.q_function = np.random.rand(self.num_state, ACTIONS)
        self.state_dict = {
            'left' : {0 : 0, 1:9, 2:18},
            'right': {0 : 0, 1:3, 2:6 },
            'front' : {0 : 0, 1:1, 2:2}
        }
        # hack initialize

        for i in range(self.num_state):
            left = i / 9
            right = (i / 3) % 3
            front = i % 3

            if front == 0:
                self.q_function[i][3] = 100
            elif front == 2:
                self.q_function[i][4] = 100
            else:
                if left == 0:
                    self.q_function[i][1] = 100
                elif right == 0:
                    self.q_function[i][2] = 100
                else:
                    self.q_function[i][4] = 100

    def state_to_int(self, state):
        return self.state_dict['left'][state.left]+ self.state_dict['right'][state.right] + self.state_dict['front'][state.front]

    def get_q_value(self, state, action):
        value = self.state_to_int(state)
        return self.q_function[value][action]

    def get_max_value(self,state):
        value = self.state_to_int(state)
        return max(self.q_function[value])

    def get_max_value_action(self, state):
        value = self.state_to_int(state)
        action = np.argmax(self.q_function[value])
        return action

    def set_q_function(self, state, action, q):
        value = self.state_to_int(state)
        self.q_function[value][action] = q


model = QModel()
memory = ReplayBuffer(10000)


def select_action(state):
    # given state, selection action,
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    # action with max score
    return model.get_max_value_action(state)



def optimize_model(action):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)[0]
    state = transitions.state
    next_state = transitions.next_state
    reward = transitions.reward

    value = reward
    if next_state:
        value += GAMMA * model.get_max_value(next_state)

    model.set_q_function(state, action, value)


def train_model(path='model'):
    # initialize the game
    s = game.GameState()
    do_nothing = 0

    image_data, reward, terminate, (x, y) , up, red, _, pedes = s.frame_step(do_nothing)
    index = time.time()
    cur_time = time.time()
    while cur_time - index < 300:
        
        state = get_state(image_data, x, y, up, red, pedes)
        
        for t in count():
            # Select and perform an action
            action = select_action(state)

            _, reward, done, (x, y), up, red, _ , pedes= s.frame_step(action)


            # Observe new state
            
            current_state = get_state(image_data, x, y, up, red, pedes)
            # print (last_screen.size(), current_screen.size())

            if not done:
                next_state = current_state
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward, done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            #optimize_model(action)

            if done:
                break

    	cur_time = time.time()

    # save model
    global final_policy
    final_policy = model

    print (model.q_function)

    test_game()


def test_game(path=False):

    if not path:
        policy = final_policy
    else:
        policy = QModel()
        from numpy import genfromtxt
        policy.q_function = genfromtxt('model.csv', delimiter=',')
    new_s = game.GameState()
    start = time.time()
    reward = 0
    speed = 0.0
    t = 0
    image_data, reward, terminate, (x, y) , up, red, _ , pedes= new_s.frame_step(0)

    while not terminate:

        state = get_state(image_data, x, y, up, red, pedes)
        
        action = policy.get_max_value_action(state)

        image_data, r, terminate, (x, y) , up, red, sp, pedes = new_s.frame_step(action)

        reward += r

        speed += sp

        t += 1

    cur = time.time()

    print('The game last for {} frames'.format(t))
    print('The game last for {} second'.format(cur-start))
    print('The total award : {}'.format(reward))
    print('The average speed is : {}'.format(speed/t))
    if not path:
        np.savetxt("model.csv", final_policy.q_function , delimiter=",")


def test_simulator(t_max):

    t = 0
    while t < t_max:
        image_data , reward , terminate , (x, y), _ , _, _ , _= s.frame_step(0)
        
        t += 1


if __name__ == "__main__":
    if sys.argv[1] == 'train':
      
        train_model()
    elif sys.argv[1] == 'test':
        test_simulator(2000)
    elif sys.argv[1] == 'play':
        test_game(path=True)
    else:
        sys.exit("Wrong command")
