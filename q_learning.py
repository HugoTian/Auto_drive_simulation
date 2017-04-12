#!/usr/bin/env python
from __future__ import print_function
import game.deep_traffic as game
from game.utils import *
import numpy as np
import random
# game constant
GAME = 'Deep Traffic' # the name of the game being played for log files
ACTIONS = 5 # number of valid actions

#  For training the model
BATCH_SIZE = 32
GAMMA = 0.999

s = game.GameState()

# defines the reward/connection graph
r = np.array([[0, 0, 0, 0],
              [-1, -1, -1, -1],
              [0, -1, 100, 0],
              [-1, -1, -1, -1],
              [0, 0, 0, 0],
              [-1, -1, -1, -1],
              [100, -1, -1, 0],
              [-1, -1, -1, -1],
              [0, 0, 0, 0],
              [-1, -1, -1, -1],
              [0, -1, -1, 100],
              [-1, -1, -1, -1],
              [-1, 0, 100, 0],
              [-1, -1, -1, -1],
              [-1, -1, 100, 0],
              [-1, -1, -1, -1],
              [-1, 100, -1, 0],
              [-1, -1, -1, -1],
              [-1, -1, -1, 100],
              [-1, -1, -1, -1],
              [-1, 0, -1, 100],
              [-1, -1, -1, -1],
              [-1, -1, -1, 100],
              [-1, -1, -1, -1]]).astype("float32")

q = np.zeros_like(r)

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
        150: (150, 190, 120),
        240: (240, 280, 120),
        330: (330, 370, -60),
        400: (400, 440, -60)
    }
}


def update_q(state, next_state, action, alpha, gamma):
    rsa = r[state, action]
    qsa = q[state, action]
    new_q = qsa + alpha * (rsa + gamma * max(q[next_state, :]) - qsa)
    q[state, action] = new_q
    # renormalize row to be between 0 and 1
    rn = q[state][q[state] > 0] / np.sum(q[state][q[state] > 0])
    q[state][q[state] > 0] = rn
    return r[state, action]


def generate_random_action():
    return random.randint(1,4)


def check_state(target, img, x, y, up, red):
    #check state

    x1, x2, y_delta = state_table[target][x]

    if not x1 :
        return 1
    y_up = max(0, y+y_delta)
    y_down = min(800, y+y_delta)

    if up:
        if y_up>y:
            roi = np.array(img[x1:x2, y-20:y_up+20])
        else:
            roi = np.array(img[x1:x2, y_up:y])
        overall = roi.sum()

        #left and  right case
        if target != 'front':
            if overall == 0:
                return 0
            else:
                return 1
        else:
            if y > RED_STOP_UP and y - 20 < RED_STOP_UP and red:
                return 2
            elif overall == 0:
                return 0
            else:
                return 1
    else:
        if y_delta != 120:
            roi = np.array(img[x1:x2, y-20:y_down+20])
        else:
            roi = np.array(img[x1:x2, y + 60:y_down])
        overall = roi.sum()

        #left and  right case
        if target != 'front':
            if overall == 0:
                return 0
            else:
                return 1
        else:
            if y + 60 < RED_STOP_DOWN and y + 80 > RED_STOP_DOWN and red:
                return 2
            elif overall == 0:
                return 0
            else:
                return 1


def env_update(action):
    # return
    # left : left lane car or not ?
    # right : right lane car or not
    # front : 0 or 1 or 2, 0 : nothing , 1: car, 2:red light
    # terminate

    image_data , reward , terminate , (x, y), up, red, _, _ = s.frame_step(action)
    left = check_state('left', image_data, x, y, up, red)
    right = check_state('right', image_data,x, y, up, red)
    front = check_state('front', image_data,x, y, up, red)

    if terminate:
        terminate = 1
    else:
        terminate = 0

    return left, front, right, terminate

def state(left, front, right, terminate):

    #convert four signals to state
    s = 12*left + 4*front + 2*right + terminate
    return s

gamma = 0.8
alpha = 1.

def test_simulator(t_max):
    # only funciton you nedd to modify
    t = 0
    left, front, right, terminate = env_update(4)
    current_state = state(left, front, right, terminate)

    while t < t_max:


        if np.sum(q[current_state]) > 0:
            action = np.argmax(q[current_state])+1
            left, front, right, terminate = env_update(action)
            next_state = state(left, front, right, terminate)
            update_q(current_state, next_state, action-1, alpha=alpha, gamma=gamma)

        else:
            action = generate_random_action()
            while r[current_state,action-1] < 0:
                action = generate_random_action()
            left, front, right, terminate = env_update(action)
            next_state = state(left, front, right, terminate)
            update_q(current_state, next_state, action-1, alpha=alpha, gamma=gamma)

        if terminate == 0:
            current_state = next_state
        else:
            left, front, right, terminate = env_update(4)
            current_state = state(left, front, right, terminate)

        print(q)

        t += 1




if __name__ == "__main__":
    
    test_simulator(10000)
