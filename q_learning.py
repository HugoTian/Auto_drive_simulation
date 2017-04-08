#!/usr/bin/env python
from __future__ import print_function

import sys
import math
import game.deep_traffic as game
from itertools import count
import numpy as np
from game.utils import *
import time
import random

try:
    import cv2
    from PIL import Image
    # need to install torch
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.autograd as autograd
    import torch.nn.functional as F
    import torchvision.transforms as T
except:
    if sys.argv[1] != 'test':
        raise Exception('Need have torch, PIL, opencv2 to train the model and play game')

# game constant
GAME = 'Deep Traffic' # the name of the game being played for log files
ACTIONS = 5 # number of valid actions

#  For training the model
BATCH_SIZE = 128
GAMMA = 0.999
# region of interest, crop screen to find ROI
ROI_WIDTH = 200
ROI_HEIGHT = 100
CONV_SIZE = 80

# learning parameter
EPS_START = 0.9
EPS_END = 0.02
EPS_DECAY = 200

try:
    USE_CUDA = torch.cuda.is_available()
except:
    USE_CUDA = False

# named tuple to record state transitions
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminate'))

# use pytorch build in model to convert image to CONV_SIZE  * CONV_SIZE
try:
    resize = T.Compose([T.ToPILImage(),
                    T.Scale(CONV_SIZE, interpolation=Image.CUBIC),
                    T.ToTensor()])
except:
    pass


def Variable(data, *args, **kwargs):
 
    # torch variable class
    
    if USE_CUDA:
    	return autograd.Variable(data, *args, **kwargs).cuda()
    else:
    	return autograd.Variable(data, *args, **kwargs)    

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


class DQN(nn.Module):
    # the network to train and test the model
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, ACTIONS)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# global variable
steps_done = 0

model = DQN()
memory = ReplayBuffer(10000)
optimizer = optim.RMSprop(model.parameters())

if USE_CUDA:
    model.cuda()


def get_roi(x, y):
    # get the region of interests
    screen = pygame.surfarray.array3d(pygame.display.get_surface())
    
    # hard code
    # need change when cars can go down
    y_min = max(0 , y - 100)
    y_max = min(SCREENHEIGHT, y + 120)
    screen = screen[300:550, y_min:y_max]
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
        state = Variable(state, volatile=True)
        return model(state).data.max(1)[1].cpu()
        
    else:
        # some time just random action
        return torch.LongTensor([[random.randrange(ACTIONS)]])


def optimize_model():
    # need to sample more
    if len(memory) < BATCH_SIZE:
        return
    # create batch
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

    if USE_CUDA:
        non_final_mask = non_final_mask.cuda()

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)

    state_batch = Variable(torch.cat(batch.state))
    reward_batch = Variable(torch.cat(batch.reward))

    action_batch = Variable(torch.cat(batch.action))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE), volatile=False)
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    # next_state_values.volatile = False
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


def train_model(path='model'):
    # initialize the game
    s = game.GameState()
    do_nothing = 0

    image_data, reward, terminate, (x, y) , _ = s.frame_step(do_nothing)
    index = time.time()
    cur_time = time.time()
    while cur_time - index < 1200:
        last_screen = get_roi(x, y)
        current_screen = get_roi(x, y)
        
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(state)

            _, reward, done, (x, y), _ = s.frame_step(action[0][0])

            reward = torch.Tensor([reward])

            # Observe new state
            last_screen = current_screen
            current_screen = get_roi(x,y)
            # print (last_screen.size(), current_screen.size())

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

    	cur_time = time.time()

    with open(path, 'w') as f:
        torch.save(model.state_dict(), f)


def test_simulator(t_max):
    s = game.GameState()
    t = 0
    while t < t_max:

        action = random.randint(0,3)
        if random.randint(0, 20) == 0:
            action = 4
        image_data , reward , terminate , (x, y) = s.frame_step(0)
        
        t += 1

 
def load_model(path):

    trained_model = DQN()
    trained_model.load_state_dict(torch.load(path))

    if USE_CUDA:
        trained_model.cuda()

    s = game.GameState()
    image_data , reward , terminate , (x, y), _ = s.frame_step(0)

    last_screen = get_roi(x,y)
    start = time.time()
    total_reward = 0
    speed = 0
    frames = 0
    while not terminate:
        current_screen = get_roi(x,y)
        state = current_screen - last_screen
        
        # Select and perform an action
        action = trained_model(Variable(state)).data.max(1)[1].cpu()
        _, reward, terminate, (x,y) , v = s.frame_step(action[0][0])
        
        last_screen = current_screen
        speed += v
        frames += 1
        total_reward += reward

    cur_time = time.time()
    
    print('The game last for {} seconds'.format(cur_time-start))
    print('The game last for {} frames'.format(frames))
    print('The average speed is {}'.format(speed/frames))

if __name__ == "__main__":
    if sys.argv[1] == 'train':
        if len(sys.argv) > 2:
            train_model(path=sys.argv[2])
        else:
            train_model()
            
    elif sys.argv[1] == 'play':
        load_model(sys.argv[2])
    elif sys.argv[1] == 'test':
        test_simulator(2000)
    else:
        sys.exit("Wrong command")
