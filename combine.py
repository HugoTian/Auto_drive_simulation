#!/usr/bin/env python
from __future__ import print_function

import sys
import math
import game.deep_traffic as game

from itertools import count
from game.utils import  *

import numpy as np
import time
import random

from q_learning import QModel
from deep_q_learning import DQN

# game constant
GAME = 'Deep Traffic' # the name of the game being played for log files
ACTIONS = 5 # number of valid actions


def combine_model():
    pass


def select_action():
    pass
