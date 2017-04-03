#!/usr/bin/env python
from __future__ import print_function

import sys
sys.path.append("game/")
import deep_traffic as game
import random
import numpy as np


GAME = 'Deep Traffic' # the name of the game being played for log files
ACTIONS = 5 # number of valid actions


def test_simulator(t_max):
	s = game.GameState()
	do_nothing = np.zeros(ACTIONS)
	do_nothing[0] = 1
	
	t = 0
	while t < t_max:
		a , b , c = s.frame_step(do_nothing)
		t = t + 1

if __name__ == "__main__":
	test_simulator(1000)
