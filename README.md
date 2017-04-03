# Deep traffic, simulator for reinforcement learning assignment

To get all required python packages

    pip install -r requirements.txt

To run the simulation

    python q_learning.py

The simulator is in game/, to build a simulator:

    import sys
    sys.path.append("PATH_TO_GAME_FOLDER")
    import deep_traffic as game
    
    s = game.GameState()

To interact with the simulator:

    import numpy as np
    ACTIONS = 5
    do_nothing = np.zeros(ACTIONS)
	do_nothing[0] = 1
    image, reward, terminate, (x,y) = s.frame_step(do_nothing)

Where image is the current screen of simulator, reward is current reward, terminate indicate whether game is over, (x, y) is the location of player. The input argument is the action, which is (1,0,0,0,0) in above example, indicating do nothing. The other actions are:

    (0,1,0,0,0) : left shfit lane
    (0,0,1,0,0) : right shift lane
    (0,0,0,1,0) : speed up
    (0,0,0,0,1) : slow down

To train the model, need torch and pytorch installed, you may look reference [here](http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

To Do 
   1. Add traffic light
   2. Add people
   3. Add park
   4. Design policy

