# Deep traffic, simulator for reinforcement learning assignment

To get required python packages on MacOS

    pip install -r requirements.txt

If you use other platform or the above command fails, basically you need to install pygame, pytorch, torchvision, opencv to train the network, please refer to [here](http://pytorch.org/) to install pytorch and torchvision.


To run the simulation

    python q_learning.py test

To train the model

    python q_learning.py train PATH_TO_SAVE_MODEL

To load a pretrained_model, and let the player play

    python q_learning.py play PATH_TO_MODEL

The simulator is in game/, to build a simulator:

    import sys
    sys.path.append("PATH_TO_GAME_FOLDER")
    import deep_traffic as game
    
    s = game.GameState()

To interact with the simulator:

    do_nothing = 0
    image, reward, terminate, (x,y) = s.frame_step(do_nothing)

Where image is the current screen of simulator, reward is current reward, terminate indicate whether game is over, (x, y) is the location of player. The input argument is the action, which is 0 in above example, indicating do nothing. The other actions are:

    1 : left shfit lane
    2 : right shift lane
    3 : speed up
    4 : slow down

To train the model, need torch and pytorch installed, you may look reference [here](http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

The AI of the game is White car, they will avoid crash and speed up in a smart way. They may also park or restart from parking. When parking, the AI car becomes yellow. In the environment, there is also traffic light and pedestrian. Everyone should obey traffic light, and pedestrian can across the road at any time, any where. Every car should not hurt pedestrian.

To Do 
   1. Design reward system to make better player
   2. Make simulator stable
   3. Reduce library dependency
   
   

