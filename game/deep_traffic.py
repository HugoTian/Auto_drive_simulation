import sys
import random
import pygame
import pygame.surfarray as surfarray

from utils import *
from pygame.locals import *
from itertools import cycle
from white_car import WhiteCar

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Deep Traffic')

IMAGES = load()

BACKGROUND_WIDTH = IMAGES['background'].get_width()

RED_CAR_WIDTH = IMAGES['red_car'].get_width()
RED_CAR_HEIGHT = IMAGES['red_car'].get_height()

WHITE_CAR_WIDTH = IMAGES['white_car'].get_width()
WHITE_CAR_HEIGHT = IMAGES['white_car'].get_height()

ROAD_WIDTH = IMAGES['road'].get_width()
ROAD_HEIGHT = IMAGES['road'].get_height()

# Reward = namedtuple('Reward', ('reward, terminal, lane, x , acc_delta, up'))
reward_scheme = {
     True: { # up
           2 : { # lane 2
                  0: Reward(0.5, False, 2, LANE[2], 0, True), # do nothing
                  1: Reward(-1, True, 2, LANE[2], 0, True), # left
                  2: Reward(0.5, False, 3, LANE[3], 0, True), # right
                  3: Reward(0.5, False, 2, LANE[2], 1, True), # speed up
                  4: Reward(0.5, False, 2, LANE[2], -3, True) # slow down
                }, 
           3 : { # lane 3
                  0: Reward(0.5, False, 3, LANE[3], 0, True),  # do nothing
                  1: Reward(0.5, False, 2, LANE[2], 0, True), # left
                  2: Reward(-1, True, 3, LANE[3], 0, True), # right
                  3: Reward(0.5, False, 3, LANE[3], 1, True), # speed up
                  4: Reward(0.5, False, 3, LANE[3], -3, True) # slow down
                }
           }
    ,False:{
            0: { # lane 0
                  0: Reward(0.5, False, 0 , LANE[0], 0, False), # do nothing
                  1: Reward(0.5, False, 1, LANE[1], 0, False), # left
                  2: Reward(-1, True, 0, LANE[0], 0, False), # right
                  3: Reward(0.5, False, 0, LANE[0], 1, False), # speed up
                  4: Reward(0.5, False, 0, LANE[0], -3, False) # slow down
                }, 
            1:  {# lane 1
                  0: Reward(0.5, False, 1, LANE[1], 0, False), # do nothing
                  1: Reward(-1, True, 1, LANE[1], 0, False), # left
                  2: Reward(0.5, False, 0, LANE[0], 0, False), # right
                  3: Reward(0.5, False, 1, LANE[1], 1, False), # speed up
                  4: Reward(0.5, False, 1, LANE[1], -3, False) # slow down
                }
            }
}
class GameState:
    # reward policy

    # keeps alive reward = 0.5
    # die , reward = -1
    # accelarate , reward = 1
    # speed = 0, reward = -0.5

    def __init__(self):

        # some important parameter
        # car_maps : lane_info for environment
        # white_cars : {id : WhiteCar()}
        # env : The global environment, every object in simulator have the same env (See Environment Class in utils)

        self.score = 0  
        self.num_frame = 0
        # player
        self.playerx = LANE[2]
        self.playery = SCREENHEIGHT - RED_CAR_HEIGHT
        self.lane = 2

        # player velocity, 
        self.pipeVelX = 0
        self.playerVelY = 1    
        self.playerAccY = 0  
        self.playerMaxV = 10
        self.up = True

        # road 
        self.basex = BASE_SHIFT 
        
        # screen
        self.circle = False # player goes so fast, need to update entire screen

        # traffic light
        self.light1 = TrafficLight(LIGHT1_POS[0], LIGHT1_POS[1], False)
        self.light2 = TrafficLight(LIGHT2_POS[0], LIGHT2_POS[1], False)
        self.green_starts = 0
        # white car
        self.max_white_car = 7
        self.white_cars = {} # key is idx, and value is white car object

        # initialize white car and environment
        self.car_maps = {0:[], 1:[], 2:[],3:[]} # key is lane, and value is (key,y, speed) pair
        self.car_maps[self.lane].append(Car(self.max_white_car, self.playerVelY, self.lane, self.playery, self.up))

        self.env = Environment(self.car_maps, [self.light1, self.light2], None, None)
        self.init_white_car()

    def add_one_car(self, i, begin=False):
        # add one car to the environment
        # i : id of the car
        
        y, lane = random.randint(0, SCREENHEIGHT), random.randint(0,3)
        
        if lane in (0, 1):
            up = False
        else:
            up = True

        # begin : whether the car should appear at border
        if begin:
            if lane in (0,1):
                y = 0
            else:
                y = SCREENHEIGHT

        # avoid collision with other initialization
        initial_collision = True
        while initial_collision :
            if self.car_maps[lane]:
                initial_collision = any( check_collision(lane, y, lane, car.y) for car in self.car_maps[lane])
                if initial_collision:
                    if begin: # if its begin state then just abort
                        return None 
                    y =  random.randint(0, SCREENHEIGHT)
            else:
                initial_collision = False

        

        speed = random.randint(1,5)
        new_car = Car(i,speed, lane , y,  up)

        # update the car_maps
        self.car_maps[lane].append(new_car)
       
        # create the WhiteCar object and store it in dict
        car = WhiteCar(i, y, lane, self.env, speed=speed)
        self.white_cars[i] = car
        return new_car

    def init_white_car(self):

        # initialize white car
        self.white_cars = {}
        for i in range(self.max_white_car):
            _ = self.add_one_car(i)
           
        # update the environment
        self.env.set_lane_info(self.car_maps)

        # update env in white car
        for i in range(self.max_white_car):
            self.white_cars[i].update_env(self.env)

    def update_player(self, actions):
        # player's movement
        # actions == 0: keep strait
        # actions == 1: turn left
        # actions == 2: tuen right
        # actions == 3: speed up
        # actions == 4: slow down
        
        reward = 0.5
        terminal = False

        # Reward = namedtuple('Reward', ('reward, terminal, lane, x , acc_delta, up'))
        # get reward by check the reward_scheme

        reward_info = reward_scheme[self.up][self.lane][actions]
        reward, terminal,  = reward_info.reward, reward_info.terminal
        self.lane, self.playerx  = reward_info.lane, reward_info.x
        self.playerAccY, self.up = self.playerAccY + reward_info.acc_delta, reward_info.up
 
        # summary

        self.playerVelY += self.playerAccY

        if self.playerVelY < 0:
            self.playerVelY = 0

        if self.playerVelY > self.playerMaxV:
            self.playerVelY = self.playerMaxV

        self.playery -= self.playerVelY
        
        if self.playery < 0 and self.up :
            self.circle = True

        # A hack, try to encourage speed up
        if not terminal and self.playerVelY == 0:
            reward = -0.5

        return reward, terminal

    def update_white_car(self):

        # update the status of white car
        r = {}
        for elem in self.white_cars:
            removed, idx = self.white_cars[elem].update()
            if removed:
                r[idx] = True

        # update system
        for idx in r:
            lane = self.white_cars[idx].lane
            self.white_cars.pop(idx)

            # update car maps
            for index, car in enumerate(self.car_maps[lane]):
        
                if idx == car.idx:
                    self.car_maps[lane].pop(index)
                    break

            # generate new cars 
            if random.randint(0,1) == 1 or len(self.white_cars) < 3:
                _  = self.add_one_car(idx, begin=True)


    def draw_parking(self):    
        # draw 4 parking plot
        SCREEN.blit(IMAGES['park'], (50,100))
        SCREEN.blit(IMAGES['park'], (50,630))
        SCREEN.blit(IMAGES['park'], (450,100))
        SCREEN.blit(IMAGES['park'], (450,630))

    def update_traffic_light(self):

        if self.num_frame - self.green_starts == 3 * LIGHT_INTERVAL:
            self.light1 = TrafficLight(LIGHT1_POS[0], LIGHT1_POS[1], True)
            self.light2 = TrafficLight(LIGHT2_POS[0], LIGHT2_POS[1], True)

        if self.num_frame - self.green_starts == 4 * LIGHT_INTERVAL:
            self.light1 = TrafficLight(LIGHT1_POS[0], LIGHT1_POS[1], False)
            self.light2 = TrafficLight(LIGHT2_POS[0], LIGHT2_POS[1], False)
            self.green_starts = self.num_frame

        self.env.set_traffic_info([self.light1, self.light2])
 
    def draw_traffic_light(self):

        if self.light1.red:
            SCREEN.blit(IMAGES['red_light'], LIGHT1_POS)
        else:
            SCREEN.blit(IMAGES['green_light'], LIGHT1_POS)

        if self.light2.red:
            SCREEN.blit(IMAGES['red_light'], LIGHT2_POS)
        else:
            SCREEN.blit(IMAGES['green_light'], LIGHT2_POS)

    def check_crash(self):
        # check crash with white car
        
        terminal = False
        for elem in self.white_cars:
            x,y = self.white_cars[elem].getXY()
            if check_collision(self.playerx, self.playery, x, y):
                terminal = True
                break

        return terminal

    def update_global_env(self):
         # reform global car map
        self.car_maps = {0:[], 1:[], 2:[],3:[]} 
        self.car_maps[self.lane].append(Car(self.max_white_car, self.playerVelY, self.lane, self.playery, self.up))


        for elem in self.white_cars:
            x, y = self.white_cars[elem].getXY()
            lane = self.white_cars[elem].getLane()
            speed = self.white_cars[elem].getSpeed()
            up = self.white_cars[elem].isUp()
            self.car_maps[lane].append(Car(elem, speed, lane, y, up))

        # update environment
        self.env.set_lane_info(self.car_maps)

        # update environment for every white car
        for elem in self.white_cars:
            self.white_cars[elem].update_env(self.env)

    def frame_step(self, input_actions):
        # update frame number 
        self.num_frame += 1

        pygame.event.pump()


        if input_actions < 0 or input_actions > 5 :
            raise ValueError('Not a valid operation')

        
        # update white car
        self.update_white_car()

        

        # update player
        reward, terminal1 = self.update_player(input_actions)

        # check if crash here
        terminal2 = self.check_crash()

        # get entire new screen
        if self.circle:
            self.playery = SCREENHEIGHT - RED_CAR_HEIGHT
            self.circle = False
            self.init_white_car()

        # update global info and each cars local info
        self.update_global_env()
        
        terminal = terminal1 or terminal2
        
        # handle termianl case
        if terminal:
            # make sure reward is -1
            reward = -1
            self.__init__()

        # draw 
        SCREEN.blit(IMAGES['background'], (0,0))


        SCREEN.blit(IMAGES['road'], (self.basex, 0))

        
        SCREEN.blit(IMAGES['red_car'], (self.playerx, self.playery))

        for elem in self.white_cars:
            x,y = self.white_cars[elem].getXY()
            if x <= LANE[1]:
                SCREEN.blit(IMAGES['white_car_reverse'], (x,y))
            else:
                SCREEN.blit(IMAGES['white_car'], (x,y))

        # traffic light
        self.update_traffic_light()
        self.draw_traffic_light()

        # encouge speed up
        if not terminal and self.playerVelY == 0:
            reward = -0.5
        
        # update score
        if reward >= 0:
            score_update = int(reward*2)
            self.score += score_update
        showScore(int(self.score/50))
        
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        
        return image_data, reward, terminal, (self.playerx, self.playery)


def showScore(score):
    """displays score in screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth)

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset,0))
        Xoffset += IMAGES['numbers'][digit].get_width()

def check_collision(x1,y1,x2,y2):
    # check whther 2 car collide with each other
    if x1 != x2:
        return False

    if abs(y1-y2) <= 20 + RED_CAR_HEIGHT:
        return True
    else:
        return False


