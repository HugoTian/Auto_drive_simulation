import numpy as np
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

class GameState:
    # reward policy

    # keeps alive reward = 0.5
    # die , reward = -1
    # accelarate , reward = 1

    def __init__(self):

        self.score = 0  
        
        # player
        self.playerx = LANE[2]
        self.playery = SCREENHEIGHT - RED_CAR_HEIGHT
        self.lane = 2

        # player velocity, 
        self.pipeVelX = 0
        self.playerVelY = 1    
        self.playerAccY = 0  
        self.playerMaxV = 10
        # road 
        self.basex = BASE_SHIFT 
        
        # screen
        self.circle = False # player goes so fast, need to update entire screen

        # white car
        self.max_white_car = 7
        self.white_cars = {} # key is idx, and value is white car object

        # initialize white car
        self.car_maps = {0:[], 1:[], 2:[],3:[]} # key is lane, and value is (key,y, speed) pair
        self.car_maps[self.lane].append((self.max_white_car, self.playery, self.playerVelY))
        self.init_white_car()

    def add_one_car(self, i, begin=False):
        # i : id of the car
        
        y, lane = random.randint(0, SCREENHEIGHT), random.randint(0,3)
        
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
                initial_collision = any( check_collision(lane, y, lane, y_axis) for _, y_axis, _ in self.car_maps[lane])
                if initial_collision:
                    if begin: # if its begin statem them just abort
                        return None, None
                    y =  random.randint(0, SCREENHEIGHT)
            else:
                initial_collision = False

        

        speed = random.randint(1,5)
        self.car_maps[lane].append((i,y,speed))
        car = WhiteCar(i, y, lane, self.car_maps, speed=speed)
        self.white_cars[i] = car
        return y, lane

    def init_white_car(self):
        # initialize white car
        self.white_cars = {}
        for i in range(self.max_white_car):
            y,lane = self.add_one_car(i)
                 

    def update_player(self, actions):
        # player's movement
        # input_actions[0] == 1: keep strait
        # input_actions[1] == 1: turn left
        # input_actions[2] == 1: tuen right
        # input_actions[3] == 1: speed up
        # input_actions[4] == 1: slow down
        
        reward = 0.5
        terminal = False

        # turn left
        if actions == 1:
            if self.lane == 0:
                self.lane = 1 
                self.playerx = LANE[self.lane] 
            elif self.lane == 1:
                reward = -1
                terminal = True
            elif self.lane == 2:
                reward = -1
                terminal = True
            else:
                self.lane  = 2
                self.playerx = LANE[self.lane] 

        # turn right
        if actions == 2:
            if self.lane == 0:
                reward = -1
                terminal = True
                
            elif self.lane == 1:
                self.lane = 0
                self.playerx = LANE[self.lane]

            elif self.lane == 2:
                self.lane = 3
                self.playerx = LANE[self.lane]
            else: 
                reward = -1
                terminal = True
        
        # speed up
        if actions == 3:
            self.playerAccY = 1
            reward = 1

        # slow down
        if actions == 4:
            self.playerAccY = -2

        # summary

        self.playerVelY += self.playerAccY

        if self.playerVelY < 0:
            self.playerVelY = 0

        if self.playerVelY > self.playerMaxV:
            self.playerVelY = self.playerMaxV

        self.playery -= self.playerVelY
        
        if self.playery < 0:
            self.circle = True

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
            for index, pair in enumerate(self.car_maps[lane]):
                key, _ , _ = pair
                if key == idx:
                    self.car_maps[lane].pop(index)
                    break

            # generate new cars 
            if random.randint(0,1) == 1 or len(self.white_cars) < 3:
                _ , lane = self.add_one_car(idx, begin=True)


            # update car maps for all other white cars
            for elem in self.white_cars:
                self.white_cars[elem].update_car_map(self.car_maps)

    def draw_parking(self):    
        # draw 4 parking plot
        SCREEN.blit(IMAGES['park'], (50,100))
        SCREEN.blit(IMAGES['park'], (50,630))
        SCREEN.blit(IMAGES['park'], (450,100))
        SCREEN.blit(IMAGES['park'], (450,630))

    def check_crash(self):
        # check crash with white car
        reward = 0.5
        terminal = False
        for elem in self.white_cars:
            x,y = self.white_cars[elem].getXY()
            if check_collision(self.playerx, self.playery, x, y):
                reward = -1
                terminal = True
                break

        return reward, terminal

    def frame_step(self, input_actions):
        pygame.event.pump()

        

        if input_actions < 0 or input_actions > 5 :
            raise ValueError('Not a valid operatio')

        # update player
        reward, terminal = self.update_player(input_actions)

        # update white car
        self.update_white_car()

        # check if crash here
        reward, terminal = self.check_crash()
       
        # get entire new screen
        if self.circle:
            self.playery = SCREENHEIGHT - RED_CAR_HEIGHT
            self.circle = False
            self.init_white_car()

        # reform global car map
        self.car_maps = {0:[], 1:[], 2:[],3:[]} # key is lane, and value is (key,y, speed) pair
        self.car_maps[self.lane].append((self.max_white_car, self.playery, self.playerVelY))

        for elem in self.white_cars:
            x, y = self.white_cars[elem].getXY()
            lane = self.white_cars[elem].getLane()
            speed = self.white_cars[elem].getSpeed()
            self.car_maps[lane].append((elem, y, speed))

        for elem in self.white_cars:
            self.white_cars[elem].update_car_map(self.car_maps)
        
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

        #self.draw_parking()
                
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


