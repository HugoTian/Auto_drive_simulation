import sys
import random
import pygame
from utils import *
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle


class WhiteCar:

    def __init__(self, key, y, lane, env , speed=1):
        
        self.key = key # the key in game state white_cars dictionary
        self.y = y
        if lane in (0,1):
            self.up = False
        else:
            self.up = True

        self.lane = lane
        self.x = LANE[lane]
        self.speed = speed
        self.env = env

    def update_env(self, env):
        self.env = env

    def direct_update(self, y):
        # directly set y value
        self.y += y

    def update(self):
        # smart update of white car
        removed = False

        if not self.up:
            # going down
            self.y = self.y + self.speed

            # check out of bound
            if self.y >  SCREENHEIGHT :
                removed = True
                return removed, self.key
            # check same lane collision
            crash, crashed_car , back = self.check_crash_white_car(self.lane)

            if crash and back:
                self.handle_crash(crashed_car)
        else:

            self.y = self.y - self.speed

            # check out of bound
            if self.y < 0:
                removed = True
                return removed, self.key

            # same lane crash
            crash , crashed_car, back = self.check_crash_white_car(self.lane)

            if crash and back:
                # if crash change lane or slow down 
                self.handle_crash(crashed_car)

        return removed, self.key

    def handle_crash(self, crashed_car):
        # if there is a possible crash, need to handle that
        change_lane_map = {0:1, 1:0, 2:3, 3:2} # possible ways to change lane

        new_lane = change_lane_map[self.lane]
        crash, _, _ = self.check_crash_white_car(new_lane)

        if not crash:
            self.lane = new_lane
            self.x = LANE[self.lane]
        else:
            self.speed = crashed_car.speed


    def get_cars_in_lane(self, lane):
        # get all the cars in lane except for myself
        car_list = []
        for elem in self.env.get_cars(lane):
            if elem.idx != self.key:
                car_list.append(elem) 

        return car_list

    def check_crash_white_car(self, lane):

        # check whether crash with  cars on lane
        # return crash, crashed_car_id, whether I am at back(slow down or change_lane)
        from deep_traffic import check_collision
        crash = False
        crashed_car = None
        back = None

        car_list = self.get_cars_in_lane(lane)
        for car in car_list:
            
            crash = check_collision(lane, self.y, lane, car.y)
            if crash:
                crashed_car = car

                if self.up:
                    back = self.y > car.y
                else:
                    back = self.y < car.y

                return crash, crashed_car, back
        
        return crash, crashed_car, back


    def getXY(self):
        return (self.x, self.y)

    def getLane(self):
        return self.lane

    def isUp(self):
        return self.up

    def getSpeed(self):
        return self.speed
