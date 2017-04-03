import numpy as np
import sys
import random
import pygame
from utils import *
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle


class WhiteCar:

    def __init__(self, key, y, lane,all_white_car, speed=1):
        
        self.key = key # the key in game state white_cars dictionary
        self.y = y
        if lane in (0,1):
            self.up = False
        else:
            self.up = True

        self.lane = lane
        self.x = LANE[lane]
        self.others = all_white_car
        self.speed = speed

    def update_car_map(self, car_map_elem):
        self.others = car_map_elem

    def update(self):
        # smart update of white car
        removed = False

        if not self.up:
            self.y = self.y + self.speed

            # check out of bound
            if self.y >  SCREENHEIGHT :
                removed = True
                return removed, self.key
            # check same lane collision
            crash, crashed_car = self.check_crash_white_car(self.lane)

            # only car at back need to do something
            back = False
            if crash:
                for idx, y, s in self.others[self.lane]:
                        if crashed_car == idx:
                            crashed_y = y 
                back = self.y < crashed_y

            if crash and back:
                # if crash change lane or slow down 
                change_lane_fail = False
                if self.lane == 0:
                    change_lane_fail, _ = self.check_crash_white_car(1)
                    new_lane = 1
                else:
                    change_lane_fail , _ = self.check_crash_white_car(0)
                    new_lane = 0

                if not change_lane_fail:
                    self.lane = new_lane
                    self.x = LANE[self.lane]
                else:
                    for idx, y, s in self.others[self.lane]:
                        if crashed_car == idx:
                            self.speed = s 
        else:

            self.y = self.y - self.speed

            # check out of bound
            if self.y < 0:
                removed = True
                return removed, self.key

            # same lane crash
            crash , crashed_car= self.check_crash_white_car(self.lane)

            back = False
            if crash:
                for idx, y, s in self.others[self.lane]:
                        if crashed_car == idx:
                            crashed_y = y 
                back = self.y > crashed_y

            if crash and back:
                # if crash change lane or slow down 
                change_lane_fail = False
                if self.lane == 2:
                    change_lane_fail, _ = self.check_crash_white_car(3)
                    new_lane = 3
                else:
                    change_lane_fail , _ = self.check_crash_white_car(2)
                    new_lane = 2

                if not change_lane_fail:
                    self.lane = new_lane
                    self.x = LANE[self.lane]
                else:
                    # change lane fail , slow down
                    for idx, y, s in self.others[self.lane]:
                        if crashed_car == idx:
                            self.speed = s

        return removed, self.key

    def check_crash_white_car(self, lane):

        # check whether crash with white cars on lane
        from deep_traffic import check_collision
        crash = False
        crashed_car = None
        car_list = self.others[lane]
        if len(car_list) > 1:
            for idx, y, _ in car_list:
                if idx != self.key:
                    crash = check_collision(lane, self.y, lane,y)
                    if crash:
                        crashed_car = idx
                        return crash, idx
        return crash, crashed_car
    def getXY(self):
        return (self.x, self.y)

    def getLane(self):
        return self.lane

    def isUp(self):
        return self.up

    def getSpeed(self):
        return self.speed
