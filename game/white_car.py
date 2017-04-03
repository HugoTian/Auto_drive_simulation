import numpy as np
import sys
import random
import pygame
from utils import *
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle


class WhiteCar:

    def __init__(self, key, y, lane, all_white_car, speed=1):
        from deep_traffic import RED_CAR_HEIGHT
        self.red_car_height = RED_CAR_HEIGHT
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
        # this dictionary encapsulates all the differences between
        # going up and down to avoid redundant code
        self.up_dict = {False:{'left_lane': 0, 'right_lane':1, 
                               'speed': self.speed, 
                               'boundary_fn': self.gt_screenheight,
                               'back_crash_fn': self.down_back,
                               'begin': self.top},
                        True:{'left_lane': 2, 'right_lane':3, 
                              'speed': -self.speed, 
                              'boundary_fn': self.lt_zero, 
                              'back_crash_fn': self.up_back,
                              'begin': self.bottom}}

    
    def bottom(self):
        """ where should cars travelling up start on wrap?
        """
        return SCREENHEIGHT + self.red_car_height + self.y
        
    def top(self):
        """ where should cars travelling down start on wrap?
        """
        return self.y - SCREENHEIGHT - self.red_car_height
                          
    def update_car_map(self, car_map_elem):
        self.others = car_map_elem

    def direct_update(self, y):
        # directly set y value
        self.y += y

    def gt_screenheight(self):
        """ greater than screenheight, used to determine if cars
        travelling down should wrap """
        return self.y > SCREENHEIGHT + self.red_car_height
        
    def lt_zero(self):
        """ less than 0, used to determine if cars
        travelling up should wrap """
        return self.y < 0 - self.red_car_height
        
    def down_back(self, crashed_y):
        """ am i behind other car when travelling down
        """
        return self.y < crashed_y
    
    def up_back(self, crashed_y):
        """ am i behind other car when travelling up
        """
        return self.y > crashed_y
        
    def update(self):
        # smart update of white car
        removed = False

        self.y = self.y + self.up_dict[self.up]['speed']

        # check out of bound
        if self.up_dict[self.up]['boundary_fn']():
            print "detected out of bound, y=" + str(self.y)
            self.y = self.up_dict[self.up]['begin']()
            print "fixed by updating to y=" + str(self.y)
        # check same lane collision
        crash, crashed_car = self.check_crash_white_car(self.lane)

        # only car at back need to do something
        back = False
        if crash:
            for idx, y, s in self.others[self.lane]:
                    if crashed_car == idx:
                        crashed_y = y 
            back = self.up_dict[self.up]['back_crash_fn'](crashed_y)

        if crash and back:
            # if crash change lane or slow down 
            change_lane_fail = False
            if self.lane == self.up_dict[self.up]['left_lane']:
                change_lane_fail, _ = self.check_crash_white_car(self.up_dict[self.up]['right_lane'])
                new_lane = self.up_dict[self.up]['right_lane']
            else:
                change_lane_fail , _ = self.check_crash_white_car(self.up_dict[self.up]['left_lane'])
                new_lane = self.up_dict[self.up]['left_lane']

            if not change_lane_fail:
                self.lane = new_lane
                self.x = LANE[self.lane]
            else:
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
