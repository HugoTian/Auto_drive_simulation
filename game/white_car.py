from .utils import *
import random

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

        self.park = False
        self.park_time = 0
        self.park_random = random.randint(200,400)

        self.wait_for_pedes = False

        self.change_lane_map = {0:1, 1:0, 2:3, 3:2} # possible ways to change lane
        self.up_dict = {True: {
                                'speed': -1,
                                'light': 1,
                                'stop': RED_STOP_UP,
                                'park':3

                              },
                       False: {
                                'speed': 1,
                                'light': 0,
                                'stop': RED_STOP_DOWN,
                                'park': 0
                               }
                       }

    def update_env(self, env):

        self.env = env

    def direct_update(self, y):
        # directly set y value
        self.y += y

    def update(self):
        # smart update of white car
        removed = False

        # update y
        y_delta = self.up_dict[self.up]['speed'] * self.speed
        self.y += y_delta

        # check out of bound
        if self.y < 0 or self.y > SCREENHEIGHT:
            removed = True
            return removed, self.key

        # handle traffic
        self.handle_traffic_light()

        # handle pedestrian
        self.handle_pedestrian()

        # handle crash
        crash, crashed_car, back = self.check_crash_white_car(self.lane)

        if crash and back:
            self.handle_crash(crashed_car)
            # debug print(self.key, self.lane, self.speed, crashed_car.speed, crashed_car.lane , crashed_car.idx)
        elif not self.park and not self.wait_for_pedes:
            # make it fun
            if random.randint(0, 5) == 0:
                self.try_speed_up()

        # try change lane when in green light, avoid forever blocking by parked car
        if not self.park and not self.wait_for_pedes and self.speed == 0 and not self.env.get_traffic()[self.up_dict[self.up]['light']].red:
            self.try_change_lane()
        
        # park ?
        self.park_or_not()

        return removed, self.key

    def park_or_not(self):
        # decide whether to park
        if self.park:
            self.park_time += 1
        else:
            self.park_time = 0

        if self.lane == self.up_dict[self.up]['park'] and (self.y > RED_STOP_UP or self.y < RED_STOP_DOWN):
            if random.randint(0,self.park_random) == 0:
                # park
                self.park = True
                self.speed = 0

        if self.park_time > self.park_random:
            self.park = False
            self.park_time = 0

    def handle_crash(self, crashed_car):
        # if there is a possible crash, need to handle that
        

        new_lane = self.change_lane_map[self.lane]
        crash, _, _ = self.check_crash_white_car(new_lane)

        if not crash:
            self.lane = new_lane
            self.x = LANE[self.lane]
        else:
            self.speed = crashed_car.speed

    def hit_pedes(self):

        hit = False
        people = self.env.get_pedestrian()
        if not people:
            return hit

        x1, y1 = people.x , people.y

        if self.up:
            if x1 < PEDES_RIGHT and x1 > PEDES_LEFT and abs(self.y - y1) < 65:
                hit = True
        else:
            if x1 < PEDES_RIGHT and x1 > PEDES_LEFT and abs(self.y - y1) < 65:
                hit = True

        return hit

    def handle_pedestrian(self):
        # handle pedestrian

        if self.hit_pedes():
            self.wait_for_pedes = True
            self.speed = 0
        else:
            self.wait_for_pedes = False


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
        from .deep_traffic import check_collision
        crash = False
        crashed_car = None
        back = None

        car_list = self.get_cars_in_lane(lane)
        for car in car_list:
            
            crash = check_collision(lane, self.y, lane, car.y)
            if crash:
                crashed_car = car

                if self.up:
                    back = bool(self.y > car.y)
                else:
                    back = bool(self.y < car.y)

                return crash, crashed_car, back
        
        return crash, crashed_car, back


    def try_speed_up(self):

        old_speed = self.speed
        old_y = self.y

        self.speed = self.speed + 1
        if self.speed > 5:
            self.speed = 5

        self.y += self.up_dict[self.up]['speed']

        crash, _, _ = self.check_crash_white_car(self.lane)

        hit = self.hit_pedes()

        if crash or hit:
            self.speed = old_speed
        self.y = old_y

    def try_change_lane(self):
        # try change lane to avoid forever blocked by parking car
        new_lane = self.change_lane_map[self.lane]
        crash, _, _ = self.check_crash_white_car(new_lane)

        if not crash:
            self.lane = new_lane
            self.x = LANE[self.lane]
        self.try_speed_up()

    def handle_traffic_light(self):

        # get the traffic light
        traffic_light = self.env.get_traffic()
        red = traffic_light[self.up_dict[self.up]['light']].red

        # stop
        if red:
            if self.up and self.y - self.up_dict[self.up]['stop'] < 5:
                self.y = self.up_dict[self.up]['stop']
                self.speed = 0

            if not self.up and self.up_dict[self.up]['stop'] - self.y < 5:
                self.y = self.up_dict[self.up]['stop']
                self.speed = 0

        # if not stop
        if not self.park and not red and self.speed == 0:
            self.try_speed_up()


    def getXY(self):
        return (self.x, self.y)

    def getLane(self):
        return self.lane

    def isUp(self):
        return self.up

    def getSpeed(self):
        return self.speed

    def getPark(self):
        return self.park
