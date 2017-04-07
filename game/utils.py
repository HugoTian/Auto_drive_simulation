import pygame
from collections import namedtuple

FPS = 30
SCREENWIDTH  = 600
SCREENHEIGHT = 800

# lane
LANE = {}
LANE[0] = 150
LANE[1] = 240
LANE[2] = 330
LANE[3] = 400

BASE_SHIFT = 100

# traffic light constant
LIGHT_INTERVAL = 100
LIGHT1_POS = (50, 360)
LIGHT2_POS = (500, 360)
RED_STOP_UP = 460
RED_STOP_DOWN = 300

Car = namedtuple('Car', ('idx', 'speed', 'lane', 'y', 'upwards'))
Pedestrain = namedtuple('Pedestrain', ('idx', 'speed', 'x', 'y', 'left'))
TrafficLight = namedtuple('Traffic', ('x', 'y', 'red'))
Reward = namedtuple('Reward', ('reward, terminal, lane, x, acc_delta, up'))


class Environment:
    # the environment, the location and status of every object in screen
    # every object in screen has a Environment obejct as attribute
    def __init__(self, lane_info, traffic_info, pedestrain):
        # lane_info : infomation of lane, dict like {0 : list(Car), 1 : list(Car)}
        # traffic_info : TrafficLight :  list(TrafficLight)
        # pedestrain : list(Pedestrain)
        # parking : list(Parking) , list of parking infomation
        self.lane_info = lane_info
        self.traffic_info = traffic_info
        self.pedestrain = pedestrain

    def get_cars(self, lane):

        # get cars in lane 
        return self.lane_info[lane]

    def get_pedestrain(self):

        # get list of pedestrain
        return self.pedestrain

    def get_traffic(self):

        # get the traffic light info
        return self.traffic_info

    def get_parking(self):

        # get the parking info
        return self.parking_info

    def set_lane_info(self, lane_info):
        self.lane_info = lane_info

    def set_traffic_info(self, traffic_info):
        self.traffic_info = traffic_info

    def set_pedestrain_info(self, pedestrain):
        self.pedestrain = pedestrain

    def set_parking(self, parking):
        self.parking_info = parking


def load():

    RED_CAR_PATH = 'Images/red_car.png'
    WHITE_CAR_PATH = 'Images/white_car.png'
    WHITE_CAR_REVERSE_PATH = 'Images/white_car_reverse.png'
    ROAD_PATH = 'Images/road.png'
    BACKGROUND_PATH = 'Images/background-black.png'
    PARKING_PATH = 'Images/parking.png'
    RED_TRIFFIC_LIGHT_PATH = 'Images/red.png'
    GREEN_TRAFFIC_LIGHT_PATH = 'Images/green.png'    
    WHITE_LINE_PATH = 'Images/white_line.png'
    PEDES_RIGHT_PATH = 'Images/walk_right.png'
    PEDES_LEFT_PATH = 'Images/walk_left.png'

    IMAGES = {}
    IMAGES['road'] =  pygame.image.load(ROAD_PATH).convert()
    IMAGES['red_car'] = pygame.image.load(RED_CAR_PATH).convert()
    IMAGES['white_car'] = pygame.image.load(WHITE_CAR_PATH).convert()
    IMAGES['white_car_reverse'] = pygame.image.load(WHITE_CAR_REVERSE_PATH).convert()
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()
    IMAGES['park'] = pygame.image.load(PARKING_PATH).convert()
    IMAGES['red_light'] = pygame.image.load(RED_TRIFFIC_LIGHT_PATH).convert()
    IMAGES['green_light'] = pygame.image.load(GREEN_TRAFFIC_LIGHT_PATH).convert()
    IMAGES['white_line'] = pygame.image.load(WHITE_LINE_PATH).convert()
    IMAGES['walk_left'] = pygame.image.load(PEDES_LEFT_PATH).convert()
    IMAGES['walk_right'] = pygame.image.load(PEDES_RIGHT_PATH).convert()

    IMAGES['numbers'] = (
        pygame.image.load('Images/0.png').convert_alpha(),
        pygame.image.load('Images/1.png').convert_alpha(),
        pygame.image.load('Images/2.png').convert_alpha(),
        pygame.image.load('Images/3.png').convert_alpha(),
        pygame.image.load('Images/4.png').convert_alpha(),
        pygame.image.load('Images/5.png').convert_alpha(),
        pygame.image.load('Images/6.png').convert_alpha(),
        pygame.image.load('Images/7.png').convert_alpha(),
        pygame.image.load('Images/8.png').convert_alpha(),
        pygame.image.load('Images/9.png').convert_alpha())

    return IMAGES




