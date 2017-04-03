import pygame
import sys


FPS = 30
SCREENWIDTH  = 600
SCREENHEIGHT = 800

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Deep Traffic')

IMAGES = utils.load()

BACKGROUND_WIDTH = IMAGES['background'].get_width()

RED_CAR_WIDTH = IMAGES['red_car'].get_width()
RED_CAR_HEIGHT = IMAGES['red_car'].get_height()

WHITE_CAR_WIDTH = IMAGES['white_car'].get_width()
WHITE_CAR_HEIGHT = IMAGES['white_car'].get_height()

ROAD_WIDTH = IMAGES['road'].get_width()
ROAD_HEIGHT = IMAGES['road'].get_height()

LANE = {}
LANE[0] = 150
LANE[1] = 240
LANE[2] = 330
LANE[3] = 400

BASE_SHIFT = 100

def load():

    RED_CAR_PATH = 'Images/red_car.png'
    WHITE_CAR_PATH = 'Images/white_car.png'
    WHITE_CAR_REVERSE_PATH = 'Images/white_car_reverse.png'
    ROAD_PATH = 'Images/road.png'
    BACKGROUND_PATH = 'Images/background-black.png'
    PARKING_PATH = 'Images/parking.png'

    IMAGES = {}
    IMAGES['road'] =  pygame.image.load(ROAD_PATH).convert()
    IMAGES['red_car'] = pygame.image.load(RED_CAR_PATH).convert()
    IMAGES['white_car'] = pygame.image.load(WHITE_CAR_PATH).convert()
    IMAGES['white_car_reverse'] = pygame.image.load(WHITE_CAR_REVERSE_PATH).convert()
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()
    IMAGES['park'] = pygame.image.load(PARKING_PATH).convert()
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