import random
from .utils import *
from .white_car import WhiteCar
from copy import deepcopy

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

        # player settings
        self.pipeVelX = 0
        self.playerVelY = 1    
        self.playerAccY = 0  
        self.playerMaxV = 6
        if random.randint(0, 1) == 0:
            self.up = True
            self.playery = random.randint(400, SCREENHEIGHT - RED_CAR_HEIGHT)
            self.playerx = LANE[2]

            self.lane = 2
        else:
            self.playerx = LANE[1]
            self.lane = 0
            self.up = False
            self.playery = random.randint(0,400)
        # road 
        self.basex = BASE_SHIFT 
        
        # screen
        self.circle = False # player goes so fast, need to update entire screen

        # traffic light
        self.light_down = TrafficLight(LIGHT1_POS[0], LIGHT1_POS[1], False)
        self.light_up = TrafficLight(LIGHT2_POS[0], LIGHT2_POS[1], False)
        self.green_starts1 = 0
        self.green_starts2 = 0
        self.yellow_starts1 = 0
        self.yellow_starts2 = 0

        # white car
        self.max_white_car = 7
        self.white_cars = {} # key is idx, and value is white car object

        # initialize white car and environment
        self.car_maps = {0:[], 1:[], 2:[],3:[]} # key is lane, and value is (key,y, speed) pair
        self.car_maps[self.lane].append(Car(self.max_white_car, self.playerVelY, self.lane, self.playery, self.up))


        self.env = Environment(self.car_maps, [self.light_down, self.light_up], None)
        self.init_white_car()

        # initialize clever pedestrian
        self.walk_pedes = None
        self.init_pedes()

        # for calculate reward
        self.old_env = deepcopy(self.env)
        self.old_y = self.playery

    def init_pedes(self):
        # init pedestrian

        if random.randint(0, 10) != 0:
            self.walk_pedes = None
            self.env.set_pedestrain_info(self.walk_pedes)
            return

        res = find_space(self.car_maps)
        if res:
            if random.randint(0,1) == 0: # left
                self.walk_pedes = Pedestrain(3, PEDES_RIGHT, res, True)
            else:
                self.walk_pedes = Pedestrain(3, PEDES_LEFT, res, False)
        else:
            self.walk_pedes = None

        self.env.set_pedestrain_info(self.walk_pedes)

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
            if lane in (0, 1):
                y = 0
            else:
                y = SCREENHEIGHT

        # avoid collision with other initialization
       
        initial_collision = any( check_collision(lane, y, lane, car.y, extra=40) for car in self.car_maps[lane])
        if initial_collision:
            return None

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
        for i in self.white_cars:
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
        self.playerAccY, self.up = reward_info.acc_delta, reward_info.up
 
        # summary

        self.playerVelY += self.playerAccY

        if self.playerVelY < 0:
            self.playerVelY = 0

        if self.playerVelY > self.playerMaxV:
            self.playerVelY = self.playerMaxV

        if self.up:
            self.playery -= self.playerVelY
        else:
            self.playery += self.playerVelY

        if self.playery < 0 and self.up :
            self.circle = True

        if self.playery > SCREENHEIGHT - 60 and not self.up:
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

    def update_traffic_light(self):
        # update the traffic light

        # down light
        if self.num_frame - self.green_starts1 == 2 * LIGHT_INTERVAL:
            self.light_down = TrafficLight(LIGHT1_POS[0], LIGHT1_POS[1], True)

            self.env.set_traffic_info([self.light_down, self.light_up])

        if self.num_frame - self.green_starts1 == 4 * LIGHT_INTERVAL:
            self.light_down = TrafficLight(LIGHT1_POS[0], LIGHT1_POS[1], False)

            self.green_starts1 = self.num_frame

        # up light
        if self.num_frame - self.green_starts2 == 3 * LIGHT_INTERVAL:
            self.light_up = TrafficLight(LIGHT2_POS[0], LIGHT2_POS[1], True)

            self.env.set_traffic_info([self.light_down, self.light_up])

        if self.num_frame - self.green_starts2 == 5 * LIGHT_INTERVAL:
            self.light_up = TrafficLight(LIGHT2_POS[0], LIGHT2_POS[1], False)

            self.green_starts2 = self.num_frame

        self.env.set_traffic_info([self.light_down, self.light_up])
        # update environment for every white car
        for elem in self.white_cars:
            self.white_cars[elem].update_env(self.env)


    def draw_traffic_light(self):

        if self.light_down.red:
            if self.yellow_starts1 < YELLOW_TIME:
                self.yellow_starts1 += 1
                SCREEN.blit(IMAGES['yellow_light'], LIGHT1_POS)
            else:

                SCREEN.blit(IMAGES['red_light'], LIGHT1_POS)
        else:
            self.yellow_starts1 = 0
            SCREEN.blit(IMAGES['green_light'], LIGHT1_POS)

        if self.light_up.red:
            if self.yellow_starts2  < YELLOW_TIME:
                self.yellow_starts2 += 1
                SCREEN.blit(IMAGES['yellow_light'], LIGHT2_POS)
            else:

                SCREEN.blit(IMAGES['red_light'], LIGHT2_POS)
        else:
            self.yellow_starts2 = 0
            SCREEN.blit(IMAGES['green_light'], LIGHT2_POS)

    def update_pedestrian(self):

        if not self.walk_pedes:
            self.init_pedes()
            return

        if self.walk_pedes.left:
            # left walk pedestrian
            new_x  = self.walk_pedes.x - self.walk_pedes.speed
            # try to avoid collion with player
            if abs(new_x - self.playerx) < 60 and abs(self.walk_pedes.y - self.playery) < 60:
                new_x = self.walk_pedes.x

            if new_x < PEDES_LEFT:
                self.walk_pedes = None
                self.init_pedes()
            else:
                self.walk_pedes = Pedestrain(3, new_x, self.walk_pedes.y, True)

        else:# right walk pedestrian
            new_x_2 = self.walk_pedes.x + self.walk_pedes.speed
            if abs(new_x_2 - self.playerx) < 60 and abs(self.walk_pedes.y - self.playery) < 60:
                new_x_2 = self.walk_pedes.x
            if new_x_2 > PEDES_RIGHT:
                self.walk_pedes = None
                self.init_pedes()
            else:
                self.walk_pedes = Pedestrain(3, new_x_2, self.walk_pedes.y, False)


        self.env.set_pedestrain_info(self.walk_pedes)

        for elem in self.white_cars:
            self.white_cars[elem].update_env(self.env)

    def draw_pedestrian(self):

        if not self.walk_pedes:
            return

        if self.walk_pedes.left:
            x, y = self.walk_pedes.x, self.walk_pedes.y
            SCREEN.blit(IMAGES['walk_left'], (x, y))
        else:
            x2, y2 = self.walk_pedes.x, self.walk_pedes.y
            SCREEN.blit(IMAGES['walk_right'], (x2, y2))


    def check_crash(self):
        # check crash with white car
        
        terminal = False
        for elem in self.white_cars:
            x,y = self.white_cars[elem].getXY()
            if check_collision(self.playerx, self.playery, x, y, extra=10):
                terminal = True
                break

        return terminal

    def check_obey_traffic(self):
        # check whether obey the traffic light
        terminal = False

        if self.up:
            my_light = self.light_up
            red = my_light.red
            if red:
                if self.playery > RED_STOP_UP and self.playery - RED_STOP_UP < 6:
                    if self.playerVelY != 0:
                        terminal = True
        else:
            my_light = self.light_down
            red = my_light.red
            if red:
                if self.playery < RED_STOP_DOWN and RED_STOP_DOWN - self.playery < 6:
                    if self.playerVelY != 0:
                        terminal = True
        return terminal

    def check_hit_pedestrian(self):

        terminate = False
        if not self.walk_pedes:
            return

        x1, y1 = self.walk_pedes.x, self.walk_pedes.y


        if abs(x1 - self.playerx) < 40 and abs(y1 - self.playery) < 60:
            terminate = True

        return terminate

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

    def calculate_passed_car(self):
        # calculate 
        calculate_map = {
            True: {2 : 3, 3 : 2},
            False: {0 : 1, 1: 0}
        }
        behind = set()
        front = set()
        
        check_lane = calculate_map[self.up][self.lane]
        for cars in self.old_env.get_cars(check_lane):
            if self.old_y > cars.y:
                front.add(cars.idx)
        for cars in self.env.get_cars(check_lane):
            if self.playery < cars.y:
                behind.add(cars.idx)
        
        self.old_y = self.playery
        self.old_env = deepcopy(self.env)

        return len(behind & front)


    def frame_step(self, input_actions):
        # update frame number 
        self.num_frame += 1

        pygame.event.pump()

        if input_actions < 0 or input_actions > 5 :
            raise ValueError('Not a valid operation')


        # traffic light
        self.update_traffic_light()

        # update_pedestrian
        self.update_pedestrian()

        # update white car
        self.update_white_car()

        # update player
        reward, terminal1 = self.update_player(input_actions)

        # check if crash here
        terminal2 = self.check_crash()

        # check whether obey traffic light
        terminal3 = self.check_obey_traffic()

        # check whether hit perestrian

        terminal4 = self.check_hit_pedestrian()

        # get entire new screen
        if self.circle:
            if random.randint(0, 1) == 0:
                self.lane = 2
                self.playerx = LANE[2]
                self.up = True
                self.playery = SCREENHEIGHT - RED_CAR_HEIGHT
            else:
                self.lane = 1
                self.playerx = LANE[1]
                self.up = False
                self.playery = 0

            self.circle = False
            self.init_white_car()

        # update global env for every object on screen
        self.update_global_env()

        terminal = terminal1 or terminal2 or terminal3 or terminal4
        # handle termianl case
        if terminal:
            # make sure reward is -1
            reward = -1
            self.__init__()
        else:
            reward = 1 #self.calculate_passed_car() * 10 + self.playerVelY
        # draw 
        SCREEN.blit(IMAGES['background'], (0,0))

        SCREEN.blit(IMAGES['road'], (self.basex, 0))

        # stop line
        SCREEN.blit(IMAGES['white_line'], (142,RED_STOP_DOWN+60))
        SCREEN.blit(IMAGES['white_line'], (310, RED_STOP_UP))

        # red car
        if self.up:
            SCREEN.blit(IMAGES['red_car'], (self.playerx, self.playery))
        else:
            SCREEN.blit(IMAGES['red_car_down'], (self.playerx, self.playery))

        for elem in self.white_cars:
            x,y = self.white_cars[elem].getXY()
            p = self.white_cars[elem].getPark()
            if x <= LANE[1]:
                if not p:
                    SCREEN.blit(IMAGES['white_car_reverse'], (x,y))
                else:
                    SCREEN.blit(IMAGES['down_park_car'], (x,y))
            else:
                if not p:
                    SCREEN.blit(IMAGES['white_car'], (x,y))
                else:
                    SCREEN.blit(IMAGES['up_park_car'], (x,y))
        # draw traffic light
        self.draw_traffic_light()

        # draw pedestrian()
        self.draw_pedestrian()

        # encouge speed up
        # if not terminal and self.playerVelY == 0:
        #    reward = -0.5
        
        # update score
        if reward >= 0:
            score_update = int(reward*2)
            self.score += score_update
        showScore(int(self.score/50))
        
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)

        if self.up:
            return image_data, reward, terminal, (self.playerx, self.playery), self.up, self.light_up.red , self.playerVelY, self.walk_pedes
        else:
            return image_data, reward, terminal, (self.playerx, self.playery), self.up, self.light_down.red, self.playerVelY, self.walk_pedes

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


def check_collision(x1, y1, x2, y2, extra=20):
    # check whther 2 car collide with each other
    if x1 != x2:
        return False

    if abs(y1-y2) <= (extra + RED_CAR_HEIGHT):
        return True
    else:
        return False


