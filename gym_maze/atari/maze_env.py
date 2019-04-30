import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt

# Libraires for sending external stimulations over TCP port
import sys
import socket
from time import time, sleep

def_maze =  np.array([
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0.],
    [ 0., 1.,  0.,  1.,  1.,  1.,  1.,  1., 0., 0.],
    [ 0., 1.,  1.,  1.,  0.,  0.,  1.,  0., 0., 0.],
    [ 0., 0.,  0.,  0.,  1.,  1.,  1.,  0., 0., 0.],
    [ 0., 1.,  1.,  1.,  1.,  0.,  0.,  0., 1., 0.],
    [ 0., 1.,  0.,  0.,  0.,  1.,  1.,  1., 1., 0.],
    [ 0., 1.,  0.,  1.,  1.,  1.,  0.,  0., 1., 0.],
    [ 0., 1.,  1.,  1.,  0.,  1.,  1.,  0., 1., 0.],
    [ 0., 0.,  1.,  1.,  1.,  0.,  0.,  0., 1., 0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0.]
])

class ALEInterface(object):
    def __init__(self):
      self._lives_left = 0

    def lives(self):
      return 0 #self.lives_left

class MazeEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second' : 50}

    def __init__(self, grid_size=(10, 10), maze=def_maze, start=(1,1), tcp_tagging=False, tcp_port=15361):

        # Atari-platform related parameters
        self._atari_dims = (210,160,3)		# Specifies standard atari resolution
        (self._atari_height, self._atari_width, self._atari_channels) = self._atari_dims

        #  Game-related paramteres
        self._screen_height = grid_size[0]
        self._screen_width = grid_size[1]
        self._screen_dims = [self._screen_height, self._screen_width]

        self._maze = np.array(maze)
        self._start = list(start)
        self._target = [grid_size[0]-2, grid_size[1]-2]
        # Sanity Checks
        if self._maze[self._target[0], self._target[1]] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if self._maze[self._start[0], self._start[1]] == 0.0:
            raise Exception("Invalid start location!")
        self._actions = [[0,-1],[1,0],[-1,0],[0,1]]     # Up, Right, Left, DOWN
        self._score = 0.0
        self._state = [self._start[0], self._start[1], self._start[0], self._start[1]]   # [agent_x, agent_y, ghost_x, ghost_y]

        # Gym-related variables [must be defined]
        self._action_set = np.array([2, 3, 4, 5],dtype=np.int32)
        self._action_space = spaces.Discrete(4)
        self._observation_space = spaces.Box(low=0, high=255, shape=(self._atari_height, self._atari_width, 3), dtype=np.uint8)
        self._viewer = None

        # Display based variables
        self._offset = 25  # need to display square screen, 210x160 -> 160x160, so 25 pixels

        # Code for TCP Tagging
        self._tcp_tagging = tcp_tagging
        if (self._tcp_tagging):
            self._host = '127.0.0.1'
            self._port = tcp_port
            self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._s.connect((self._host, self._port))

        # Methods
        self._ale = ALEInterface()
        self.seed()
        self.reset()

    # Act by taking an action # return observation (object), reward (float), done (boolean) and info (dict)
    def step(self, action):
        if isinstance(action, np.ndarray):
          action = action[0]
        assert self._action_space.contains(action)   # makes sure the action is valid

        # Updating the state, state is hidden from observation
        [agent_x, agent_y, ghost_x, ghost_y] = self._state
        current_action = self._actions[action]
        ghost_x = agent_x + current_action[0]
        ghost_y = agent_y + current_action[1]

        reward, done = 0.0, False
        if self._maze[ghost_x][ghost_y] == 1.0: # Correct action
            agent_x, agent_y = ghost_x, ghost_y
        else:
            reward = -0.75
        if agent_x==self._target[0] and agent_y==self._target[1]:
            reward = 1.0
            done = True
        self._score = self._score + reward

        self._state = [agent_x, agent_y, ghost_x, ghost_y]

        # Sending the external stimulation over TCP port
        if self._tcp_tagging:
            padding=[0]*8
            event_id = [0, 0, 0, agent_x, agent_y, ghost_x, ghost_y, action]
            timestamp=list(self.to_byte(int(time()*1000), 8))
            self._s.sendall(bytearray(padding+event_id+timestamp))

        return self._get_observation(), reward, done, {"ale.lives": self._ale.lives(), "internal_state": self._state}

    def reset(self):
        self._score = 0.0
        self._state = self._state = [self._start[0], self._start[1], self._start[0], self._start[1]]
        return self._get_observation()

    def _get_observation(self):
        img = np.zeros(self._atari_dims, dtype=np.uint8) # Black screen
        block_width = int(self._atari_width/self._screen_width)
        [agent_x, agent_y, ghost_x, ghost_y] = self._state

        #Draw wall blocks
        for idx_x in range(self._screen_height):
            for idx_y in range(self._screen_width):
                if self._maze[idx_x, idx_y] == 0.0: # mark with red
                    img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 0] = 255
                else: # mark with white
                    img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 0:3] = 255


        #Draw ghost
        x_c = ghost_x*block_width + block_width/2 + self._offset
        y_c = ghost_y*block_width + block_width/2
        r = block_width/3
        for x in range(r):
            for y in range(r):
                if x*x + y*y < r*r:
                    img[x_c+x,y_c+y,1]=64
                    img[x_c+x,y_c-y,1]=64
                    img[x_c-x,y_c+y,1]=64
                    img[x_c-x,y_c-y,1]=64
                    img[x_c+x,y_c+y,[0,2]] = 0
                    img[x_c+x,y_c-y,[0,2]] = 0
                    img[x_c-x,y_c+y,[0,2]] = 0
                    img[x_c-x,y_c-y,[0,2]] = 0
        #Draw agent
        x_c = agent_x*block_width + block_width/2 + self._offset
        y_c = agent_y*block_width + block_width/2
        r = block_width/3
        for x in range(r):
            for y in range(r):
                if x*x + y*y < r*r:
                    img[x_c+x,y_c+y,2]=255
                    img[x_c+x,y_c-y,2]=255
                    img[x_c-x,y_c+y,2]=255
                    img[x_c-x,y_c-y,2]=255
                    img[x_c+x,y_c+y,0:2] = 0
                    img[x_c+x,y_c-y,0:2] = 0
                    img[x_c-x,y_c+y,0:2] = 0
                    img[x_c-x,y_c-y,0:2] = 0




        return img

    def render(self, mode='human', close=False):
        img = self._get_observation()
        if mode == 'rgb_array':
            return img
        #return np.array(...) # return RGB frame suitable for video
        elif mode is 'human':
            #... # pop up a window and render
            from gym.envs.classic_control import rendering
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer(maxwidth=1920)
            self._viewer.imshow(np.repeat(np.repeat(img, 4, axis=0), 4, axis=1))
            return self._viewer.isopen
            #plt.imshow(img)
            #plt.show()
        else:
            super(CatchEnv, self).render(mode=mode) # just raise an exception

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self.tcp_tagging:
            self.s.close()

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    @property
    def _n_actions(self):
        return len(self._action_set)

    def seed(self, seed=None):
        self._np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]

    # A function for TCP_tagging in openvibe
    # transform a value into an array of byte values in little-endian order.
    def to_byte(self, value, length):
        for x in range(length):
            yield value%256
            value//=256


    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'UP':      ord('w'),
            'DOWN':    ord('s'),
            'LEFT':    ord('a'),
            'RIGHT':   ord('d'),
            'FIRE':    ord(' '),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self._get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action


ACTION_MEANING = {
    0 : "NOOP",
    1 : "FIRE",
    2 : "UP",
    3 : "RIGHT",
    4 : "LEFT",
    5 : "DOWN",
    6 : "UPRIGHT",
    7 : "UPLEFT",
    8 : "DOWNRIGHT",
    9 : "DOWNLEFT",
    10 : "UPFIRE",
    11 : "RIGHTFIRE",
    12 : "LEFTFIRE",
    13 : "DOWNFIRE",
    14 : "UPRIGHTFIRE",
    15 : "UPLEFTFIRE",
    16 : "DOWNRIGHTFIRE",
    17 : "DOWNLEFTFIRE",
}
