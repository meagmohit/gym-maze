import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt

# Libraires for sending external stimulations over TCP port
import sys
import socket
from time import time, sleep
from  maze_graphics import *

# Action Codes: 0,1,2,3 : Up, Right, Left and Down respectively
# Stimulation code: [0, 0, 0, agent_x, agent_y, ghost_x, ghost_y, action]
# (1,1) is the starting position, (8,8) is the end position


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
        self._actions = [[-1,0],[0,1],[0,-1],[1,0]]     # Up, Right, Left, DOWN
        self._score = 0.0
        self._state = [self._start[0], self._start[1], self._start[0], self._start[1]]   # [agent_x, agent_y, ghost_x, ghost_y]

        # Gym-related variables [must be defined]
        self.action_set = np.array([0,1,2,3],dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self._atari_height, self._atari_width, 3), dtype=np.uint8)
        self.viewer = None

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
        assert self.action_space.contains(action)   # makes sure the action is valid

        # Updating the state, state is hidden from observation
        # Ghost position is the agent old position
        [agent_x, agent_y, agent_prev_x, agent_prev_y, prev_action] = self._state
        agent_prev_x = agent_x
        agent_prev_y = agent_y

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


        self._state = [agent_x, agent_y, agent_prev_x, agent_prev_y, action]
        # print self._state

        # Sending the external stimulation over TCP port
        if self._tcp_tagging:
            padding=[0]*8
            event_id = [0, 0, 0, agent_x, agent_y, agent_prev_x, agent_prev_y, action]
            timestamp=list(self.to_byte(int(time()*1000), 8))
            self._s.sendall(bytearray(padding+event_id+timestamp))

        return self._get_observation(), reward, done, {"ale.lives": self._ale.lives(), "internal_state": self._state}

    def reset(self):
        self._score = 0.0
        self._state = [self._start[0], self._start[1], self._start[0], self._start[1], -1]
        return self._get_observation()

    # def _get_observation_animateA(self):
    #     img = self._get_observation()
    #     [agent_x, agent_y, agent_prev_x, agent_prev_y, prev_action] = self._state
    #
    #     if (agent_x == agent_prev_x) and (agent_y == agent_prev_y):


    def _get_observation(self, dstate=0):
        img = np.zeros(self._atari_dims, dtype=np.uint8) # Black screen
        block_width = int(self._atari_width/self._screen_width)
        [agent_x, agent_y, agent_prev_x, agent_prev_y, prev_action] = self._state

        #Draw wall blocks
        for idx_x in range(self._screen_height):
            for idx_y in range(self._screen_width):
                if self._maze[idx_x, idx_y] == 0.0: # mark with red
                    img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 0] = 200*arr_brick
                    # temp = img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 0]
                    # print temp.shape
                else: # mark with white
                    img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 0:3] = 255

        idx_x, idx_y = 8, 8
        img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 2] = 255
        img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 1] = (1-arr_target)*255
        img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 0] = (1-arr_target)*255

        # dstate=0: shows arrow with the intended movements
        # dstate=1 and same new position i.e. no movement: no agent is displayed
        # dstate=2: new position of the agent

        if dstate == 0:
            #Draw Previous Agent
            x_c = agent_prev_x*block_width + self._offset
            y_c = agent_prev_y*block_width

            for idx_x in range(arr_ghost.shape[0]):
                for idx_y in range(arr_ghost.shape[1]):
                    if arr_ghost[idx_x, idx_y] == 0:    # Boundary
                        img[x_c+idx_x, y_c+idx_y, :] = 0
                    elif arr_ghost[idx_x, idx_y] == 1:    # Body
                        img[x_c+idx_x, y_c+idx_y, 2] = 255
                        img[x_c+idx_x, y_c+idx_y, 0] = 0
                        img[x_c+idx_x, y_c+idx_y, 1] = 0
                        # img[x_c+idx_x, y_c+idx_y, 1] = 0
                    elif arr_ghost[idx_x, idx_y] == 2:    # Eyes
                        img[x_c+idx_x, y_c+idx_y, 1] = 0
                        img[x_c+idx_x, y_c+idx_y, 0] = 0
                        img[x_c+idx_x, y_c+idx_y, 2] = 0
                    else:   # background
                        img[x_c+idx_x, y_c+idx_y, :] = 255

            #Draw Arrow
            x_c = agent_prev_x*block_width + self._offset #+ block_width/2
            y_c = agent_prev_y*block_width# + block_width/2
            arr_arrow2 = np.copy(arr_arrow)
            if prev_action==0:  # Up
                arr_arrow2 = np.rot90(arr_arrow2)
                x_c = x_c - block_width/2
            elif prev_action==1:    #RIGHT
                y_c = y_c + block_width/2
            elif prev_action==2:    # left
                arr_arrow2 = np.flip(arr_arrow2)
                y_c = y_c - block_width/2
            elif prev_action==3:
                arr_arrow2 = np.transpose(arr_arrow2)
                x_c = x_c + block_width/2

            if prev_action in [0,1,2,3]:
                for idx_x in range(arr_arrow2.shape[0]):
                    for idx_y in range(arr_arrow2.shape[1]):
                        if arr_arrow2[idx_x, idx_y] == 1:
                            img[x_c+idx_x, y_c+idx_y, 1] = 255
                            img[x_c+idx_x, y_c+idx_y, 0] = 0
                            img[x_c+idx_x, y_c+idx_y, 2] = 0


        if ((dstate == 1) and not ((agent_x == agent_prev_x) and (agent_y == agent_prev_y))) or dstate==2:
            x_c = agent_x*block_width + self._offset
            y_c = agent_y*block_width

            for idx_x in range(arr_ghost.shape[0]):
                for idx_y in range(arr_ghost.shape[1]):
                    if arr_ghost[idx_x, idx_y] == 0:    # Boundary
                        img[x_c+idx_x, y_c+idx_y, :] = 0
                    elif arr_ghost[idx_x, idx_y] == 1:    # Body
                        img[x_c+idx_x, y_c+idx_y, 2] = 255
                        img[x_c+idx_x, y_c+idx_y, 0] = 0
                        img[x_c+idx_x, y_c+idx_y, 1] = 0
                    elif arr_ghost[idx_x, idx_y] == 2:    # Eyes
                        img[x_c+idx_x, y_c+idx_y, 1] = 0
                        img[x_c+idx_x, y_c+idx_y, 0] = 0
                        img[x_c+idx_x, y_c+idx_y, 2] = 0
                    else:
                        img[x_c+idx_x, y_c+idx_y, :] = 255

        return img

    def render(self, mode='human', close=False, dstate=0):
        img = self._get_observation(dstate)
        if mode == 'rgb_array':
            return img
        #return np.array(...) # return RGB frame suitable for video
        elif mode is 'human':
            #... # pop up a window and render
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=1920)
            self.viewer.imshow(np.repeat(np.repeat(img, 5, axis=0), 5, axis=1))
            return self.viewer.isopen
            #plt.imshow(img)
            #plt.show()
        else:
            super(CatchEnv, self).render(mode=mode) # just raise an exception

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self._tcp_tagging:
            self._s.close()

    def _get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self.action_set]

    @property
    def _n_actions(self):
        return len(self.action_set)

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
    0 : "UP",
    1 : "RIGHT",
    2 : "LEFT",
    3 : "DOWN",
}
