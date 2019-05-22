import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# try:
#
# except ImportError as e:
#     logger.warn('failed to set matplotlib backend, plotting will not work: %s' % str(e))
#     plt = None

import gym
import pygame
import argparse
from gym import logger
from collections import deque
from pygame.locals import VIDEORESIZE

import gym_maze
import gym_catch
import time
from gym.utils.play import *

def play2(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None):
    env.reset()
    rendered=env.render( mode='rgb_array')

    if keys_to_action is None:
        if hasattr(env, 'get_keys_to_action'):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, 'get_keys_to_action'):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert False, env.spec.id + " does not have explicit key to action mapping, " + \
                          "please specify one manually"
    relevant_keys = set(sum(map(list, keys_to_action.keys()),[]))

    video_size=[rendered.shape[1],rendered.shape[0]]
    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    env_done = True

    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()


    while running:
        if env_done:
            env_done = False
            obs = env.reset()
        else:
            if len(pressed_keys) ==1:
                action = keys_to_action.get(tuple(sorted(pressed_keys)))
                prev_obs = obs
                obs, rew, env_done, info = env.step(action)
                if callback is not None:
                    callback(prev_obs, obs, action, rew, env_done, info)
        if obs is not None:
            rendered=env.render( mode='rgb_array')
            display_arr(screen, rendered, transpose=transpose, video_size=video_size)
            rendered = env.render(mode='rgb_array', dstate=1)
            display_arr(screen, rendered, transpose=transpose, video_size=video_size)
            time.sleep(0.05)
            rendered = env.render(mode='rgb_array', dstate=2)
            display_arr(screen, rendered, transpose=transpose, video_size=video_size)
            time.sleep(0.05)

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in relevant_keys:
                    pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                print(video_size)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


env = gym.make("maze-v0")
play2(env, zoom=4, fps=10)
