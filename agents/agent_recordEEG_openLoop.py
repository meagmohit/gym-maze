import gym
import gym_maze
import time
import numpy as np

moves_maze =  np.array([
    [ 9., 9.,  9.,  9.,  9.,  9.,  9.,  9., 9., 9.],
    [ 9., 3.,  9.,  1.,  1.,  1.,  3.,  2., 9., 9.],
    [ 9., 1.,  1.,  0.,  9.,  9.,  3.,  9., 9., 9.],
    [ 9., 9.,  9.,  9.,  3.,  2.,  2.,  9., 9., 9.],
    [ 9., 3.,  2.,  2.,  2.,  9.,  9.,  9., 3., 9.],
    [ 9., 3.,  9.,  9.,  9.,  1.,  1.,  1., 3., 9.],
    [ 9., 3.,  9.,  1.,  1.,  0.,  9.,  9., 3., 9.],
    [ 9., 1.,  1.,  0.,  9.,  0.,  2.,  9., 3., 9.],
    [ 9., 9.,  1.,  0.,  2.,  9.,  9.,  9., 3., 9.],
    [ 9., 9.,  9.,  9.,  9.,  9.,  9.,  9., 9., 9.]
])

env = gym.make('MazeNoFrameskip-v3')

env.reset()
action_set = [0,1,2,3]
p_err = 0.2
speed = 0.5 # in seconds must be greater than 0.2
for _ in range(4):
    done = False
    env.reset()
    env.render()
    time.sleep(10)
    while not done:
        state = env.unwrapped._state[0:2] # agent_x, agent_y
        action_correct = int(moves_maze[state[0]][state[1]])
        print action_correct

        actions_incorrect = []
        for idx in range(4):
        	if not idx==action_correct:
        		actions_incorrect.append(idx)

        action_incorrect = np.random.choice(actions_incorrect)

        action = np.random.choice([action_correct, action_incorrect], p=[1-p_err, p_err])
        # action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print action, reward
        env.render(dstate=0)
        time.sleep(speed)
        env.render(dstate=1)
        time.sleep(0.1)
        env.render(dstate=2)
        time.sleep(speed - 0.1)

env.close()
