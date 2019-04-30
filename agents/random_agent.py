import gym
import gym_maze
import time
import numpy as np

env = gym.make('maze-v0')
env.reset()
action_idx = [2, 3, 4, 5]
# actions = [0, 1, -1]	# 0 means stay, 1 mean right i.e. 1, 2 means -1 i.e. left
# p_err = 0.5
for _ in range(1):
	done = False
	env.reset()
	while not done:
		# state = env.unwrapped.state
		# [ball_row, ball_col, bar_col] = state
		# if ball_col == bar_col:	# correct action is stay
		# 	action = np.random.choice(action_idx, p=[1-p_err, p_err/2.0, p_err/2.0])
		# elif ball_col > bar_col: #correct action is move right
		# 	action = np.random.choice(action_idx, p=[p_err/2.0, 1-p_err, p_err/2.0])
		# else:
		# 	action = np.random.choice(action_idx, p=[p_err/2.0, p_err/2.0, 1-p_err])
		action = env._action_space.sample()
		observation, reward, done, info = env.step(action)
		print action, reward
		env.render()
		time.sleep(2.0)

print env.unwrapped.score
env.close()
