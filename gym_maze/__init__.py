from gym.envs.registration import register

register(
    id='maze-v0',
    entry_point='gym_maze.atari:MazeEnv',
)


# register(
#     id='CatchNoFrameskip-v4',
#     entry_point='gym_catch.atari:CatchEnv',
#     #kwargs={'game': 'catch', 'obs_type': 'image', 'frameskip': 1}, # A frameskip of 1 means we get every frame
#     max_episode_steps=10000,
#     nondeterministic=False,
# )
#
# register(
#     id='CatchNoFrameskip-v1',
#     entry_point='gym_catch.atari:CatchEnv',
#     kwargs={'grid_size': (10,10), 'bar_size': 1, 'total_balls': 10}, # A frameskip of 1 means we get every frame
#     max_episode_steps=10000,
#     nondeterministic=False,
# )
#
# register(
#     id='CatchNoFrameskip-v2',
#     entry_point='gym_catch.atari:CatchEnv',
#     kwargs={'grid_size': (42,10), 'bar_size': 1, 'total_balls': 10}, # A frameskip of 1 means we get every frame
#     max_episode_steps=10000,
#     nondeterministic=False,
# )
#
register(
    id='MazeNoFrameskip-v3',
    entry_point='gym_maze.atari:MazeEnv',
    kwargs={'tcp_tagging': True}, # A frameskip of 1 means we get every frame
    max_episode_steps=10000,
    nondeterministic=False,
)
