from gym.envs.registration import register

register(
    id='maze-v0',
    entry_point='gym_maze.atari:MazeEnv',
)

register(
    id='MazeNoFrameskip-v3',
    entry_point='gym_maze.atari:MazeEnv',
    kwargs={'tcp_tagging': True}, # A frameskip of 1 means we get every frame
    max_episode_steps=10000,
    nondeterministic=False,
)
