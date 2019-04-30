# gym-catch
A simple catch game (with atari rendering) in the gym OpenAI environment

<p align="center">
  <img src="extras/catch_screenshot.png" width="150" title="Screenshot of Wobble Game">
</p>

## Installation instructions
----------------------------

Requirements: gym with atari dependency

```shell
git clone https://github.com/meagmohit/gym-catch
cd gym-catch
python setup.py install
```

```python
import gym
import gym_catch
env = gym.make('catch-v0') # The other option is 'CatchNoFrameskip-v4'
env.render()
```

## Environment Details
----------------------

* **catch-v0 :** Default settings (`grid_size=(105,20)`, `bar_size=5`, `total_balls=10`, `tcp_tagging=False`, `tcp_port=15361`)
* **CatchNoFrameskip-v1 :** Default settings and `grid_size=(10,10)`, `bar_size=1`
* **CatchNoFrameskip-v2 :** Default settings and `grid_size=(42,10)`, `bar_size=1`
* **CatchNoFrameskip-v3 :** Default settings and `grid_size=(10,10)`, `bar_size=1`, `tcp_tagging=True`
* **CatchNoFrameskip-v4 :** Default settings

## Agent Details
----------------

* `agents/random_agent.py` random agent plays game with given error probability to take actions (Perr)

## References
-------------
