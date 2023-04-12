# noqa
"""
# Simple

```{figure} mpe_simple.gif
:width: 140px
:name: simple
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_v2` |
|--------------------|----------------------------------------|
| Actions            | Discrete/Continuous                    |
| Parallel API       | Yes                                    |
| Manual Control     | No                                     |
| Agents             | `agents= [agent_0]`                    |
| Agents             | 1                                      |
| Action Shape       | (5)                                    |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (5,))        |
| Observation Shape  | (4)                                    |
| Observation Values | (-inf,inf)                             |
| State Shape        | (4,)                                   |
| State Values       | (-inf,inf)                             |


In this environment a single agent sees a landmark position and is rewarded based on how close it gets to the landmark (Euclidean distance). This is not a multiagent environment, and is primarily intended for debugging purposes.

Observation space: `[self_vel, landmark_rel_position]`

### Arguments

``` python
simple_v2.env(max_cycles=25, continuous_actions=False)
```



`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""
import sys
sys.path.append("./multiagent-particle-envs/multiagent")
import numpy as np
#from gymnasium.utils import EzPickle
#from pettingzoo.utils.conversions import parallel_wrapper_fn
# from _mpe_utils.core import Agent, Landmark, World
# from _mpe_utils.scenario import BaseScenario
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
#from _mpe_utils.simple_env import SimpleEnv, make_env

# from .._mpe_utils.core import Agent, Landmark, World
# from .._mpe_utils.scenario import BaseScenario
# from .._mpe_utils.simple_env import SimpleEnv, make_env

# class raw_env(SimpleEnv, EzPickle):
#     def __init__(self, N=1, max_cycles=25, continuous_actions=False, render_mode=None):
#         EzPickle.__init__(self, N, max_cycles, continuous_actions, render_mode)
#         scenario = Scenario()
#         world = scenario.make_world(N)
#         super().__init__(
#             scenario=scenario,
#             world=world,
#             render_mode=render_mode,
#             max_cycles=max_cycles,
#             continuous_actions=continuous_actions,
#         )
#         self.metadata["name"] = "simple_multi_v2"


# env = make_env(raw_env)
# parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, N=1):
        world = World()
        world.dim_c = 2
        num_agents = N
        world.num_agents = num_agents
        num_adversaries = 0
        num_landmarks = max(num_agents - 1, 1)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            
            agent.name = f"{base_name}_{base_index}" #f"agent_{i}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.85, 0.35, 0.35])
        # random properties for landmarks
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.color = np.array([0.75, 0.75, 0.75])
        # world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.15, 0.15])
        goal = np.random.choice(world.landmarks)
        goal.color = np.array([0.15, 0.65, 0.15])
        for agent in world.agents:
            agent.goal_a = goal

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]


    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        #return np.concatenate([agent.state.p_vel] + entity_pos)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        if not agent.adversary:
            return np.concatenate(
                [agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos
            )
        else:
            return np.concatenate(entity_pos + other_pos)

