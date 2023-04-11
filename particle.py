# from pettingzoo.mpe import simple_multi_v2
# from pettingzoo.mpe import simple_adversary_v2
import simple_multi_v2

env = simple_multi_v2.env(N=1, render_mode='human')
#env = simple_adversary_v2.env(N=2, render_mode='human')

env.reset()
i = 0
while(True):
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
            env.reset()
            break
        else:
            action = env.action_space(agent).sample()
        print(action)
        env.step(action)
    i = i + 1
env.close()