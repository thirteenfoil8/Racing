import numpy as np
import os
import gym
import torch
import torch.nn as nn
from network import Net
from agent import Agent_test, Env

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(0)
if use_cuda:
    torch.cuda.manual_seed(0)


render=False
if __name__ == "__main__":
    agent = Agent_test()
    agent.load_param()
    env = Env()

    training_records = []
    running_score = 0
    state = env.reset()
    for i_ep in range(1):
        score = 0
        state = env.reset()

        for t in range(1000):
            action = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if render:
                env.render()
            score += reward
            state = state_
            if done or die:
                break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
    os.system('cmd /c "ffmpeg -i ./recording/vid.mp4 -vf  "setpts=10*PTS" ./recording/vid2.mp4"')
