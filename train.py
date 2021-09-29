import argparse

import numpy as np
import gym
import torch
from utils import DrawLine
from agent import Agent,Env
from network import Net
vis = False
render=True

if __name__ == "__main__":
    agent = Agent()
    env = Env()
    if vis:
        draw_reward = DrawLine(env="car", title="PPO", xlabel="Episode", ylabel="Moving averaged episode reward")

    training_records = []
    running_score = 0
    moving_average = np.array
    state = env.reset()
    for i_ep in range(2000):
        score = 0
        state = env.reset()

        for t in range(1000):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if render:
                env.render()
            if agent.store((state, action, a_logp, reward, state_)):
                print('updating')
                agent.update()
            score += reward
            state = state_
            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01
        print('Score: {:.2f}, Action taken: {}'.format(score, t+1))

        if i_ep % 10 == 0:
            if vis:
                draw_reward(xdata=i_ep, ydata=running_score)
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
            agent.save_param()
        if running_score > env.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break
