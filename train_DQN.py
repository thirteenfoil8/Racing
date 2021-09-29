import argparse

import numpy as np
import gym
import torch
from utils import DrawLine
from agent import Agent_DQN,Env
from network import Net
import itertools as it
vis = False
render=True

if __name__ == "__main__":
    agent = Agent_DQN()
    env = Env()
    all_actions = np.array([[-1, 0, 0],  [0,1, 0], [0, 0, 0.5], [0, 0, 0],[1, 0, 0]])
    n_actions = len(all_actions)
    if vis:
        draw_reward = DrawLine(env="car", title="PPO", xlabel="Episode", ylabel="Moving averaged episode reward")

    training_records = []
    running_score = 0
    moving_average = np.array
    state = env.reset()
    for i_ep in range(300000):
        score = 0
        state = env.reset()
        for t in range(1000):
            action = agent.select_action(state,t,n_actions)
            state_, reward, done, die = env.step(action)
            if render:
                env.render()
            if agent.store((state, action, reward, state_)):
                print('updating')
                agent.update()
            score += reward
            state = state_ 
            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01
        print('Score: {:.2f}, Action taken: {}, epsilon: {:.2f}'.format(score, t+1,agent.eps))

        if i_ep % 10 == 0:
            if vis:
                draw_reward(xdata=i_ep, ydata=running_score)
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
            agent.save_param()
        if running_score > env.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break