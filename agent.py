import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
import numpy as np
from torch.distributions import Beta
import gym
from network import Qnetwork,Net

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
transition = np.dtype([('s', np.float64, (4, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (4, 96, 96))])

class Env():
    """
    Environment wrapper for CarRacing
    """

    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.env.seed(0)
        self.reward_threshold = self.env.spec.reward_threshold

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * 4  # four frames for decision
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(8):
            img_rgb, reward, die, _ = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == 4
        return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 1200, 128

    def __init__(self):
        self.training_step = 0
        self.net = Net().double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        del state
        return action, a_logp

    def save_param(self):
        torch.save(self.net.state_dict(), 'param/ppo_net_params.pkl')

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + args.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
        # Del from gpu to avoid overflow.
        del s, a, r, s_, old_a_logp

class Agent_test():
    """
    Agent for testing
    """

    def __init__(self):
        self.net = Net().float().to(device)

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)

        action = action.squeeze().cpu().numpy()
        return action

    def load_param(self,path= 'param/ppo_net_params.pkl'):
        print(path)
        self.net.load_state_dict(torch.load(path))

class Agent_DQN(object):
    def __init__(self, gamma, epsilon, epsilonDecay, epsilonMin, alpha, maxMemSize, actionSpace, targetReplaceCount, episodeEnd):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin
        self.alpha = alpha
        self.maxMemSize = maxMemSize
        self.actionSpace = actionSpace
        self.episodeEnd = episodeEnd
        self.targetReplaceCount = targetReplaceCount

        self.steps = 0
        self.stepCounter = 0
        self.memory = []
        self.memCounter = 0

        self.Qevaluation = Qnetwork(len(self.actionSpace), alpha)
        self.Qprediction = Qnetwork(len(self.actionSpace), alpha)

    def storeTransition(self, state, action, reward, state_):
        if self.memCounter < self.maxMemSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCounter % self.maxMemSize] = [state, action, reward, state_]
        self.memCounter += 1

    def chooseAction(self, observation):
        randomActionChance = np.random.random()

        if randomActionChance < self.epsilon:
            action = np.random.choice(self.actionSpace)

        else:
            action = torch.argmax(self.Qevaluation.forward(observation)[1].item())

        self.steps += 1
        return action

    def learn(self, batchSize):
        self.Qevaluation.optimizer.zero_grad()
        # if self.targetReplaceCount is not None and self.stepCounter%self.targetReplaceCount == 0:
        #     #TODO remove ? not used ?
        #     self.Qprediction.load_state_dict(self.Qevaluation.state_dict())

        minibatch = []
        if self.memCounter < batchSize:
            minibatch = self.memory[0:self.memCounter-1]
        else:
            batchStart = np.random.randint(0, self.memCounter - batchSize)
            minibatch = self.memory[batchStart: batchStart + batchSize]

        memory = np.array(minibatch)

        # memory : []
        evaluation = self.Qevaluation.forward(list(memory[:, 0][:])).to(self.Qevaluation.device)
        prediction = self.Qprediction.forward(list(memory[:, 3][:])).to(self.Qprediction.device)

        maxAction = torch.argmax(prediction, dim=1).to(self.Qevaluation.device)
        rewards = torch.Tensor(list(memory[:, 2])).to(self.Qevaluation.device)
        target = evaluation

        # BELLMAN
        targetCalc = rewards + self.gamma * torch.max(prediction[1])
        maxPrediction= torch.max(prediction[1])
        target[:, maxAction] = rewards + self.gamma * torch.max(prediction[1])

        #LOSS Function
        loss = self.Qevaluation.loss(target, prediction).to(self.Qevaluation.device)
        loss.backward()

        self.Qevaluation.optimizer.step()
        self.stepCounter += 1
        # self.Qevaluation.optimizer.zero_grad()

        # EPSILON DECAY
        if self.steps > 500:
            if self.epsilon > self.epsilonMin:
                self.epsilon *= self.epsilonDecay
            else:
                self.epsilon = self.epsilonMin