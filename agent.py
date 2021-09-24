import sys
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
import numpy as np
from torch.distributions import Beta
from collections import namedtuple, deque
from gym import spaces
import gym
from network import Net,DQN
import math

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
            # green penalty7
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
    ppo_epoch = 20
    buffer_capacity, batch_size = 1200, 100

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
        gamma = 0.99

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + gamma * self.net(s_)[1]
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

###--------------------------------------------------DQN Section 1-----------------------------------------####

BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE = 10
transition_dqn = np.dtype([('s', np.float64, (4, 96, 96)), ('a', np.float64, (3,)),
                       ('r', np.float64), ('s_', np.float64, (4, 96, 96))])


class Agent_DQN():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 1200, 100
    steps_done = 0
    
    

    def __init__(self):
        self.training_step = 0
        self.net = DQN().double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition_dqn)
        self.counter = 0
        self.eps=1


        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self,state,t,n_actions):
        if random.random() < self.eps:
            action = np.random.randint(8)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).double().to(device).unsqueeze(0)
                action = self.net(state)
                action = action.squeeze().cpu().numpy()
                return action

        if action == 1:
            return [0, 0, 0.5] # brake
        elif action == 2:
            return [0, 1, 0] # accelerate
        elif action == 3:
            return [1, 0, 0] # steer right
        elif action == 4:
            return [-1, 0, 0] # steer left
        elif action == 5:
            return [1, 0.7, 0] # steer right while accelerating
        elif action == 6:
            return [-1, 0.7, 0] # steer left while accelerating
        elif action == 7:
            return [1, 0, 0.3] # steer right while braking
        elif action == 0:
            return [-1, 0, 0.3] # steer left while braking

    def save_param(self):
        torch.save(self.net.state_dict(), 'param/ppo_net_params_DQN.pkl')

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)
        with torch.no_grad():
            target_v = r + GAMMA * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
        for _ in range(TARGET_UPDATE):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), BATCH_SIZE, True):
                value_loss = F.smooth_l1_loss(self.net(s[index]), target_v[index])
                loss = value_loss
                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
        # Del from gpu to avoid overflow.
        self.eps = max(0.99 * self.eps, 0.2)
        del s, a, r, s_

###--------------------------------------------------DQN Section 2-----------------------------------------####

from skimage import color, transform
import tf_slim as slim
import tensorflow as tf
import itertools as it
tf.compat.v1.disable_eager_execution()

class ExperienceHistory:
    """
    This saves the agent's experience in windowed cache.
    Each frame is saved only once but state is stack of num_frame_stack frames
    In the beginning of an episode the frame-stack is padded
    with the beginning frame
    """

    def __init__(self,
            num_frame_stack=4,
            capacity=int(1e5),
            pic_size=(96, 96)
    ):
        self.num_frame_stack = num_frame_stack
        self.capacity = capacity
        self.pic_size = pic_size
        self.counter = 0
        self.frame_window = None
        self.init_caches()
        self.expecting_new_episode = True

    def add_experience(self, frame, action, done, reward):
        assert self.frame_window is not None, "start episode first"
        self.counter += 1
        frame_idx = self.counter % self.max_frame_cache
        exp_idx = (self.counter - 1) % self.capacity

        self.prev_states[exp_idx] = self.frame_window
        self.frame_window = np.append(self.frame_window[1:], frame_idx)
        self.next_states[exp_idx] = self.frame_window
        self.actions[exp_idx] = action
        self.is_done[exp_idx] = done
        self.frames[frame_idx] = frame
        self.rewards[exp_idx] = reward
        if done:
            self.expecting_new_episode = True

    def start_new_episode(self, frame):
        # it should be okay not to increment counter here
        # because episode ending frames are not used
        assert self.expecting_new_episode, "previous episode didn't end yet"
        frame_idx = self.counter % self.max_frame_cache
        self.frame_window = np.repeat(frame_idx, self.num_frame_stack)
        self.frames[frame_idx] = frame
        self.expecting_new_episode = False

    def sample_mini_batch(self, n):
        count = min(self.capacity, self.counter)
        batchidx = np.random.randint(count, size=n)

        prev_frames = self.frames[self.prev_states[batchidx]]
        next_frames = self.frames[self.next_states[batchidx]]

        return {
            "reward": self.rewards[batchidx],
            "prev_state": prev_frames,
            "next_state": next_frames,
            "actions": self.actions[batchidx],
            "done_mask": self.is_done[batchidx]
        }

    def current_state(self):
        # assert not self.expecting_new_episode, "start new episode first"'
        assert self.frame_window is not None, "do something first"
        return self.frames[self.frame_window]

    def init_caches(self):
        self.rewards = np.zeros(self.capacity, dtype="float32")
        self.prev_states = -np.ones((self.capacity, self.num_frame_stack),
            dtype="int32")
        self.next_states = -np.ones((self.capacity, self.num_frame_stack),
            dtype="int32")
        self.is_done = -np.ones(self.capacity, "int32")
        self.actions = -np.ones(self.capacity, dtype="int32")

        # lazy to think how big is the smallest possible number. At least this is big enough
        self.max_frame_cache = self.capacity + 2 * self.num_frame_stack + 1
        self.frames = -np.ones((self.max_frame_cache,) + self.pic_size, dtype="float32")
class DQN_2:
    """
    General DQN agent.
    Can be applied to any standard environment
    The implementation follows:
    Mnih et. al - Playing Atari with Deep Reinforcement Learning https://arxiv.org/pdf/1312.5602.pdf
    The q-network structure is different from the original paper
    see also:
    David Silver's RL course lecture 6: https://www.youtube.com/watch?v=UoPei5o4fps&t=1s
    """

    def __init__(self,
            env,
            batchsize=64,
            pic_size=(96, 96),
            num_frame_stack=4,
            gamma=0.95,
            frame_skip=1,
            train_freq=4,
            initial_epsilon=1.0,
            min_epsilon=0.1,
            render=True,
            epsilon_decay_steps=int(1e6),
            min_experience_size=int(1e3),
            experience_capacity=int(1e5),
            network_update_freq=5000,
            regularization = 1e-6,
            optimizer_params = None,
            action_map=None
    ):
        self.exp_history = ExperienceHistory(
            num_frame_stack,
            capacity=experience_capacity,
            pic_size=pic_size
        )

        # in playing mode we don't store the experience to agent history
        # but this cache is still needed to get the current frame stack
        self.playing_cache = ExperienceHistory(
            num_frame_stack,
            capacity=num_frame_stack * 5 + 10,
            pic_size=pic_size
        )

        if action_map is not None:
            self.dim_actions = len(action_map)
        else:
            self.dim_actions = env.action_space.n

        self.network_update_freq = network_update_freq
        self.action_map = action_map
        self.env = env
        self.batchsize = batchsize
        self.num_frame_stack = num_frame_stack
        self.gamma = gamma
        self.frame_skip = frame_skip
        self.train_freq = train_freq
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.render = render
        self.min_experience_size = min_experience_size
        self.pic_size = pic_size
        self.regularization = regularization
        # These default magic values always work with Adam
        self.optimizer_params = optimizer_params or dict(learning_rate=0.0004, epsilon=1e-7)

        self.do_training = True
        self.playing_epsilon = 0.0
        self.session = None

        self.state_size = (self.num_frame_stack,) + self.pic_size
        self.global_counter = 0
        self.episode_counter = 0

    @staticmethod
    def process_image(img):
        return 2 * color.rgb2gray(transform.rescale(img[34:194], 0.5)) - 1

    def build_graph(self):
        input_dim_with_batch = (self.batchsize, self.num_frame_stack) + self.pic_size
        input_dim_general = (None, self.num_frame_stack) + self.pic_size

        self.input_prev_state = tf.compat.v1.placeholder(tf.float32, input_dim_general, "prev_state")
        self.input_next_state = tf.compat.v1.placeholder(tf.float32, input_dim_with_batch, "next_state")
        self.input_reward = tf.compat.v1.placeholder(tf.float32, self.batchsize, "reward")
        self.input_actions = tf.compat.v1.placeholder(tf.int32, self.batchsize, "actions")
        self.input_done_mask = tf.compat.v1.placeholder(tf.int32, self.batchsize, "done_mask")

        
        qsa_targets = self.create_network(self.input_next_state, trainable=False)

        qsa_estimates = self.create_network(self.input_prev_state, trainable=True)

        self.best_action = tf.argmax(qsa_estimates, axis=1)

        not_done = tf.cast(tf.logical_not(tf.cast(self.input_done_mask, "bool")), "float32")
        q_target = tf.reduce_max(qsa_targets, -1) * self.gamma * not_done + self.input_reward
        # select the chosen action from each row
        # in numpy this is qsa_estimates[range(batchsize), self.input_actions]
        action_slice = tf.stack([tf.range(0, self.batchsize), self.input_actions], axis=1)
        q_estimates_for_input_action = tf.gather_nd(qsa_estimates, action_slice)

        training_loss = tf.nn.l2_loss(q_target - q_estimates_for_input_action) / self.batchsize

        optimizer = tf.compat.v1.train.AdamOptimizer(**(self.optimizer_params))

        reg_loss = tf.add_n(tf.compat.v1.losses.get_regularization_losses())
        self.train_op = optimizer.minimize(reg_loss + training_loss)

        train_params = self.get_variables("train")
        fixed_params = self.get_variables("fixed")

        assert (len(train_params) == len(fixed_params))
        self.copy_network_ops = [tf.compat.v1.assign(fixed_v, train_v)
            for train_v, fixed_v in zip(train_params, fixed_params)]

    def get_variables(self, scope):
        vars = [t for t in tf.compat.v1.global_variables()
            if "%s/" % scope in t.name and "Adam" not in t.name]
        return sorted(vars, key=lambda v: v.name)

    def create_network(self, input, trainable):
        if trainable:
            wr = slim.l2_regularizer(self.regularization)
        else:
            wr = None

        # the input is stack of black and white frames.
        # put the stack in the place of channel (last in tf)
        input_t = tf.transpose(input, [0, 2, 3, 1])

        net = slim.conv2d(input_t, 8, (7, 7), data_format="NHWC",
            activation_fn=tf.nn.relu, stride=3, weights_regularizer=wr, trainable=trainable)
        net = slim.max_pool2d(net, 2, 2)
        net = slim.conv2d(net, 16, (3, 3), data_format="NHWC",
            activation_fn=tf.nn.relu, weights_regularizer=wr, trainable=trainable)
        net = slim.max_pool2d(net, 2, 2)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu,
            weights_regularizer=wr, trainable=trainable)
        q_state_action_values = slim.fully_connected(net, self.dim_actions,
            activation_fn=None, weights_regularizer=wr, trainable=trainable)

        return q_state_action_values

    def check_early_stop(self, reward, totalreward):
        return False, 0.0

    def get_random_action(self):
        return np.random.choice(self.dim_actions)

    def get_epsilon(self):
        if not self.do_training:
            return self.playing_epsilon
        elif self.global_counter >= self.epsilon_decay_steps:
            return self.min_epsilon
        else:
            # linear decay
            r = 1.0 - self.global_counter / float(self.epsilon_decay_steps)
            return self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * r

    def train(self):
        batch = self.exp_history.sample_mini_batch(self.batchsize)

        fd = {
            self.input_reward: "reward",
            self.input_prev_state: "prev_state",
            self.input_next_state: "next_state",
            self.input_actions: "actions",
            self.input_done_mask: "done_mask"
        }
        fd1 = {ph: batch[k] for ph, k in fd.items()}
        self.session.run([self.train_op], fd1)

    def play_episode(self):
        eh = (
            self.exp_history if self.do_training
            else self.playing_cache
        )
        total_reward = 0
        frames_in_episode = 0

        first_frame = self.env.reset()
        first_frame_pp = self.process_image(first_frame)

        eh.start_new_episode(first_frame_pp)

        while True:
            if np.random.rand() > self.get_epsilon():
                action_idx = self.session.run(
                    self.best_action,
                    {self.input_prev_state: eh.current_state()[np.newaxis, ...]}
                )[0]
            else:
                action_idx = self.get_random_action()

            if self.action_map is not None:
                action = self.action_map[action_idx]
            else:
                action = action_idx

            reward = 0
            for _ in range(self.frame_skip):
                observation, r, done, info = self.env.step(action)
                if self.render:
                    self.env.render()
                reward += r
                if done:
                    break

            early_done, punishment = self.check_early_stop(reward, total_reward)
            if early_done:
                reward += punishment

            done = done or early_done

            total_reward += reward
            frames_in_episode += 1

            eh.add_experience(self.process_image(observation), action_idx, done, reward)

            if self.do_training:
                self.global_counter += 1
                if self.global_counter % self.network_update_freq:
                    self.update_target_network()
                train_cond = (
                    self.exp_history.counter >= self.min_experience_size and
                    self.global_counter % self.train_freq == 0
                )
                if train_cond:
                    self.train()

            if done:
                if self.do_training:
                    self.episode_counter += 1

                return total_reward, frames_in_episode

    def update_target_network(self):
        self.session.run(self.copy_network_ops)


class CarRacingDQN(DQN):
    """
    CarRacing specifig part of the DQN-agent
    Some minor env-specifig tweaks but overall
    assumes very little knowledge from the environment
    """

    def __init__(self, max_negative_rewards=100, **kwargs):
        all_actions = np.array(
            [k for k in it.product([-1, 0, 1], [1, 0], [0.2, 0])]
        )
        # car racing env gives wrong pictures without render
        kwargs["render"] = True
        super().__init__(
            action_map=all_actions,
            pic_size=(96, 96),
            **kwargs
        )

        self.gas_actions = np.array([a[1] == 1 and a[2] == 0 for a in all_actions])
        self.break_actions = np.array([a[2] == 1 for a in all_actions])
        self.n_gas_actions = self.gas_actions.sum()
        self.neg_reward_counter = 0
        self.max_neg_rewards = max_negative_rewards

    @staticmethod
    def process_image(obs):
        return 2 * color.rgb2gray(obs) - 1.0

    def get_random_action(self):
        """
        Here random actions prefer gas to break
        otherwise the car can never go anywhere.
        """
        action_weights = 14.0 * self.gas_actions + 1.0
        action_weights /= np.sum(action_weights)

        return np.random.choice(self.dim_actions, p=action_weights)

    def check_early_stop(self, reward, totalreward):
        if reward < 0:
            self.neg_reward_counter += 1
            done = (self.neg_reward_counter > self.max_neg_rewards)

            if done and totalreward <= 500:
                punishment = -20.0
            else:
                punishment = 0.0
            if done:
                self.neg_reward_counter = 0

            return done, punishment
        else:
            self.neg_reward_counter = 0
            return False, 0.0
