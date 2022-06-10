import numpy as np
import random
from rl.utils.utils import eval_mo, linearly_decaying_epsilon
from rl.rl_algorithm import RLAlgorithm
from rl.successor_features.gpi import GPI
from rl.utils.buffer import ReplayBuffer
from rl.utils.prioritized_buffer import PrioritizedReplayBuffer
import wandb as wb 
import matplotlib.pyplot as plt
import seaborn as sns


class SF(RLAlgorithm):

    def __init__(self,
                env,
                alpha: float = 0.01, 
                gamma: float = 0.99,
                initial_epsilon: float = 0.01,
                final_epsilon: float = 0.01,
                epsilon_decay_steps: int = None,
                learning_starts: int = 0,
                use_replay: bool = False,
                buffer_size: int = 10000,
                batch_size: int = 5,
                per: bool = False,
                min_priority: float = 0.01,
                gpi: GPI = None,
                use_gpi: bool = False,
                envelope: bool = False,
                project_name: str = 'sf',
                experiment_name: str = 'sf',
                log: bool = False):

        super().__init__(env, device=None)
        self.phi_dim = len(env.unwrapped.w)
        self.alpha = alpha
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.learning_starts = learning_starts
        self.gpi = gpi
        self.use_gpi = use_gpi
        self.use_replay = use_replay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.per = per
        self.min_priority = min_priority
        self.envelope = envelope

        self.q_table = dict()
        if self.use_replay:
            if self.per:
                self.replay_buffer = PrioritizedReplayBuffer(self.observation_dim, 1, rew_dim=self.phi_dim, max_size=buffer_size, obs_dtype=np.int32, action_dtype=np.int32)
            else:
                self.replay_buffer = ReplayBuffer(self.observation_dim, 1, rew_dim=self.phi_dim, max_size=buffer_size, obs_dtype=np.int32, action_dtype=np.int32)
        else:
            self.replay_buffer = None

        self.log = log
        if self.log:
            self.setup_wandb(project_name, experiment_name)   

    def act(self, obs: np.array, w: np.array):
        np_obs = obs
        obs = tuple(obs)
        if obs not in self.q_table:
            self.q_table[obs] = np.zeros((self.action_dim, self.phi_dim))
        self.policy_index = None
        if np.random.rand() < self.epsilon:
            return int(self.env.action_space.sample())
        else:
            if self.gpi is not None and self.use_gpi:
                action, self.policy_index = self.gpi.eval(np_obs, w, return_policy_index=True)
                return action
            else:
                return np.argmax(np.dot(self.q_table[obs], w))
    
    def eval(self, obs: np.array, w: np.array) -> int:
        if self.gpi is not None and self.use_gpi:
            return self.gpi.eval(obs, w)
        else:
            obs = tuple(obs)
            if obs not in self.q_table:
                return int(self.env.action_space.sample())
            return np.argmax(np.dot(self.q_table[obs], w))
    
    def q_values(self, obs: np.array, w: np.array) -> np.array:
        obs = tuple(obs)
        if obs not in self.q_table:
            self.q_table[obs] = np.zeros((self.action_dim, self.phi_dim))
        return np.dot(self.q_table[obs], w)

    def train(self, w: np.array):
        obs = tuple(self.obs)
        next_obs = tuple(self.next_obs)
        if next_obs not in self.q_table:
            self.q_table[next_obs] = np.zeros((self.action_dim, self.phi_dim))

        if self.gpi is not None:
            if self.envelope:
                max_q = self.gpi.max_q(self.next_obs, w)
            else:
                max_q = self.q_table[next_obs][self.gpi.eval(self.next_obs, w)]
        else:
            max_q = self.q_table[next_obs][np.argmax(np.dot(self.q_table[next_obs], w))]
        td_error =  self.reward + (1-self.terminal)*self.gamma*max_q - self.q_table[obs][self.action]
        self.q_table[obs][self.action] += self.alpha * td_error

        # Update other policies
        if self.gpi is not None:
            if self.policy_index is not None and self.policy_index != len(self.gpi.policies) - 1:
                i, pi = self.policy_index, self.gpi.policies[self.policy_index]
                if next_obs not in pi.q_table:
                    pi.q_table[next_obs] = np.zeros((self.action_dim, self.phi_dim))
                pi_w = self.gpi.tasks[i]
                if self.envelope:
                    pi_max_q = self.gpi.max_q(self.next_obs, pi_w)
                else:
                    pi_max_q = pi.q_table[next_obs][self.gpi.eval(self.next_obs, pi_w)]
                pi_td_error = self.reward + (1-self.terminal)*self.gamma*pi_max_q - pi.q_table[obs][self.action]
                pi.q_table[obs][self.action] += self.alpha * pi_td_error

        if self.use_replay:
            if self.per:
                priority = np.dot(td_error, w)
                self.replay_buffer.add(self.obs, self.action, self.reward, self.next_obs, self.terminal, max(priority, self.min_priority)**1.0)
            else:
                self.replay_buffer.add(self.obs, self.action, self.reward, self.next_obs, self.terminal)

            if self.per:
                samp_s, samp_a, samp_r, samp_next_s, samp_terminal, idxes = self.replay_buffer.sample(self.batch_size)
                new_priorities = np.zeros(shape=idxes.shape)
            else:
                samp_s, samp_a, samp_r, samp_next_s, samp_terminal = self.replay_buffer.sample(self.batch_size)

            for i in range(self.batch_size):
                s = tuple(samp_s[i])
                if s not in self.q_table:
                    self.q_table[s] = np.zeros((self.action_dim, self.phi_dim))

                a, r, next_s, terminal = int(samp_a[i]), samp_r[i], tuple(samp_next_s[i]), samp_terminal[i]
                
                if next_s not in self.q_table:
                    self.q_table[next_s] = np.zeros((self.action_dim, self.phi_dim))

                if self.gpi is not None:
                    if self.envelope:
                        max_q = self.gpi.max_q(np.array(next_s), w)
                    else:
                        max_q = self.q_table[next_s][self.gpi.eval(np.array(next_s), w)]
                else:
                    max_q = self.q_table[next_s][np.argmax(np.dot(self.q_table[next_s], w))]
                td_err =  r + (1-terminal)*self.gamma*max_q - self.q_table[s][a]
                self.q_table[s][a] += self.alpha * td_err
                
                if self.per:
                    priority = np.dot(td_err, w)
                    new_priorities[i] = priority
                    
            if self.per:
                new_priorities = new_priorities.clip(min=self.min_priority)**1.0
                self.replay_buffer.update_priorities(idxes, new_priorities)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_epsilon(self.initial_epsilon, self.epsilon_decay_steps, self.num_timesteps, self.learning_starts, self.final_epsilon)
        
        if self.log and self.num_timesteps % 1000 == 0:
            self.writer.add_scalar("losses/td_error", np.dot(td_error, w), self.num_timesteps)
            self.writer.add_scalar("metrics/epsilon", self.epsilon, self.num_timesteps)
            if self.per:
                self.writer.add_scalar("metrics/mean_priority", new_priorities.mean(), self.num_timesteps)

    def get_config(self) -> dict:
        return {'alpha': self.alpha,
                'gamma': self.gamma,
                'initial_epsilon': self.initial_epsilon,
                'final_epsilon': self.final_epsilon,
                'epsilon_decay_steps': self.epsilon_decay_steps,
                'use_replay': self.use_replay,
                'buffer_size': self.buffer_size,
                'batch_size': self.batch_size,
                'per' : self.per,
                }
    
    def learn(self, total_timesteps, total_episodes=None, reset_num_timesteps=True, eval_env=None, eval_freq=1000, w=np.array([1.0,0.0])):
        episode_reward = 0.0
        episode_vec_reward = np.zeros(w.shape[0])
        num_episodes = 0
        self.obs, done = self.env.reset(), False

        self.env.unwrapped.w = w

        self.num_timesteps = 0 if reset_num_timesteps else self.num_timesteps
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes
        for _ in range(1, total_timesteps+1):
            if total_episodes is not None and num_episodes == total_episodes:
                break

            self.num_timesteps += 1

            self.action = self.act(self.obs, w)
            self.next_obs, reward, done, info = self.env.step(self.action)
            self.reward = info['phi'] # vectorized reward
            self.terminal = done if 'TimeLimit.truncated' not in info else not info['TimeLimit.truncated']

            self.train(w)

            if eval_env is not None and self.log and self.num_timesteps % eval_freq == 0:
                total_reward, discounted_return, total_vec_r, total_vec_return = eval_mo(self, eval_env, w)
                self.writer.add_scalar("eval/total_reward", total_reward, self.num_timesteps)
                self.writer.add_scalar("eval/discounted_return", discounted_return, self.num_timesteps)
                for i in range(episode_vec_reward.shape[0]):
                    self.writer.add_scalar(f"eval/total_reward_obj{i}", total_vec_r[i], self.num_timesteps)
                    self.writer.add_scalar(f"eval/return_obj{i}", total_vec_return[i], self.num_timesteps)

            episode_reward += reward
            episode_vec_reward += info['phi']
            if done:
                self.obs, done = self.env.reset(), False
                num_episodes += 1
                self.num_episodes += 1
                
                if num_episodes % 1000 == 0:
                    print(f"Episode: {self.num_episodes} Step: {self.num_timesteps}, Ep. Total Reward: {episode_reward}, {episode_vec_reward}")
                if self.log:
                    self.writer.add_scalar("metrics/episode", self.num_episodes, self.num_timesteps)
                    self.writer.add_scalar("metrics/episode_reward", episode_reward, self.num_timesteps)
                    for i in range(episode_vec_reward.shape[0]):
                        self.writer.add_scalar(f"metrics/episode_reward_obj{i}", episode_vec_reward[i], self.num_timesteps)

                episode_reward = 0.0
                episode_vec_reward = np.zeros(w.shape[0])
            else:
                self.obs = self.next_obs
