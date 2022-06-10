from typing import Union, Callable
import numpy as np
import torch as th
import wandb
from copy import deepcopy
from rl.rl_algorithm import RLAlgorithm


class GPI(RLAlgorithm):

    def __init__(self,
                 env,
                 algorithm_constructor: Callable,
                 log: bool = True,
                 project_name: str = 'gpi',
                 experiment_name: str = 'gpi',
                 device: Union[th.device, str] = 'auto'):
        super(GPI, self).__init__(env, device)

        self.algorithm_constructor = algorithm_constructor
        self.policies = []
        self.tasks = []

        self.log = log
        if self.log:
            self.setup_wandb(project_name, experiment_name)

    def eval(self, obs, w, return_policy_index=False, exclude=None) -> int:
        if not hasattr(self.policies[0], 'q_table'):
            if isinstance(obs, np.ndarray):
                obs = th.tensor(obs).float().to(self.device)
                w = th.tensor(w).float().to(self.device)
            q_vals = th.stack([policy.q_values(obs, w) for policy in self.policies])
            max_q, a = th.max(q_vals, dim=2)
            policy_index = th.argmax(max_q)
            if return_policy_index:
                return a[policy_index].detach().long().item(), policy_index.item()
            return a[policy_index].detach().long().item()
        else:
            q_vals = np.stack([policy.q_values(obs, w) for policy in self.policies if policy is not exclude])
            policy_index, action = np.unravel_index(np.argmax(q_vals), q_vals.shape)
            if return_policy_index:
                return action, policy_index
            return action
     
    def max_q(self, obs, w, tensor=False, exclude=None):
        if tensor:
            with th.no_grad():
                psi_values = th.stack([policy.target_psi_net(obs) for policy in self.policies if policy is not exclude])
                q_values = th.einsum('r,psar->psa', w, psi_values)
                max_q, a = th.max(q_values, dim=2)
                polices = th.argmax(max_q, dim=0)
                max_acts = a.gather(0, polices.unsqueeze(0)).squeeze(0)
                psi_i = psi_values.gather(0, polices.reshape(1,-1,1,1).expand(1, psi_values.size(1), psi_values.size(2), psi_values.size(3))).squeeze(0)
                max_psis = psi_i.gather(1, max_acts.reshape(-1,1,1).expand(psi_i.size(0), 1, psi_i.size(2))).squeeze(1)
                return max_psis
        else:
            q_vals = np.stack([policy.q_values(obs, w) for policy in self.policies])
            policy_ind, action = np.unravel_index(np.argmax(q_vals), q_vals.shape)
            return self.policies[policy_ind].q_table[tuple(obs)][action]
    
    def delete_policies(self, delete_indx):
        for i in sorted(delete_indx, reverse=True):
            self.policies.pop(i)
            self.tasks.pop(i)

    def learn(self, w, total_timesteps, total_episodes=None, reset_num_timesteps=False, eval_env=None, eval_freq=1000, use_gpi=True, reset_learning_starts=True, new_policy=True, reuse_value_ind=None):
        if new_policy:
            new_policy = self.algorithm_constructor()
            self.policies.append(new_policy)
        self.tasks.append(w)
        
        self.policies[-1].gpi = self if use_gpi else None

        if self.log:
            self.policies[-1].log = self.log
            self.policies[-1].writer = self.writer
            wandb.config.update(self.policies[-1].get_config())

        if len(self.policies) > 1:
            self.policies[-1].num_timesteps = self.policies[-2].num_timesteps
            self.policies[-1].num_episodes = self.policies[-2].num_episodes
            if reset_learning_starts:
                self.policies[-1].learning_starts = self.policies[-2].num_timesteps  # to reset exploration schedule

            if reuse_value_ind is not None:
                if hasattr(self.policies[-1], 'q_table'):
                    self.policies[-1].q_table = deepcopy(self.policies[reuse_value_ind].q_table)
                else:
                    self.policies[-1].psi_net.load_state_dict(self.policies[reuse_value_ind].psi_net.state_dict())
                    self.policies[-1].target_psi_net.load_state_dict(self.policies[reuse_value_ind].psi_net.state_dict())

            self.policies[-1].replay_buffer = self.policies[-2].replay_buffer

        self.policies[-1].learn(w=w,
                                total_timesteps=total_timesteps, 
                                total_episodes=total_episodes,
                                reset_num_timesteps=reset_num_timesteps,
                                eval_env=eval_env,
                                eval_freq=eval_freq)

    @property
    def gamma(self):
        return self.policies[0].gamma

    def train(self):
        pass

    def get_config(self) -> dict:
        if len(self.policies) > 0:
            return self.policies[0].get_config()
        return {}
