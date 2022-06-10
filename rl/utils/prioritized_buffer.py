import numpy as np
import torch as th


# Code adapted from https://github.com/sfujim/LAP-PAL
class SumTree(object):

    def __init__(self, max_size):
        self.nodes = []
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        for _ in range(int(np.ceil(np.log2(max_size))) + 1):
            nodes = np.zeros(level_size)
            self.nodes.append(nodes)
            level_size *= 2

    # Batch binary search through sum tree
    # Sample a priority between 0 and the max priority
    # and then search the tree for the corresponding index
    def sample(self, batch_size):
        query_value = np.random.uniform(0, self.nodes[0][0], size=batch_size)
        node_index = np.zeros(batch_size, dtype=int)

        for nodes in self.nodes[1:]:
            node_index *= 2
            left_sum = nodes[node_index]

            is_greater = np.greater(query_value, left_sum)
            # If query_value > left_sum -> go right (+1), else go left (+0)
            node_index += is_greater
            # If we go right, we only need to consider the values in the right tree
            # so we subtract the sum of values in the left tree
            query_value -= left_sum * is_greater
        
        return node_index

    def set(self, node_index, new_priority):
        priority_diff = new_priority - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2

    def batch_set(self, node_index, new_priority):
        # Confirm we don't increment a node twice
        node_index, unique_index = np.unique(node_index, return_index=True)
        priority_diff = new_priority[unique_index] - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2

class PrioritizedReplayBuffer:

    def __init__(self, obs_dim, action_dim, rew_dim=1, max_size=100000, obs_dtype=np.float32, action_dtype=np.float32):
        self.max_size = max_size
        self.ptr, self.size, = 0, 0

        self.obs = np.zeros((max_size, obs_dim), dtype=obs_dtype)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=obs_dtype)
        self.actions = np.zeros((max_size, action_dim), dtype=action_dtype)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

        self.tree = SumTree(max_size)
        self.max_priority = 0.1
        
    def add(self, obs, action, reward, next_obs, done, priority=None):
        self.obs[self.ptr] = np.array(obs).copy()
        self.next_obs[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.dones[self.ptr] = np.array(done).copy()

        self.tree.set(self.ptr, self.max_priority if priority is None else priority)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, to_tensor=False, device=None):
        idxes = self.tree.sample(batch_size)

        experience_tuples = (self.obs[idxes], self.actions[idxes], self.rewards[idxes], self.next_obs[idxes], self.dones[idxes])
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples)) + (idxes,) #, weights)
        else:
            return experience_tuples + (idxes,)
    
    def sample_obs(self, batch_size, to_tensor=False, device=None):
        idxes = self.tree.sample(batch_size)
        if to_tensor:
            return th.tensor(self.obs[idxes]).to(device)
        else:
            return self.obs[idxes]

    def update_priorities(self, idxes, priorities):
        self.max_priority = max(self.max_priority, priorities.max())
        self.tree.batch_set(idxes, priorities)
    
    """ def update_priority(self, ind, priority):
        self.max_priority = max(self.max_priority, priority)
        self.tree.set(ind, priority) """

    def get_all_data(self, max_samples=None, to_tensor=False, device=None):
        if max_samples is not None:
            inds = np.random.choice(self.size, min(max_samples, self.size), replace=False)
        else:
            inds = np.arange(self.size)
        tuples = (self.obs[inds], self.actions[inds], self.rewards[inds], self.next_obs[inds], self.dones[inds])
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(device), tuples))
        else:
            return tuples
        
    def __len__(self):
        return self.size


class MOPrioritizedReplayBuffer:

    def __init__(self, obs_dim, action_dim, alpha=0.6, rew_dim=1, max_size=100000):
        self.max_size = max_size
        self.ptr, self.size, = 0, 0

        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        self.w = np.zeros((max_size, rew_dim), dtype=np.float32)

        it_capacity = 1
        while it_capacity < self.max_size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self._alpha = alpha
        
    def add(self, obs, action, reward, next_obs, done, w):
        self._it_sum[self.ptr] = self._max_priority ** self._alpha
        self._it_min[self.ptr] = self._max_priority ** self._alpha

        self.obs[self.ptr] = np.array(obs).copy()
        self.next_obs[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.dones[self.ptr] = np.array(done).copy()
        self.w[self.ptr] = np.array(w).copy()
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, self.size) # -1???
        # TODO(szymon): should we ensure no repeats?
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size, beta=0.4, to_tensor=False, device=None):
        idxes = self._sample_proportional(batch_size)
        #weights = []
        #p_min = self._it_min.min() / self._it_sum.sum()
        #max_weight = (p_min * self.size) ** (-beta)
        #p_sample = self._it_sum[idxes] / self._it_sum.sum()
        #weights = (p_sample * self.size) ** (-beta) / max_weight

        experience_tuples = (self.obs[idxes], self.actions[idxes], self.rewards[idxes], self.next_obs[idxes], self.dones[idxes])
        if to_tensor:
            bla = tuple(map(lambda x: th.tensor(x).to(device), experience_tuples)) + (idxes,) #, weights)
            return bla
        else:
            return experience_tuples + (idxes,)

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) >= 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < self.size
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha
        self._max_priority = max(self._max_priority, np.max(priorities))

    def __len__(self):
        return self.size