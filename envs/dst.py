from contextlib import suppress
import gym
from gym.spaces import Discrete, Box
import numpy as np
np.set_printoptions(precision=3, suppress=True)

class DeepSeaTreasure(gym.Env):

    def __init__(self, weight: np.array = np.array([0.5,0.5]), state_as_float=False):
        # the map of the deep sea treasure (convex version)
        # CCS: [1,0], [0.7,0.3], [0.67,0.33], [0.6,0.4], [0.56,0.44], [0.52,0.48], [0.5,0.5], [0.4,0.6], [0.3,0.7], [0, 1]
        self.state_as_float = state_as_float

        self.sea_map = np.array(
            [[0,    0,    0,   0,   0,  0,   0,   0,   0,   0,   0],
             [0.7,  0,    0,   0,   0,  0,   0,   0,   0,   0,   0],
             [-10,  8.2,  0,   0,   0,  0,   0,   0,   0,   0,   0],
             [-10, -10, 11.5,  0,   0,  0,   0,   0,   0,   0,   0],
             [-10, -10, -10, 14.0, 15.1,16.1,0,   0,   0,   0,   0],
             [-10, -10, -10, -10, -10, -10,  0,   0,   0,   0,   0],
             [-10, -10, -10, -10, -10, -10,  0,   0,   0,   0,   0],
             [-10, -10, -10, -10, -10, -10, 19.6, 20.3,0,   0,   0],
             [-10, -10, -10, -10, -10, -10, -10, -10,  0,   0,   0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 22.4, 0,   0],
             [-10, -10, -10, -10, -10, -10, -10, -10, -10, 23.7, 0]]
        )
        self.dir = {
            0: np.array([-1, 0], dtype=np.int32),  # up
            1: np.array([1, 0], dtype=np.int32),  # down
            2: np.array([0, -1], dtype=np.int32),  # left
            3: np.array([0, 1], dtype=np.int32)  # right
        }

        # DON'T normalize
        self.w = weight

        # state space specification: 2-dimensional discrete box
        self.state_spec = [['discrete', 1, [0, 10]], ['discrete', 1, [0, 10]]]
        self.observation_space = Box(np.zeros(2), np.ones(2))

        # action space specification: 1 dimension, 0 up, 1 down, 2 left, 3 right
        self.action_space = Discrete(4)

        self.current_state = np.array([0, 0], dtype=np.int32)

    def get_map_value(self, pos):
        return self.sea_map[pos[0]][pos[1]]
    
    def render(self, mode=None):
        pass

    def reset(self):
        '''
            reset the location of the submarine
        '''
        self.current_state = np.array([0, 0], dtype=np.int32)
        self.step_count = 0.0
        if self.state_as_float:
            state = self.current_state.astype(np.float32) * 0.1
        else:
            state = self.current_state.copy()
        return state

    def step(self, action):
        '''
            step one move and feed back reward
        '''
        next_state = self.current_state + self.dir[action]

        valid = lambda x, ind: (x[ind] >= self.state_spec[ind][2][0]) and (x[ind] <= self.state_spec[ind][2][1])

        if valid(next_state, 0) and valid(next_state, 1):
            if self.get_map_value(next_state) != -10:
                self.current_state = next_state

        treasure_value = self.get_map_value(self.current_state)
        if treasure_value == 0 or treasure_value == -10:
            treasure_value = 0.0
            terminal = False
        else:
            terminal = True
        time_penalty = -1.0
        vec_reward = np.array([treasure_value, time_penalty], dtype=np.float32)
        scalar_reward = np.dot(vec_reward, self.w)

        if self.state_as_float:
            state = self.current_state.astype(np.float32) * 0.1
        else:
            state = self.current_state.copy()

        return state, scalar_reward, terminal, {'phi': vec_reward}