# -*- coding: UTF-8 -*- 
# Code from: https://github.com/mike-gimelfarb/deep-successor-features-for-transfer/blob/main/source/tasks/gridworld.py
import numpy as np
import random
import gym
from gym.spaces import Discrete, Box

MAZE=np.array([
    ['1', ' ', ' ', ' ', ' ', '2', 'X', ' ', ' ', ' ', ' ', ' ', 'G'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', '1', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['2', ' ', ' ', ' ', ' ', '3', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['X', 'X', '3', ' ', 'X', 'X', 'X', 'X', 'X', ' ', '1', 'X', 'X'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', '2', ' ', ' ', ' ', ' ', '3'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', '2', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['_', ' ', ' ', ' ', ' ', ' ', 'X', '3', ' ', ' ', ' ', ' ', '1']])

class Shapes(gym.Env):
    """
    A discretized version of the gridworld environment introduced in [1]. 
    The gridworld is split into four rooms separated by walls with passage-ways.
    
    References
    ----------
    [1] Barreto, André, et al. "Successor Features for Transfer in Reinforcement Learning." NIPS. 2017.
    """

    LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3
 
    def __init__(self, maze=MAZE):
        """
        Creates a new instance of the shapes environment.
        
        Parameters
        ----------
        maze : np.ndarray
            an array of string values representing the type of each cell in the environment:
                G indicates a goal state (terminal state)
                _ indicates an initial state (there can be multiple, and one is selected at random
                    at the start of each episode)
                X indicates a barrier 
                0, 1, .... 9 indicates the type of shape to be placed in the corresponding cell
                entries containing other characters are treated as regular empty cells
        shape_rewards : dict
            a dictionary mapping the type of shape (0, 1, ... ) to a corresponding reward to provide
            to the agent for collecting an object of that type
        """
        self.height, self.width = maze.shape
        self.maze = maze
        #self.shape_rewards = shape_rewards
        shape_types = ['1', '2', '3']  # sorted(list(shape_rewards.keys()))
        self.all_shapes = dict(zip(shape_types, range(len(shape_types))))
        
        self.goal = None
        self.initial = []
        self.occupied = set()
        self.shape_ids = dict()
        for c in range(self.width):
            for r in range(self.height):
                if maze[r, c] == 'G':
                    self.goal = (r, c)
                elif maze[r, c] == '_':
                    self.initial.append((r, c))
                elif maze[r, c] == 'X':
                    self.occupied.add((r, c))
                elif maze[r, c] in {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}:
                    self.shape_ids[(r, c)] = len(self.shape_ids)
        
        self.w = np.zeros(3)
        self.action_space = Discrete(4)
        self.observation_space = Box(low=np.zeros(2+len(self.shape_ids)), high=np.ones(2+len(self.shape_ids)))

    def state_to_array(self, state):
        s = [element for tupl in state for element in tupl]
        return np.array(s, dtype=np.int32)

    def reset(self):
        self.state = (random.choice(self.initial), tuple(0 for _ in range(len(self.shape_ids))))
        return self.state_to_array(self.state)
    
    def step(self, action): 
        old_state = self.state
        (row, col), collected = self.state
        
        # perform the movement
        if action == Shapes.LEFT: 
            col -= 1
        elif action == Shapes.UP: 
            row -= 1
        elif action == Shapes.RIGHT: 
            col += 1
        elif action == Shapes.DOWN: 
            row += 1
        else:
            raise Exception('bad action {}'.format(action))
        
        # out of bounds, cannot move
        if col < 0 or col >= self.width or row < 0 or row >= self.height:
            return self.state_to_array(self.state), 0., False, {'phi': np.zeros(len(self.all_shapes), dtype=np.float32)}

        # into a blocked cell, cannot move
        s1 = (row, col)
        if s1 in self.occupied:
            return self.state_to_array(self.state), 0., False, {'phi': np.zeros(len(self.all_shapes), dtype=np.float32)}
        
        # can now move
        self.state = (s1, collected)
        
        # into a goal cell
        if s1 == self.goal:
            phi = np.ones(len(self.all_shapes), dtype=np.float32)
            return self.state_to_array(self.state), 1., True, {'phi': phi}
        
        # into a shape cell
        if s1 in self.shape_ids:
            shape_id = self.shape_ids[s1]
            if collected[shape_id] == 1:
                # already collected this flag
                return self.state_to_array(self.state), 0., False, {'phi': np.zeros(len(self.all_shapes), dtype=np.float32)}
            else:
                # collect the new flag
                collected = list(collected)
                collected[shape_id] = 1
                collected = tuple(collected)
                self.state = (s1, collected)
                phi = self.features(old_state, action, self.state)
                reward = np.dot(phi, self.w)
                return self.state_to_array(self.state), reward, False, {'phi': phi}
        
        # into an empty cell
        return self.state_to_array(self.state), 0., False, {'phi': np.zeros(len(self.all_shapes), dtype=np.float32)}

    # ===========================================================================
    # STATE ENCODING FOR DEEP LEARNING
    # ===========================================================================
    def encode(self, state):
        (y, x), coll = state
        n_state = self.width + self.height
        result = np.zeros((n_state + len(coll),))
        result[y] = 1
        result[self.height + x] = 1
        result[n_state:] = np.array(coll)
        result = result.reshape((1, -1))
        return result
    
    def encode_dim(self):
        return self.width + self.height + len(self.shape_ids)
        
    # ===========================================================================
    # SUCCESSOR FEATURES
    # ===========================================================================
    def features(self, state, action, next_state):
        s1, _ = next_state
        _, collected = state
        nc = len(self.all_shapes)
        phi = np.zeros(nc, dtype=np.float32)
        if s1 in self.shape_ids:
            if collected[self.shape_ids[s1]] != 1:
                y, x = s1
                shape_index = self.all_shapes[self.maze[y, x]]
                phi[shape_index] = 1.
        elif s1 == self.goal:
            phi[nc] = np.ones(nc, dtype=np.float32)
        return phi
    
    def feature_dim(self):
        return len(self.all_shapes)
    
    """ def get_w(self):
        ns = len(self.all_shapes)
        w = np.zeros((ns + 1, 1))
        for shape, shape_index in self.all_shapes.items():
            w[shape_index, 0] = self.shape_rewards[shape]
        w[ns, 0] = 1.
        return w """
            

""" [GENERAL]
n_samples=20000
n_tasks=20
n_trials=20

[TASK]
maze=[
    ['1', ' ', ' ', ' ', ' ', '2', 'X', ' ', ' ', ' ', ' ', ' ', 'G'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', '1', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['2', ' ', ' ', ' ', ' ', '3', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['X', 'X', '3', ' ', 'X', 'X', 'X', 'X', 'X', ' ', '1', 'X', 'X'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', '2', ' ', ' ', ' ', ' ', '3'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', '2', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['_', ' ', ' ', ' ', ' ', ' ', 'X', '3', ' ', ' ', ' ', ' ', '1']]

[AGENT]
gamma=0.95
epsilon=0.15
T=200
print_ev=2000
save_ev=200
encoding=None

[SFQL]
learning_rate=0.5
learning_rate_w=0.5
use_true_reward=False

[QL]
learning_rate=0.5 """

""" if __name__ == "__main__":

    register(
        id='MiniGrid-FourRoom-v0',
        entry_point='envs.four_room:FourRoomsEnv'
    )

    env = gym.make('MiniGrid-FourRoom-v0')
    env.reset()
    while True:
        env.step(env.action_space.sample())
        env.render() """