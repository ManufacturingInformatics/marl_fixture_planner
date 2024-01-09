"""
Multi-Agent bandit for fixture placement - ONLY TO BE USED FOR THE WING SPAR DEFORMATION

Raises:
    NotImplementedError: _description_

Returns:
    _type_: _description_
"""
import numpy as np
import gym
from gym import spaces, utils, error
from gym.utils import seeding
from sympy import symbols
import calculateDeformationMARLSpar
import matlab

def create_contexts(y_low=-950, y_high=-50):
    y_points = np.linspace(y_low, y_high, num=500, dtype=np.float32)
    x_points = np.zeros((500,1), dtype=np.float32)+30
    return np.append(np.reshape(x_points, (500,1)), np.reshape(y_points, (500,1)), axis=1)

def create_actions(num_points=100):
    """
    Creates the discrete action space that is used for selecting where the fixtures are placed on the wing panel. The points are only generated in the x- and y- directions. 

    Args:
        num_points (int, optional): Number of fixture positions in each direction. Square the number to find out how many possible fixtures are generated before filtering. Defaults to 100.

    Returns:
        np.ndarray : (10000,2) array where the first column is the fixture position in the y-direction and the second column is the position in the z-direction
    """
    y = np.arange(-950, -40, (950-40)/num_points)
    z = np.arange(20, 180, (180-20)/num_points)
    Y, Z = np.meshgrid(y, z)
    positions = np.vstack([Y.ravel(), Z.ravel()]).T
    y_points = []
    z_points = []

    for y, z in positions:
        rules = [
            y > -1000,
            y < 0,
            z > 10,
            z < 190
        ]
        if all(rules):
            y_points.append(y)
            z_points.append(z)
                
    yvec = np.asarray(y_points, dtype=np.float32)
    zvec = np.asarray(z_points, dtype=np.float32)
    return np.append(np.reshape(yvec, (10000,1)), np.reshape(zvec, (10000,1)), axis=1)

class MultiAgentFixtureBandit(gym.Env):
    """
    Class for performing fixturing design using the contextual bandit example. The environment extends the OpenAI Gym environment but relies on external control for the contexts

    Args:
        gym (gym.Env): Extends the gym.Env class to include the functionality that it provides
    """
    
    def __init__(
        self, 
        contexts, 
        actions, 
        num_agents=1,
        n_features=1):
        
        self.num_agents = num_agents
        self.done = False
        self.num_contexts = len(contexts)
        self.num_actions = len(actions)
        
        self.action_list = actions
        self.context_list = contexts
        self.actions = gym.spaces.Discrete(self.num_actions)
        self.contexts = gym.spaces.Box(low=np.array([10, -1000]), high=np.array([50, 0]), shape=(2,)) # , dtype=np.float64
        
        self.xvec, self.zvec = symbols('x z')
        self.eq_coefficients = {
            'a': 1,
            'b': 1,
            'c': 0
        }
        
        self.mean_reward = 0.7854
        
        self.seed()
    
    def seed(self, seed=None):
        """
        Creates a random seed for training the algorithm

        Returns:
            np.ndarry : Random seed value for this iteration
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, actions, context):
        
        actions_list = []
        
        for i in range(self.num_agents):
            actions_list.append(self.action_list[tuple(actions[i])])
            
        
        for action in actions_list:
            assert action in self.action_list
            
        assert context in self.context_list
        
        observation = self._get_obs(np.array(actions_list), context)
        reward = self._get_reward(observation)
        
        done = True
        
        return observation, reward, done, {}
    
    def reset(self):
        raise NotImplementedError()
    
    def _get_obs(self, actions, context):
        """
        Calls the MATLAB function and receives the deformation from the MATLAB script

        Args:
            action (np.ndarray): Action locator that is provided as coordinates
            context (np.ndarray): Specific context that relates to the drilling

        Returns:
            np.ndarray: N-dimensional array of the deformations and residual stresses of the component
        """
        fixture_pos_n = np.append(
            np.zeros((self.num_agents, 1), dtype=np.float32),
            actions.reshape((self.num_agents,2)),
            axis=1
        )
        selected_actions = fixture_pos_n.tolist()
        selected_context = context.tolist()
        selected_context.append(200)

        my_calculate = calculateDeformationMARLSpar.initialize()
        fixture_pos_n = matlab.double(selected_actions, size=(self.num_agents,3))
        drill_pos = matlab.double([selected_context], size=(1, 3))
        
        points = my_calculate.calculateDeformationMARLSpar(fixture_pos_n, drill_pos)
        
        my_calculate.terminate()

        return np.asarray(points)
    
    def _get_done(self):
        return self.done
    
    def _get_reward(self, observation=None):
        """
        Calculates the reward based on the Gaussian reward function

        Args:
            observation (np.ndarray, optional): N-dimensional array that contains the X-,Y-,Z-dimension deformations and the residual . Defaults to None.

        Returns:
            np.ndarray: Returns an array of a singular reward based on the X- and Z-dimension deformation values
        """
        if observation is None:
            return 0
        else:
            x = observation[0,0]*10**6
            z = observation[0,2]*10**6
            
            R = (1/(1 + ((self.xvec-self.eq_coefficients['c'])/self.eq_coefficients['a'])**(2*self.eq_coefficients['b'])) + 1/(1 + ((self.zvec-self.eq_coefficients['c'])/self.eq_coefficients['a'])**(2*self.eq_coefficients['b'])))/2
            
            return np.array(R.subs(self.xvec, x).subs(self.zvec, z)).astype(np.float32)