from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()



class MyPyEnv(py_environment.PyEnvironment):

    def __init__(self, verbose_env=False):
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(9,), dtype=np.float64, minimum=[0,0,0,0,0,0,0,0,0], name='observation')

        self._entry_pos = np.float64(0)
        self._pos_type = np.int32(1)
        self._pos_type = np.int32(1)
        self._pos_type = np.int32(1)
        self._mid_price = np.random.uniform(low=0.0, high=100.0)
        self._MA_slow = np.random.uniform(low=0.0, high=100.0)
        self._MA_fast = np.random.uniform(low=0.0, high=100.0)
        self._MA_close = np.random.uniform(low=0.0, high=100.0)        
        self._num_steps = np.int32(0)
        self.max_steps = 100
        self._state = self.get_state()
        self._episode_ended = False
        self._epi_counter = 0



    def get_state(self):
        return [self._num_steps, self._entry_pos, self._pos_type, self._mid_price, self._MA_slow, self._MA_fast, self._MA_close]

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._entry_pos = np.float64(0)
        self._pos_type = np.int32(1)
        self._mid_price = np.float64(12)
        self._MA_slow = np.float64(10)
        self._MA_fast = np.float64(11)
        self._MA_close = np.float64(12)
        self._num_steps = np.int32(0)
        self._state = self.get_state()
        
        # print("TO-DO: First state needs to be taken from data feed")
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float64))

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        if self._num_steps == 0:
            # print(f"[Episode {self._epi_counter }]")
            self._epi_counter += 1
        
        # print(f"[{self._num_steps}] Action: {action}")

        # Make sure episodes don't go on forever. 
        # Say stop loss or end of day - we need to exit?
        forced_exit = self._check_exit_condition()
        if forced_exit:
            if self._pos_type == 2: # Long
                action = 0 # Sell
            elif self._pos_type == 0: # Short
                action = 2 # Buy to sell

        reward = 0
        action_string = ""
        if action == 0: # Short or sell
            if self._pos_type == 1: # Enter Short
                self._pos_type = 0
                # record the price entered short on
                self._entry_pos = self._mid_price
                action_string = "Enter short"
            elif self._pos_type == 2: # Long, so sell
                reward = self._mid_price - self._entry_pos
                self._pos_type = 1
                action_string = "Sell"
            elif self._pos_type == 0: # already short..
                action_string = "Flat"
        elif action == 1: # Flat
            action_string = "Flat"
        elif action == 2: # Long or buy to sell
            if self._pos_type == 1: # Enter Long
                self._pos_type = 2
                # record the price entered Long on
                action_string = "Enter long"
                self._entry_pos = self._mid_price
            elif self._pos_type == 0: # Short, so buy to sell
                reward = self._entry_pos - self._mid_price
                self._pos_type = 1
                action_string = "Buy to sell"
            elif self._pos_type == 2: # already long..
                action_string = "Flat"
                
        else:
            raise ValueError('`action` should be in [0,1,2].')
        

        print(f"[Episode {self._epi_counter }][{self._num_steps}] Action: {action} {action_string}             ", end="\r")
        self._num_steps += 1


        if self._episode_ended or forced_exit:
            # if forced_exit:
            #     reward = abs(reward) * -1
            #     reward = 0
            
            self._state = self.get_state()
            self.print_state(reward=reward, final=True)
            return ts.termination(np.array(self._state, dtype=np.float64), reward)
        else:
            # Get next values for MA etc
            self._get_next_indicators()
            
            self._state = self.get_state()
            
            self.print_state(reward=reward)
            
            return ts.transition(
                np.array(self._state, dtype=np.float64), reward=reward, discount=1.0)

    def _check_exit_condition(self):
        if self._num_steps == self.max_steps:
            self._episode_ended = True
            return True
        return False

    def _get_next_indicators(self):
        # print("TO-DO: Get next values for MA etc")
        self._mid_price += 1
        self._MA_slow += 1
        self._MA_fast += 1
        self._MA_close += 1


    def reset_ep_counter(self):
        self._epi_counter = 0


    def print_state(self, reward = None, final=False):
        if self._pos_type == 0:
            pos_string = "Short"
        elif self._pos_type == 1:
            pos_string = "Flat"
        elif self._pos_type == 2:
            pos_string = "Long"
        state_string = "\n"
        state_string += f"[step num: {self._num_steps}],"
        state_string += f"[entry pos: {self._entry_pos}],"
        state_string += f"[pos type: {self._pos_type} {pos_string}],"
        state_string += f"[mid price: {self._mid_price}],"
        state_string += f"[MA slow: {self._MA_slow}],"
        state_string += f"[MA fast: {self._MA_fast}],"
        state_string += f"[MA close: {self._MA_close}]"
        if reward is not None:
            if final:
                state_string += f",[Final reward: {reward}]"
            else:
                state_string += f",[Step reward: {reward}]"
        print(state_string)
        reward = None

class MyTFEnv(tf_py_environment.TFPyEnvironment):
    def __init__(self, verbose_env=False):
        self._pyenv = MyPyEnv(verbose_env=verbose_env)
        super().__init__(self._pyenv)

    def reset_ep_counter(self):
        self._pyenv.reset_ep_counter()

__all__ = ["MyPyEnv", "MyTFEnv"]
