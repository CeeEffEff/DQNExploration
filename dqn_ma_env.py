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



class MAEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.float32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(6,), dtype=np.float64, minimum=[0,0,0,0,0,0], name='observation')

        self._entry_pos = np.float64(0)
        self._pos_type = np.int32(0)
        # self._close_price = random.uniform(low=0.0, high=100.0)
        #self._ask_price = random.uniform(low=0.0, high=100.0)
        #self._bid_price = random.uniform(low=0.0, high=100.0)
        self._mid_price = np.random.uniform(low=0.0, high=100.0)
        self._MA_slow = np.random.uniform(low=0.0, high=100.0)
        self._MA_fast = np.random.uniform(low=0.0, high=100.0)
        self._MA_close = np.random.uniform(low=0.0, high=100.0)        
        self._num_steps = 0
        self.max_steps = 100
        self._state = self.get_state()
        self._episode_ended = False

    def get_state(self):
        # return [self._entry_pos, self._pos_type, self._ask_price, self._bid_price, self._MA_slow, self._MA_fast, self._MA_close]
        return [self._entry_pos, self._pos_type, self._mid_price, self._MA_slow, self._MA_fast, self._MA_close]

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._entry_pos = np.float64(0)
        self._pos_type = np.int32(0)
        self._mid_price = np.float64(12)
        self._MA_slow = np.float64(10)
        self._MA_fast = np.float64(11)
        self._MA_close = np.float64(12)
        self._num_steps = 0
        self._state = self.get_state()
        
        print("TO-DO: First state needs to be taken from data feed")
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float64))

    def _step(self, action):
        self._num_steps += 1
        print(self._num_steps)


        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever. End on exit action (3) for now
        print ("Action: ", action)
        if action == 3:
            self._episode_ended = True
        elif action == 0: # flat
            # new_card = np.random.randint(1, 11)
            # self._state += new_card
            # Flat means stay in position - reward should not change?
            pass
        elif action == 1: # enter short
            if self._pos_type == 0:
                # record the price entered short on
                self._entry_pos = self._mid_price
                self._pos_type = 1
        elif action == 2: # enter long
            if self._pos_type == 0:
                # record the price entered short on
                self._entry_pos = self._mid_price
                self._pos_type = 2
        else:
            raise ValueError('`action` should be in [0,1,2,3].')

        # Say stop loss or end of day - we need to exit?
        forced_exit = self._check_exit_condition()
        

        if self._episode_ended or forced_exit:
            if self._pos_type == 1 and not forced_exit:
                reward = self._entry_pos - self._mid_price 
                self._pos_type = 0
            elif self._pos_type == 2 and not forced_exit:
                reward = self._mid_price - self._entry_pos
                self._pos_type = 0
            else:
                reward = 0 # We never entered
            #reward = self._state - 21 if self._state <= 21 else -21
            self._state = self.get_state()
            
            self.print_state()
            return ts.termination(np.array(self._state, dtype=np.float64), reward)
        else:

            # Get next values for MA etc
            self._get_next_indicators()
            
            self._state = self.get_state()
            
            self.print_state()

            return ts.transition(
                np.array(self._state, dtype=np.float64), reward=0.0, discount=1.0)

    def _check_exit_condition(self):
        if self._num_steps == self.max_steps:
            self._episode_ended = True
            return True
        return False

    def _get_next_indicators(self):
        print("TO-DO: Get next values for MA etc")
        self._mid_price += 1
        self._MA_slow += 1
        self._MA_fast += 1
        self._MA_close += 1





    def print_state(self):
        print("entry pos: ", self._entry_pos)
        print("pos type: ", self._pos_type)
        print("mid price: ", self._mid_price)
        print("MA slow: ", self._MA_slow)
        print("MA fast: ", self._MA_fast)
        print("MA close: ", self._MA_close)


def TF_MAEnv():
    return tf_py_environment.TFPyEnvironment(MAEnv())


__all__ = ["MAEnv", "TF_MAEnv"]
