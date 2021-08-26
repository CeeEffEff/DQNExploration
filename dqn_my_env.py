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
        
        self._num_prices = 5
        np.random.seed(123123)
        #self.random_seed = random_seed

        self._observation_spec = {
            'price':array_spec.BoundedArraySpec(shape=(self._num_prices,4), dtype=np.float64, minimum=0, name='obs_price'),
            'pos':array_spec.BoundedArraySpec(shape=(2,), dtype=np.int32, minimum=0, maximum=1, name='obs_pos'),
            'pos_price':array_spec.BoundedArraySpec(shape=(1,), dtype=np.float64, minimum=0, maximum=1, name='obs_pos_price'),
            'time':array_spec.BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=0, maximum=1, name='obs_time')
        }

        self.data_length = 1000
        self.largest_av_length = 200
        self.total_length = self.data_length + self.largest_av_length

        #price
        self.init_prices()
        #pos
        self._short_pos = np.int32(0)
        self._long_pos = np.int32(0)
        #time
        self._step_no = np.int32(0)
        self._bar_number = self._step_no + self._num_prices
        
        self.set_current_bar_prices()

        self.max_steps = 10
        self._state = self.get_state()
        self._episode_ended = False
        self._epi_counter = 0


    def set_current_bar_prices(self):
        index = self._bar_number
        self._price = self._prices[index]
        self._MA_slow = self._MA_slows[index]
        self._MA_fast = self._MA_fasts[index]
        self._MA_close = self._MA_closes[index]

    def init_prices(self):
        
        self._prices = np.empty(shape=(self.total_length,))
        #self._prices[0] = np.random.uniform(low=0.0, high=100.0)
        
        self._prices[0] = np.random.uniform(low=200.0, high=300.0)

        self._MA_slows = np.zeros_like(self._prices)
        self._MA_fasts = np.zeros_like(self._prices)
        self._MA_closes = np.zeros_like(self._prices)

        #np.random.seed(self.random_seed)
        for time in range(1,self.total_length):
            #step = np.random.randint(0, high=2)
            step = np.random.randint(-1, high=1)
            self._prices[time] = self._prices[time - 1] + step
            
            if time > self.largest_av_length:
                self._MA_slows[time] = np.average(self._prices[time-self.largest_av_length:time])
                self._MA_fasts[time] = np.average(self._prices[time-self.largest_av_length:time])
                self._MA_closes[time] = np.average(self._prices[time-self.largest_av_length:time])

        self._prices = self._prices[self.largest_av_length + 1:]
        self._MA_slows = self._MA_slows[self.largest_av_length + 1:]
        self._MA_fasts = self._MA_fasts[self.largest_av_length + 1:]
        self._MA_closes = self._MA_closes[self.largest_av_length + 1:]

        self._entry_price = np.float64(0)



    def get_state(self):
        return {
            'price': np.array(list(zip(self._prices, self._MA_slows, self._MA_fasts, self._MA_closes))[self._bar_number-self._num_prices:self._bar_number], dtype=np.float64),
            'pos': np.array([self._short_pos, self._long_pos], dtype=np.int32),
            'pos_price': np.array([self._entry_price], dtype=np.float64),
            'time': np.array([self._bar_number], dtype=np.int32)
        }

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.init_prices()
        self._short_pos = np.int32(0)
        self._long_pos = np.int32(0)
        self._step_no = np.int32(0)
        self._bar_number = self._step_no + self._num_prices

        self._state = self.get_state()
        
        # print("TO-DO: First state needs to be taken from data feed")
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        if self._step_no == 0:
            self._epi_counter += 1
        
        # Make sure episodes don't go on forever. 
        # Say stop loss or end of day - we need to exit?
        forced_exit = self._check_exit_condition()
        if forced_exit:
            if self._long_pos == 1: # Long
                action = 2 # Sell
            elif self._short_pos == 1: # Short
                action = 0 # Buy to sell

        previous_price = self._price
        self.set_current_bar_prices()

        reward = 0
        action_string = ""
        flat = False

        # DEBUG
        # if action == 0:
        #     reward = 100
        # elif action == 1:
        #     reward = 25
        # elif action == 2:
        #     reward = -100

        if action == 0: # Short or exit short (buy to sell)
            if self._short_pos == 0 and self._long_pos == 0: #  Enter Short as no pos
                self._short_pos = 1
                # record the price entered short on
                self._entry_price = self._price
                action_string = "Enter short"
            elif self._long_pos == 1: # Long, leave alone - flat
                action_string = "Flat"
                flat = True
            elif self._short_pos == 1: # Exit short (buy to sell)
                reward = self._entry_price - self._price
                self._short_pos = 0
                action_string = "Buy to sell"
                self._entry_price = 0
                
        elif action == 1: # Flat
            action_string = "Flat"
            flat = True

        elif action == 2: # Long or exit long (sell)
            if self._short_pos == 0 and self._long_pos == 0: # Enter Long as no pos
                self._long_pos = 1
                # record the price entered Long on
                action_string = "Enter long"
                self._entry_price = self._price
            elif self._short_pos == 1: # Short, leave alone - flat
                action_string = "Flat"
                flat = True
            elif self._long_pos == 1: # exit long (sell)
                reward = self._price - self._entry_price
                self._long_pos = 0
                self._entry_price = 0
                action_string = "Sell"

        else:
            raise ValueError('`action` should be in [0,1,2].')
        
        if flat and (self._short_pos == 1 or self._long_pos == 1):
            reward = self._price - previous_price 
            if self._short_pos:
                reward *= -1



        print(f"[Episode {self._epi_counter }][{self._step_no}] Action: {action} {action_string}             ", end="\r")
        self._step_no += 1
        self._bar_number += 1


        if self._episode_ended or forced_exit:
            # if forced_exit:
            #     reward = abs(reward) * -1
            #     reward = 0
            
            self._state = self.get_state()
            self.print_state(reward=reward, final=True)
            return ts.termination(self._state, reward)
        else:
            # Get next values for MA etc
            self._get_next_indicators()
            
            self._state = self.get_state()
            
            self.print_state(reward=reward)
            
            return ts.transition(self._state, reward=reward, discount=1.0)

    def _check_exit_condition(self):
        if self._step_no == self.max_steps or self._bar_number >= self.data_length - 1:
            self._episode_ended = True
            return True
        return False

    def _get_next_indicators(self):
        # print("TO-DO: Get next values for MA etc")
        return


    def reset_ep_counter(self):
        self._epi_counter = 0


    def print_state(self, reward = None, final=False):
        if self._short_pos == 1:
            pos_string = "Short"
        elif self._long_pos == 1:
            pos_string = "Long"
        else:
            pos_string = "Flat"
        state_string = "\n"
        state_string += f"[step num: {self._step_no}],"
        state_string += f"[entry pos: {self._entry_price}],"
        state_string += f"[pos type: {pos_string}],"
        state_string += f"[mid price: {self._price}],"
        state_string += f"[MA slow: {self._MA_slow}],"
        state_string += f"[MA fast: {self._MA_fast}],"
        state_string += f"[MA close: {self._MA_close}]"
        if reward is not None:
            if final:
                state_string += f",[Final reward: {reward}]"
            else:
                state_string += f",[Step reward: {reward}]"
        # state = self.get_state()
        # state_string += f"\n {state}]"
        print(state_string)
        reward = None

class MyTFEnv(tf_py_environment.TFPyEnvironment):
    def __init__(self, verbose_env=False):
        self._pyenv = MyPyEnv(verbose_env=verbose_env)
        super().__init__(self._pyenv)

    def reset_ep_counter(self):
        self._pyenv.reset_ep_counter()

__all__ = ["MyPyEnv", "MyTFEnv"]
