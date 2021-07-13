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

######## https://www.tensorflow.org/agents/tutorials/2_environments_tutorial #######

# step(action) -> next_time_step

# next_time_step:
#   Observation structure:
#       ?position vector - R^3 : [short?, flat?, long?] - one-hot encoding
#       ?position vector - R^2 : [short price, long price] - price of entered position, 0 indicates no position
#       ?MA vector - R^3 : [MA slow, MA Fast, MA close]
#       ?current price - R [price]
#       ?Prev bar?? - R^4 : [open, high, low, close]?
#       ? Top of book ask/sell?
#       ? mid price (bid + ask) / 2
#
#       [entered position, position type, mid price, MA slow, MA Fast, MA close]



#   Reward structure:
#       sum of relative returns after exits - float
#   Step type:
#       FIRST/MID/LAST
#       Depends on what is considered the start of an episode, first time_step or first entry
#   Discount:
#       Hyper-parameter



class MAEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
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
        
        self._state = self.get_state()
        
        print("TO-DO: First state needs to be taken from data feed")
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float64))

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever. End on exit action (3) for now
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
            if self._pos_type == 1:
                reward = self._entry_pos - self._mid_price 
                self._pos_type = 0
            elif self._pos_type == 2:
                reward = self._mid_price - self._entry_pos
                self._pos_type = 0
            else:
                reward = 0 # We never entered
            #reward = self._state - 21 if self._state <= 21 else -21
            self._state = self.get_state()

            return ts.termination(np.array(self._state, dtype=np.float64), reward)
        else:

            # Get next values for MA etc
            self._get_next_indicators()
            
            self._state = self.get_state()

            return ts.transition(
                np.array(self._state, dtype=np.float64), reward=0.0, discount=1.0)

    def _check_exit_condition(self):
        return False

    def _get_next_indicators(self):
        print("TO-DO: Get next values for MA etc")
        self._mid_price += 1
        self._MA_slow += 1
        self._MA_fast += 1
        self._MA_close += 1


# Random policy to validate specs are correct
environment = MAEnv()
utils.validate_py_environment(environment, episodes=5)


# Fixed policy to see how environment unfolds
environment = MAEnv()
flat_action = np.array(0, dtype=np.int32)
enter_short_action = np.array(1, dtype=np.int32)
enter_long_action = np.array(2, dtype=np.int32)
exit_action = np.array(3, dtype=np.int32)

print("Starting env")
time_step = environment.reset()
print(time_step)
cumulative_reward = time_step.reward

print("Wait one turn")
time_step = environment.step(flat_action)
print(time_step)
cumulative_reward += time_step.reward

print("Enter long")
time_step = environment.step(enter_long_action)
print(time_step)
cumulative_reward += time_step.reward

for _ in range(3):
    print("Wait one turn")
    time_step = environment.step(flat_action)
    print(time_step)
    cumulative_reward += time_step.reward


print("Exit position")
time_step = environment.step(exit_action)
print(time_step)
cumulative_reward += time_step.reward
print("Final Reward = ", cumulative_reward)



# Wrapping an environment

# ActionDiscretizeWrapper: Converts a continuous action space to a discrete action space.
# RunStats: Captures run statistics of the environment such as number of steps taken, number of episodes completed etc.
# TimeLimit: Terminates the episode after a fixed number of steps.


# We will also want to wrap the environment in Tensorflow
environment = MAEnv()
tf_env = tf_py_environment.TFPyEnvironment(environment)

print(isinstance(tf_env, tf_environment.TFEnvironment))
print("TimeStep Specs:", tf_env.time_step_spec())
print("Action Specs:", tf_env.action_spec())



# Usage of the tf environment is similar to the python environment. There is still: 
# reset()
# step(action)
# 
# Below shows an almost identical version to the above python env.
# Additionally I am creating a transitions list as the tutorial does. This is useful for debugging I suppose.
# The tutorial additionally converts this to numpy arrays, and then str, and then a newline seperated statement for printing.
environment = MAEnv()
tf_env = tf_py_environment.TFPyEnvironment(environment) # Wrapper use

# Tf tensors instead of numpy array (although this is the underlying component)
flat_action = tf.convert_to_tensor(np.array(0, dtype=np.int32)) 
enter_short_action = tf.convert_to_tensor(np.array(1, dtype=np.int32))
enter_long_action = tf.convert_to_tensor(np.array(2, dtype=np.int32))
exit_action = tf.convert_to_tensor(np.array(3, dtype=np.int32))

transitions = []
transition = []

print("\n\n\n\nTF env simple example of some steps:")
time_step = tf_env.reset()
cumulative_reward = time_step.reward
transition.append(time_step)

# print("Wait one turn")
time_step = tf_env.step(flat_action)
transition.append(flat_action)
transition.append(time_step)
transitions.append(transition)
transition = []
transition.append(time_step)
cumulative_reward += time_step.reward

# print("Enter long")
time_step = tf_env.step(enter_long_action)
transition.append(enter_long_action)
transition.append(time_step)
transitions.append(transition)
transition = []
transition.append(time_step)
cumulative_reward += time_step.reward

for _ in range(3):
    # print("Wait one turn")
    time_step = tf_env.step(flat_action)
    transition.append(flat_action)
    transition.append(time_step)
    transitions.append(transition)
    transition = []
    transition.append(time_step)
    cumulative_reward += time_step.reward


# print("Exit position")
time_step = tf_env.step(exit_action)
transition.append(exit_action)
transition.append(time_step)
transitions.append(transition)
transition = []
transition.append(time_step)
cumulative_reward += time_step.reward

np_transitions = tf.nest.map_structure(lambda x: x.numpy(), transitions)
print('\n'.join(map(str, np_transitions)))
print('Total reward:', cumulative_reward.numpy())




print("\n\n\n\nTF env example involving episodes:")

environment = MAEnv()
tf_env = tf_py_environment.TFPyEnvironment(environment) # Wrapper use


time_step = tf_env.reset()
rewards = []
steps = []
num_episodes = 5


for _ in range(num_episodes):
    # Each episode has its own reward
    episode_reward = 0
    episode_steps = 0

    while not time_step.is_last():
        action = tf.random.uniform([1], 0, 4, dtype=tf.int32) # Uniformly select one of the actions 0-3
        time_step = tf_env.step(action)
        episode_steps += 1
        episode_reward += time_step.reward.numpy()
       
    
    rewards.append(episode_reward)
    steps.append(episode_steps)

    time_step = tf_env.reset()


num_steps = np.sum(steps)
avg_length = np.mean(steps)
avg_reward = np.mean(rewards)

print('num_episodes:', num_episodes, 'num_steps:', num_steps)
print('avg_length', avg_length, 'avg_reward:', avg_reward)






