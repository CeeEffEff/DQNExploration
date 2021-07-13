from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_py_policy
from tf_agents.policies import random_tf_policy
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import py_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.policies import q_policy
from tf_agents.replay_buffers import replay_buffer

tf.compat.v1.enable_v2_behavior()

# From: https://www.tensorflow.org/agents/tutorials/4_drivers_tutorial


### In short, drivers control the loop of steps and episodes for an agent policy in an environment.
# This includes during data collection, evaluation and generating a video of the agent.
#
# Built in is also the functionality of recording trajectories tuples.
#
# Drivers, like much of what we have covered can be defined in python or tensorflow.
# I am not going to cover the python drivers, but they are in the above tutorial link.
# 
# Both are defined as classes which take a environment, policy and list of observers to update at each timestep.
# Drivers are started by invocation of the run() method.
# When run is invoked the environment is stepped through using actions from the policy until either:
# a) The number of steps reaches max_steps 
#   - in TF this is a DynamicStepDriver    
# b) The number of episodes reaches max_episodes
#   - in TF this is a DynamicEpisodeDriver
#
# We have already defined the environment and policy in TF
# Similarly, tf_metrics (tf_agents.metrics) contains metrics we can pass as observers




## Getting my env and policy

from dqn_ma_env import TF_MAEnv
from dqn_q_network import MyQNetwork

tf_env = TF_MAEnv()
action_spec = tf_env.action_spec()
num_actions = action_spec.maximum - action_spec.minimum + 1 # As our action spec is defined on N 
observation_spec = tf_env.observation_spec()
time_step_spec = tf_env.time_step_spec()

my_q_network = MyQNetwork( input_tensor_spec=observation_spec, action_spec=action_spec, num_actions = num_actions)
tf_policy = q_policy.QPolicy( time_step_spec, action_spec, q_network=my_q_network)
## Getting my env and policy


num_episodes = tf_metrics.NumberOfEpisodes()
env_steps = tf_metrics.EnvironmentSteps()
average_rtn = tf_metrics.AverageReturnMetric()
replay_buffer = [] # Note that replay_buffer is a tf variable so it tracked 
observers = [replay_buffer.append, num_episodes, env_steps, average_rtn]
# A driver which terminates when max number of episodes has been reached
driver = dynamic_episode_driver.DynamicEpisodeDriver(tf_env, tf_policy, observers=observers, num_episodes=2)


# Initial driver.run will reset the environment and initialize the policy.
final_time_step, policy_state = driver.run()

print('final_time_step', final_time_step)
print('Number of Steps: ', env_steps.result().numpy())
print('Number of Episodes: ', num_episodes.result().numpy())
print('Average return: ', average_rtn.result().numpy())
print('Replay buffer first', replay_buffer[0])


# Continue running from previous state
final_time_step, _ = driver.run(final_time_step, policy_state)

print('final_time_step', final_time_step)
print('Number of Steps: ', env_steps.result().numpy())
print('Number of Episodes: ', num_episodes.result().numpy())
print('Average return: ', average_rtn.result().numpy())
print('Replay buffer first',  replay_buffer[0])

# batch_size = 2
# observations =  tf.repeat(tf_env.reset().observation, repeats=[batch_size], axis=0)
# time_steps = ts.restart(observations, batch_size)

# action_step = my_q_policy.action(time_steps)
# distribution_step = my_q_policy.distribution(time_steps)

# print('Q Action:')
# print(action_step.action)

# print('Q Action distribution:')
# print(distribution_step.action)
