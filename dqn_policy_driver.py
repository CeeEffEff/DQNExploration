from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dqn_ma_env import TF_MAEnv
from dqn_q_network import MyQNetwork

import numpy as np
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.policies import q_policy
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import replay_buffer
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()



class PolicyDriver():

    def __init__(self, num_episodes):
        self._env = TF_MAEnv()
        action_spec = self._env.action_spec()
        num_actions = action_spec.maximum - action_spec.minimum + 1 # As our action spec is defined on N 
        observation_spec = self._env.observation_spec()
        time_step_spec = self._env.time_step_spec()

        my_q_network = MyQNetwork( input_tensor_spec=observation_spec, action_spec=action_spec, num_actions = num_actions)

        self._policy = q_policy.QPolicy( time_step_spec, action_spec, q_network=my_q_network)

        self._num_episodes = tf_metrics.NumberOfEpisodes()
        self._env_steps = tf_metrics.EnvironmentSteps()
        self._average_rtn = tf_metrics.AverageReturnMetric()
        replay_buffer = []
        self._replay_buffer = replay_buffer
        observers = [self._replay_buffer.append, self._num_episodes, self._env_steps, self._average_rtn]

        # A driver which terminates when max number of episodes has been reached
        self._driver = dynamic_episode_driver.DynamicEpisodeDriver(self._env, self._policy, observers=observers, num_episodes=num_episodes)


        # Initial driver.run will reset the environment and initialize the policy.
        _, _ = self._driver.run()


    def run(self, verbose=False):
        final_time_step, policy_state = self._driver.run()
        if verbose:
            self.display_metrics()
        return final_time_step, policy_state

    def display_metrics(self):
        #print('final_time_step', final_time_step)
        print('Number of Steps: ', self._env_steps.result().numpy())
        print('Number of Episodes: ', self._num_episodes.result().numpy())
        print('Average Return: ', self._average_rtn.result().numpy())