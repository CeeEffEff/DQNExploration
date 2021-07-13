from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.policies import q_policy

tf.compat.v1.enable_v2_behavior()


from dqn_ma_env import TF_MAEnv
from dqn_q_network import MyQNetwork


class MyPolicy(q_policy.QPolicy):
    def __init__(self):
        tf_env = TF_MAEnv()
        action_spec = tf_env.action_spec()
        num_actions = action_spec.maximum - action_spec.minimum + 1 # As our action spec is defined on N 
        observation_spec = tf_env.observation_spec()
        time_step_spec = tf_env.time_step_spec()
        my_q_network = MyQNetwork( input_tensor_spec=observation_spec, action_spec=action_spec, num_actions = num_actions)


        super().__init__(time_step_spec, action_spec, my_q_network)
