from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.policies import q_policy
from tf_agents.agents.dqn import dqn_agent
from tf_agents.typing import types

tf.compat.v1.enable_v2_behavior()


from dqn_my_env_v1_2 import MyTFEnv
from dqn_q_network import MyQNetwork
from tf_agents.networks import q_network


class MyAgent(dqn_agent.DqnAgent):
    def __init__(self, verbose_env=False):
        self._tf_env = MyTFEnv(verbose_env=verbose_env)
        action_spec = self._tf_env.action_spec()
        num_actions = action_spec.maximum - action_spec.minimum + 1 # As our action spec is defined on N 
        observation_spec = self._tf_env.observation_spec()
        time_step_spec = self._tf_env.time_step_spec()
        self._q_network = MyQNetwork( input_tensor_spec=observation_spec, action_spec=action_spec, num_actions = num_actions, dense_layers=1)
        
        # preprocessing_layers = {
        #     'price':tf.keras.layers.Flatten(),
        #     'pos':tf.keras.layers.Dense(2),
        #     'pos_price':tf.keras.layers.Dense(2),
        #     'time':tf.keras.layers.Dense(1)
        # }

        self._q_network = q_network.QNetwork(
            input_tensor_spec=observation_spec,
            action_spec= action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner= tf.keras.layers.Concatenate(axis=-1),
            fc_layer_params = [(40,),(40,)]
        )

        
        super().__init__(
            time_step_spec,
            action_spec,
            q_network=self._q_network,
            optimizer=tf.compat.v1.train.AdadeltaOptimizer(learning_rate=0.001),
        )

    def reset_ep_counter(self):
        self._tf_env.reset_ep_counter()
