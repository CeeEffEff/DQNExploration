from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tf_agents.policies import q_policy
from tf_agents.networks import network

tf.compat.v1.enable_v2_behavior()

# action_spec = tensor_spec.BoundedTensorSpec((),
#                                             tf.int32,
#                                             minimum=0,
#                                             maximum=2)
# print("\nTheir Action spec", action_spec)                                    
# action_spec = tf_env.action_spec()
# print("\nAction spec", action_spec)

# input_tensor_spec = tensor_spec.TensorSpec((4,), tf.float32)
# print("\nTheir input spec", input_tensor_spec)               
# observation_spec = tf_env.observation_spec()
# print("\nInput spec", observation_spec)


# batch_size = 2
# observations =  tf.repeat(tf_env.reset().observation, repeats=[batch_size], axis=0)

# time_steps = ts.restart(observations, batch_size)

# time_step_spec = ts.time_step_spec(input_tensor_spec)
# print("\nTheir time_step_spec spec", time_step_spec)               
# time_step_spec = tf_env.time_step_spec()
# print("\ntime_step_spec", time_step_spec)

# print("\n\n\n")

# observation_new = tf.ones([batch_size] + time_step_spec.observation.shape.as_list())
# time_steps_new = ts.restart(observation_new, batch_size=batch_size)
# print("\nTheir time_steps", time_steps_new)               
# print("\n Time steps", time_steps)




# num_actions = action_spec.maximum - action_spec.minimum + 1 # As our action spec is defined on N 

class MyQNetwork(network.Network):
  def __init__(self, input_tensor_spec, action_spec, dense_layers:int, num_actions, name=None):
    super(MyQNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)
    self._sub_layers = [
        tf.keras.layers.Dense(num_actions) for _ in range(dense_layers)
    ]
    # For use of scale_to_spec we require a "_single_action_spec"
    # flat_action_spec = tf.nest.flatten(action_spec)
    # if len(flat_action_spec) > 1:
    #   raise ValueError('Only a single action is supported by this network')
    # self._single_action_spec = flat_action_spec[0]

    # self._output_tensor_spec = action_spec
    # self.action_spec = action_spec
    # self._action_spec

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._sub_layers:
      inputs = layer(inputs)

    # Scale and shift actions to the correct range if necessary.
    # inputs = common.scale_to_spec(inputs, self._single_action_spec)
    # inputs = tf.nest.pack_sequence_as(self._output_tensor_spec,
    #                                            [inputs])
    
    return inputs, network_state

