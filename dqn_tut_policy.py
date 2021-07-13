from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.networks import network
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.policies import py_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import scripted_py_policy

from tf_agents.policies import tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import actor_policy
from tf_agents.policies import q_policy
from tf_agents.policies import greedy_policy

from tf_agents.environments import tf_environment, tf_py_environment

from tf_agents.trajectories import time_step as ts

from tf_agents.utils import common


tf.compat.v1.enable_v2_behavior()


######### https://www.tensorflow.org/agents/tutorials/3_policies_tutorial ##########

#### Policies can be of two types: python and tf
# Python policies are recommended for scripted policies
# TF policies are recommended for those learnt by a NN


# Policies take an observation and output a PolicyStep named tuple of:
#   action: The action to be applied to the environment.
#   state: The state of the policy (e.g. RNN state) to be fed into the next call to action 
#   info: Optional side information such as action log probabilities.

# This is achieved through a call of the action method.



### For examples of python policies see https://www.tensorflow.org/agents/tutorials/3_policies_tutorial


##### TensorFlow policy example
# We will note that instead of ___ArraySpec or ___PyPolicy we will use ___TensorSpec or ___TFPolicy
# A python environment which has been wrapped in TF will have these conversions applied during the interface methods available:
from dqn_ma_env import TF_MAEnv

tf_env = TF_MAEnv()
print(type(tf_env))
assert type(tf_env) is tf_py_environment.TFPyEnvironment
print(tf_env)
print(tf_env.time_step_spec())
print(ts.time_step_spec(tf_env.observation_spec())) # Proves that below the input_shape is referring to that of the observation spec.
print(tf_env.observation_spec())
print(tf_env.action_spec())


# Random agent. Note that the action for such a policy/agent does not depend on the timestep (obs) as the choice is random.
# We still provide it as an argument to the action invocation nonetheless

action_spec_random = tf_env.action_spec()
time_step_spec_random = tf_env.time_step_spec()

test_random_policy = random_tf_policy.RandomTFPolicy(
    action_spec=action_spec_random, time_step_spec=time_step_spec_random
)

time_step = tf_env.reset()

action_step = test_random_policy.action(time_step)


print('Random Action:')
print(action_step.action)




## Actor policy
## "An actor policy can be created using either a network that maps time_steps to actions or a network that maps time_steps to distributions over actions."
# In my environment I don't think we want a distribution of actions

# Seems that regardless, actor policies are defined over action networks.
# These are NNs, but it seems that the tf_agents.networks namespace provides some helpful wrappers/methods.
# To create an actor policy we must first define the action network as a class inheriting from tf_agents.networks.Network

# The action net is the mapping of action observation to action potentially considering internal state and the step type.
# The example I am going to use  has a single dense sublayer with size = num actions
# This is so that output of the network is defined as a probability of each action.
# The activation is tanh which is pretty standard
# !! While tanh is pretty standard, the tutorial mentions that it is all the more important in order to scale and shift the outputs
# !! to be in the range of the actions.

# For my actions (in the range 0-3 inclusive) I will require this.
# The tutorial suggested to look at ActorNetwork() from tf_agents.agents.ddpg.actor_network
# The source code for this is available at https://github.com/tensorflow/agents/blob/v0.7.1/tf_agents/agents/ddpg/actor_network.py#L32-L129
# It's call method is as follows:
#   def call(self, observations, step_type=(), network_state=(), training=False):
#     del step_type  # unused.
#     observations = tf.nest.flatten(observations)
#     output = tf.cast(observations[0], tf.float32)
#     for layer in self._mlp_layers:
#       output = layer(output, training=training)

#     actions = common.scale_to_spec(output, self._single_action_spec)
#     output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec,
#                                               [actions])

#     return output_actions, network_state


# We can see that the ActorNetwork class makes use of tf_agents.utils.common.scale_to_spec
# This makes my life rather easy. I took the example and made a few changes in call
class ActionNet(network.Network):

  def __init__(self, input_tensor_spec, output_tensor_spec):
    super(ActionNet, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name='ActionNet')
    self._output_tensor_spec = output_tensor_spec

    # For use of scale_to_spec we require a "_single_action_spec"
    flat_action_spec = tf.nest.flatten(output_tensor_spec)
    if len(flat_action_spec) > 1:
      raise ValueError('Only a single action is supported by this network')
    self._single_action_spec = flat_action_spec[0]

    self._sub_layers = [
        tf.keras.layers.Dense(
            action_spec.shape.num_elements(), activation=tf.nn.tanh),
    ]

  def call(self, observations, step_type, network_state):
    del step_type

    output = tf.cast(observations, dtype=tf.float32)
    for layer in self._sub_layers:
      output = layer(output)
    # actions = tf.reshape(output, [-1] + self._output_tensor_spec.shape.as_list())

    # Scale and shift actions to the correct range if necessary.
    actions = common.scale_to_spec(output, self._single_action_spec)
    output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec,
                                               [actions])

    # tf.nest.pack_sequence_as: Returns a given flattened sequence packed into a given structure.
    print("In ActionNet output_actions ", output_actions)
    # return actions, network_state
    return output_actions, network_state


##### Creating a deterministic actor policy from an actor network
observation_spec = tf_env.observation_spec()
action_spec = tf_env.action_spec()
time_step_spec = tf_env.time_step_spec()

action_net = ActionNet(input_tensor_spec=observation_spec, output_tensor_spec=action_spec)
my_actor_policy = actor_policy.ActorPolicy(
    time_step_spec=time_step_spec,
    action_spec=action_spec,
    actor_network=action_net
    )

# The great thing about a tf agent and env is that they accept batches.
# So we can provide any batch of time_steps that follow the spec.
# For training this would be great - can provide many different starting dates


batch_size = 2
observations =  tf.repeat(tf_env.reset().observation, repeats=[batch_size], axis=0)

time_step = ts.restart(observations, batch_size) #  This is a little redundant as we could have just taken the reset objects? Maybe not in every case
                # At the very least it deals with setting up the component tensors in the correct shape which makes things easier
# print(time_step)

print("Starting batch time_step actions [deterministic]...")
action_step = my_actor_policy.action(time_step) # time_step now has batch dimension, and therefore also aciton_step

print('Batch of actions:')
print(action_step.action)

# We see from the distribution invocation that we are not using a distribution, but mappings are instead determininistic
# This is shown by the fact that we provided the same initial time_steps and will as a result always see the same initial action performed across the batch
distribution_step = my_actor_policy.distribution(time_step)
print('Action distribution:')
print(distribution_step.action)

print("\n\n\n\n\n\n\n")
##### Creating a stochastic actor policy from an actor network
# This policy is stochastic in the sense that there is a probability of each action being taken
# From the example this really is some syntactic sugar for taking the outputs as means and generating a PDF from the means.
# The policy network is in fact an extentions to the deterministic one, overriding call()
# From the example:
class ActionDistributionNet(ActionNet):

  def call(self, observations, step_type, network_state):
    action_means, network_state = super(ActionDistributionNet, self).call(
        observations, step_type, network_state)
    print("In ActionDistributionNet action_means: ", action_means)
    action_std = tf.ones_like(action_means, dtype=tf.float64) # create standard deviations of one in shape of action outputs
    return tfp.distributions.MultivariateNormalDiag(tf.cast(action_means, dtype=tf.float64), action_std), network_state


batch_size = 2
observations =  tf.repeat(tf_env.reset().observation, repeats=[batch_size], axis=0)

time_step = ts.restart(observations, batch_size)

observation_spec = tf_env.observation_spec()
action_spec = tf_env.action_spec()
time_step_spec = tf_env.time_step_spec()

action_distribution_net = ActionDistributionNet(input_tensor_spec=observation_spec, output_tensor_spec=action_spec)




# my_actor_policy = actor_policy.ActorPolicy(
#     time_step_spec=time_step_spec,
#     action_spec=action_spec,
#     actor_network=action_distribution_net)

# action_step = my_actor_policy.action(time_step)
# print('Action:')
# print(action_step.action)
# distribution_step = my_actor_policy.distribution(time_step)
# print('Action distribution:')
# print(distribution_step.action)


# !!! The above would need to be changed to have an action spec that is float64.  I am not going to do this as my env has int32 actions.
# !!! Alternatively we could implement a wrapper around the stochastic policy such as a GreddyPolicy wrapper to provide a deterministic action ouput and distribution.





##### Creating a Q actor policy from an actor network
# From the tutorial:
# "A Q policy is used in agents like DQN and is based on a Q network that predicts a Q value for each discrete 
# action. For a given time step, the action distribution in the Q Policy is a categorical distribution
# created using the q values as logits."

# This is what I am aiming for. Note the mention of "discrete action" - this fits the environment spec we have

# In fact, the network structure is very similar to the the deterministic action network we previously created.
# The main difference is indeed in the policy.
# !!! Interstingly there seems to be no reshape, scaling or shifting of the out actions before they are returned like with the actionnet.
# !!! my only assumption is that the q learning policy is written such that it is happy to deal with this shifting/scaling itself.



action_spec = tensor_spec.BoundedTensorSpec((),
                                            tf.int32,
                                            minimum=0,
                                            maximum=2)
print("\nTheir Action spec", action_spec)                                    
action_spec = tf_env.action_spec()
print("\nAction spec", action_spec)

input_tensor_spec = tensor_spec.TensorSpec((4,), tf.float32)
print("\nTheir input spec", input_tensor_spec)               
observation_spec = tf_env.observation_spec()
print("\nInput spec", observation_spec)



batch_size = 2
observations =  tf.repeat(tf_env.reset().observation, repeats=[batch_size], axis=0)

time_steps = ts.restart(observations, batch_size)

time_step_spec = ts.time_step_spec(input_tensor_spec)
print("\nTheir time_step_spec spec", time_step_spec)               
time_step_spec = tf_env.time_step_spec()
print("\ntime_step_spec", time_step_spec)

print("\n\n\n")

observation_new = tf.ones([batch_size] + time_step_spec.observation.shape.as_list())
time_steps_new = ts.restart(observation_new, batch_size=batch_size)
print("\nTheir time_steps", time_steps_new)               
print("\n Time steps", time_steps)




num_actions = action_spec.maximum - action_spec.minimum + 1 # As our action spec is defined on N 

class QNetwork(network.Network):
  def __init__(self, input_tensor_spec, action_spec, num_actions=num_actions, name=None):
    super(QNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)
    self._sub_layers = [
        tf.keras.layers.Dense(num_actions),
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






my_q_network = QNetwork( input_tensor_spec=observation_spec, action_spec=action_spec)
# print("my_q_network.action_spec ", my_q_network.action_spec)
# print("action_spec ", action_spec)
my_q_policy = q_policy.QPolicy( time_step_spec, action_spec, q_network=my_q_network)

action_step = my_q_policy.action(time_steps)
distribution_step = my_q_policy.distribution(time_steps)

print('Q Action:')
print(action_step.action)

print('Q Action distribution:')
print(distribution_step.action)





# Example of wrapping this in a GreedyWrapper
my_greedy_policy = greedy_policy.GreedyPolicy(my_q_policy)

action_step = my_greedy_policy.action(time_steps)
print('Action:')
print(action_step.action)

distribution_step = my_greedy_policy.distribution(time_steps)
print('Action distribution:')
print(distribution_step.action)



