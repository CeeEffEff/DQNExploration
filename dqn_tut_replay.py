from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tf_agents import specs
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step

tf.compat.v1.enable_v2_behavior()



### https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial

### "Reinforcement learning algorithms use replay buffers to store trajectories 
# of experience when executing a policy in an environment. During training,
#  replay buffers are queried for a subset of the trajectories (either a sequential
#  subset or a sample) to "replay" the agent's experience."


# Tensorflwo agents provides tf and python implementations of a replay buffer api.
# The buffer is initialised with a data spec - this describes a single (trajectory) item which can be stored in the buffer
# Typically we retrieve this from the agent (agent.collect_data_spec)




##### TFUniformReplayBuffer
# TFUniformReplayBuffer is the most commonly used replay buffer in TF-Agents, thus we will use it in our tutorial here.
# In TFUniformReplayBuffer the backing buffer storage is done by tensorflow variables and thus is part of the compute graph.
# The buffer stores batches of elements and has a maximum capacity max_length elements per batch segment.
# Thus, the total buffer capacity is batch_size x max_length elements. The elements stored in the buffer must all have a matching data spec.
# When the replay buffer is used for data collection, the spec is the agent's collect data spec.

# To create a TFUniformReplayBuffer we pass in:

# the spec of the data elements that the buffer will store
# the batch size corresponding to the batch size of the buffer
# the max_length number of elements per batch segment

# For my example what is the trajectory?
# For this I think the best way is to create an agent from our environment and call collect_data_spec


from dqn_my_agent import MyAgent

agent = MyAgent()

# batch_size = 32 # suggests we should take this from the env
# My environment is not currently set up to accept batch inputs though
batch_size = 1 if not agent._tf_env.batched else agent._tf_env.batch_size
max_length = 1000

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = agent.collect_data_spec,
    batch_size = batch_size,
    max_length=max_length
)


## To write to the buffer we can use add_batch

## To read:
# get_next() - returns one sample from the buffer. The sample batch size and number of timesteps returned can be specified via arguments to this method.
# as_dataset() - returns the replay buffer as a tf.data.Dataset. One can then create a dataset iterator and iterate through the samples of the items in the buffer.
# gather_all() - returns all the items in the buffer as a Tensor with shape [batch, time, data_spec]

# As in the driver tut, we can add a replay.add_batch method as an observer so that the 
# replay buffer gets updated with each step
# Well there we actually used a list and append, so slightly different

# replay_observer =  [replay_buffer.add_batch]




# Running our agent on its collect policy and adding trajectories to the replay buffer:
# num_episodes = 2
# from dqn_agent_driver import AgentCollectPolicyDriver
# driver = AgentCollectPolicyDriver(num_episodes=num_episodes)

# collect_op = driver.run(verbose=True)

# print(collect_op)

# # I have a replay buffer of shape:
# '''
#     block1 ep1 frame1
#                frame2
#            ...
#            ep2 frame1
#                frame2
#            ...
#            <L frames total>
#     block2 ep1 frame1
#                frame2
#            ...
#            ep2 frame1
#                frame2
#            ...
#            <L frames total>
#     ...
#     blockB ep1 frame1
#                frame2
#            ...
#            ep2 frame1
#                frame2
#            ...
#            <L frames total>
# '''
# # Batch size is one, so we have dimension (1,num_episodes,traj_in_episode)
# # num_steps seems to be how many steps from a given episode we sample
# # sample_batch_size, is how many batches of samples we take
# dataset = driver._replay_buffer.as_dataset(
#     sample_batch_size=4, # Simply influences when we update - analyse 4 then update. Lower batch size - more responsive to one training
#     num_steps=2 # Shows directly the transition of one step to another
#                 # The agent requires that this be 2 as it learns transitions in this way
# )
# iterator = iter(dataset)

# # Now we have defined how we want to pull data out (sample) we sample and train for a set number of samples - 10 here
# num_train_steps = 10
# for i in range(num_train_steps):
#     trajectories, _ = next(iterator)
# #   print(f"[{i}] Traj: {trajectories}")
#     loss = agent.train(experience=trajectories)
# #   print(f"[{i}] Loss: {loss}")
#     print(f"[{i}] Loss: {loss.loss}")

# # This trains the agent's target network

# # To evaluate the target network?



from dqn_agent_driver import AgentDriver

num_episodes = 2
driver = AgentDriver(num_episodes=2)

driver.run_collect(verbose=True)
driver.train_target(verbose=True)
driver.run_target(verbose=True)