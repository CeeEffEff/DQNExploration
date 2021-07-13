# from __future__ import absolute_import, division, print_function

import base64
import imageio
imageio.plugins.ffmpeg.download()
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pyvirtualdisplay

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

# Set up a virtual display for rendering OpenAI gym environments.
display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

print(tf.version.VERSION)


num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}




env_name = 'CartPole-v0'
env = suite_gym.load(env_name)


#@test {"skip": true}
env.reset()
im = Image.fromarray(env.render())
im.show()

print("\n\n\n\n\n")

print('\nObservation Spec:')
print(env.time_step_spec().observation)

print('\nReward Spec:')
print(env.time_step_spec().reward)

print('\nAction Spec:')
print(env.action_spec())


# In the Cartpole environment:

# observation is an array of 4 floats:
#   the position and velocity of the cart
#   the angular position and velocity of the pole
# reward is a scalar float value
# action is a scalar integer with only two possible values:
#   0 — "move left"
#   1 — "move right"

time_step = env.reset()
print('Time step:')
print(time_step)

action = np.array(1, dtype=np.int32)

next_time_step = env.step(action)
print('Next time step:')
print(next_time_step)

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

# Must convert to TF env - converts numpy arrays to tensors
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)



# Create a simple set of dense layers, first having 100 nodes, second 50, and the final having nodes = num actions
fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# it's output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])



# Now wrap this q_net in a DQN agent.
# Train step is a recorded variable
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()


###### Policies
# Agents contain two policies:

# agent.policy — The main policy that is used for evaluation and deployment.
# agent.collect_policy — A second policy that is used for data collection.
eval_policy = agent.policy
collect_policy = agent.collect_policy


# We could generate a policy independent of our agent, say just a random one:
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())




# To get an action from a policy, call the policy.action(time_step) method. 
# The time_step contains the observation from the environment. 
# This method returns a PolicyStep, which is a named tuple with three components:
    # action — the action to be taken (in this case, 0 or 1)
    # state — used for stateful (that is, RNN-based) policies
    # info — auxiliary data, such as log probabilities of actions


# Let's test the random policy:
print("\n\n\n\n\n\n\n\n\n\n")
example_environment = tf_py_environment.TFPyEnvironment(
    suite_gym.load('CartPole-v0'))
time_step = example_environment.reset() # Set to first ts's observation
action_step = random_policy.action(time_step)
print(example_environment.step(action_step.action))




# Metrics and Evaluation
# The most common metric used to evaluate a policy is the average return. 
# The return is the sum of rewards obtained while running a policy in an environment for an episode. 
# Several episodes are run, creating an average return.

# The following function computes the average return of a policy, given the policy, environment, and a number of episodes.
#@test {"skip": true}
def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics




# Grabbing the random policy is good because this serves as a baseline for the agent.
# If we perform as well as random we may as well just select actions randomly.

print(compute_avg_return(eval_env, random_policy, num_eval_episodes))







# Replay Buffer
# The replay buffer keeps track of data collected from the environment. 
# This tutorial uses tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer, as it is the most common.

# The constructor requires the specs for the data it will be collecting. 
# This is available from the agent using the collect_data_spec method. The batch size and maximum buffer length are also required.

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

# For most agents, collect_data_spec is a named tuple called Trajectory, containing the specs for observations, actions, rewards, and other items.
print("collect_data_spec")
print(type(agent.collect_data_spec))
print(agent.collect_data_spec)
print()
print(agent.collect_data_spec._fields)
input()
# It is named so as the tuple encapsulates the current time step (observation), the action taken, and the following time step (observation)
# So it is the trajectory of the agent in the environment after taking a given action





#### Data collection
# Data collection is then the execution of a policy over a number of timesteps, with the trajectories being stored in the replay buffer

# We achieve this in the following common loops:
#@test {"skip": true}
def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)

collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

# This loop is so common in RL, that we provide standard implementations. 
# For more details see the drivers module.
# https://www.tensorflow.org/agents/api_docs/python/tf_agents/drivers







# We have the replay data now, sure, but how exactly will we make it available to the agent?
# Currently each row in the replay data is a single timestep.
# An agent, when evaluating/training, requires the current timestep AND the next one in order to compute loss
#  - how benefical it's chosen action was to the one in the replay data.

# There are options for prefetching data and  having parallel calls to help with efficient processing
# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)


print("\n\n\n\n", dataset)

iterator = iter(dataset)
# print("\n\n\n\n", iterator)
# print("\n\n",iterator.next())
# print("\n\n\n\n",iter(replay_buffer.as_dataset()).next())


#### Training the agent

#@test {"skip": true}
#try:
# %time
# # except:
#     pass

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)
# train_step_counter is a counter which is defined as:
#   An optional counter to increment every time the train op is run. Defaults to the global_step.


# Evaluate the agent's policy once before training. 
# This allows us to compare the results from training. 
# Note we also have our random policy to compare against
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
    # So we take the current collect policy that the agent has, and use it to get some data into the replay buffer
    # What is this collect_policy?
    # Above we noted that # Agents contain two policies:
            # agent.policy — The main policy that is used for evaluation and deployment.
            # agent.collect_policy — A second policy that is used for data collection.
    # So I think that this means we will learn a policy of what information is the most important to the agent, 
    # and therefore what information to collect?
    # Well the collection policy is still the application of policy to get an action and then a trajectory.
    # It is not just simply retrieving time_steps.
    # Rather, it seems the agent learns a policy for acting in a training environment, which gives it a replay buffer/
    # Perhaps the agent will attempt to learn a policy here which gives it the best information about different actions
    # it can perform in the "real" environment, not the best actions.
    # Yes this seems to be exactly it:
        # There are some reinforcement learning algorithms, such as Q-learning, that use a policy to behave in (or interact with) 
        # the environment to collect experience, which is different than the policy they are trying to learn (sometimes known as the target policy). 
        # These algorithms are known as off-policy algorithms. 
        # An algorithm that is not off-policy is known as on-policy (i.e. the behaviour policy is the same as the target policy). 
        # An example of an on-policy algorithm is SARSA. That's why we have both policy and collect_policy in TF-Agents, i.e., in general, 
        # the behavioural policy can be different than the target policy (though this may not always be the case).

        # Why should this be the case? Because during learning and interaction with environment, 
        # you need to explore the environment (i.e. take random actions), while, once you have learned the near-optimal policy, 
        # you may not need to explore anymore and can just take the near-optimal action (I say near-optimal rather than optimal because you may not have learned the optimal one)


    collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    # Iterator is tied to the dataset variable which we linked to the replay buffer

    train_loss = agent.train(experience).loss
    # agent.train takes a batch of experience in the form of a trajectory.
    # So in this case this will be the experience retrieved from the agent's collect policy
    # the return object is a LossInfo object
    # LossInfo.Loss returns the loss component of this
    # One would assume that this loss component has already been used in agent.train to adjust weights
    # as also available is agent.loss which one would assume to not change any weights

    # So we got our experience from the collect policy, and we use it to train the target policy
    # Below we see that we evaluate the target policy at intervals of collection training

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)


def create_policy_eval_video(policy, filename, num_episodes=20, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())
    # return embed_mp4(filename)


create_policy_eval_video(agent.policy, "trained-agent")
create_policy_eval_video(random_policy, "random-agent")