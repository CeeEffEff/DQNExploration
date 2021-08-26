from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

from dqn_my_agent import MyAgent

import numpy as np
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer, episodic_replay_buffer

import tensorflow as tf
tf.compat.v1.enable_v2_behavior()



class AgentDriver:
    def __init__(self, num_collect_episodes, num_eval_episodes, replay_buffer_capacity, learning_rate, fc_layer_units, fc_layer_depth, verbose_env=False, show_summary=False):
        self._agent = MyAgent(learning_rate, fc_layer_units, fc_layer_depth, verbose_env=verbose_env, show_summary=show_summary)
        self._agent.initialize()
        self._collect_driver = AgentCollectPolicyDriver(self._agent, num_collect_episodes, replay_buffer_capacity)
        self._target_driver = AgentTargetPolicyDriver(self._agent, num_eval_episodes)
        

    def run_collect(self,verbose=False):
        self._agent.reset_ep_counter()
        _ = self._collect_driver.run(verbose=verbose)
        if verbose:
            print()

    def train_target(self,train_steps:int, sample_batch_size:int, verbose=False):
        
        dataset = self._collect_driver._replay_buffer.as_dataset(
            num_parallel_calls=AUTOTUNE,
            single_deterministic_pass=True,
            sample_batch_size=sample_batch_size, # Simply influences when we update - analyse 4 then update. Lower batch size - more responsive to one training
            num_steps=2 # Shows directly the transition of one step to another
                        # The agent requires that this be 2 as it learns transitions in this way
            
        )
        iterator = iter(dataset)
        
        # Now we have defined how we want to pull data out (sample) we sample and train for a set number of samples
        num_train_steps = train_steps
        print("Number of frames in replay: ", self._collect_driver._replay_buffer.num_frames().numpy())
        num_train_steps = int(self._collect_driver._replay_buffer.num_frames().numpy()/sample_batch_size)
        if num_train_steps == 0:
            num_train_steps = 1

        total_loss = 0
        max_loss = 0
        all_loss = []
        for i in range(num_train_steps):
            trajectories, _ = next(iterator)
            loss = self._agent.train(experience=trajectories)
            all_loss.append(loss.loss)
            max_loss = max(max_loss, loss.loss)
            total_loss += loss.loss
            if verbose:
                print(f"[{i}] Loss: {loss.loss}", end="\r")
        if verbose:
            print()
            print(f"[Total] Loss: {total_loss}")
            print(f"[Average] Loss: {total_loss/num_train_steps}")
            print(f"[Max] Loss: {max_loss}")
            print()

        return all_loss


    def train_target_on_all(self, verbose=False):
        trajectories = self._collect_driver._replay_buffer.gather_all()
        loss = self._agent.train(experience=trajectories)
        if verbose:
            print(f"[All] Loss: {loss.loss}")
        if verbose:
            print()


    def run_target(self,verbose=False):
        self._agent.reset_ep_counter()
        _, _, num_episodes, average_return = self._target_driver.run(verbose=verbose)
        if verbose:
            print()
        return num_episodes, average_return

    

class AgentCollectPolicyDriver(dynamic_episode_driver.DynamicEpisodeDriver):

    def __init__(self, agent, num_episodes, replay_buffer_capacity):

        self._agent = agent


        batch_size = 1 if not self._agent._tf_env.batched else self._agent._tf_env.batch_size
        
        self._replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec = self._agent.collect_data_spec,
            batch_size = batch_size,
            max_length=replay_buffer_capacity
        )
        self._num_episodes_metric = tf_metrics.NumberOfEpisodes()
        self._env_steps = tf_metrics.EnvironmentSteps()
        self._average_rtn = tf_metrics.AverageReturnMetric()
        
        observers = [self._replay_buffer.add_batch, self._num_episodes_metric, self._env_steps, self._average_rtn]

        super().__init__(self._agent._tf_env, self._agent.collect_policy, observers=observers, num_episodes=num_episodes)
        
        # Initial driver.run will reset the environment and initialize the policy.
        # _, _ = self._driver.run()

    def reset_observers(self):
        self._num_episodes_metric.reset()
        self._env_steps.reset()
        self._average_rtn.reset()
        self._replay_buffer.clear()

    def run(self, verbose=False):
        self.reset_observers()
        final_time_step, policy_state = super().run()
        if verbose:
            self.display_metrics()
        return final_time_step, policy_state

    def display_metrics(self):
        print()
        print('Number of Steps: ', self._env_steps.result().numpy())
        print('Number of Episodes: ', self._num_episodes_metric.result().numpy())
        print('Average Return: ', self._average_rtn.result().numpy())



class AgentTargetPolicyDriver(dynamic_episode_driver.DynamicEpisodeDriver):

    def __init__(self, agent, num_episodes):

        self._agent = agent

        
        self._num_episodes_metric = tf_metrics.NumberOfEpisodes()
        self._env_steps = tf_metrics.EnvironmentSteps()
        self._average_rtn = tf_metrics.AverageReturnMetric()
        
        observers = [self._num_episodes_metric, self._env_steps, self._average_rtn]

        super().__init__(self._agent._tf_env, self._agent.policy, observers=observers, num_episodes=num_episodes)

        # Initial driver.run will reset the environment and initialize the policy.
        # _, _ = self._driver.run()

    def reset_observers(self):
        self._num_episodes_metric.reset()
        self._env_steps.reset()
        self._average_rtn.reset()

    def run(self, verbose=False):
        self.reset_observers()
        final_time_step, policy_state = super().run()
        if verbose:
            self.display_metrics()
        return final_time_step, policy_state, self._num_episodes_metric.result().numpy(), self._average_rtn.result().numpy()

    def display_metrics(self):
        print()
        print('Number of Steps: ', self._env_steps.result().numpy())
        print('Number of Episodes: ', self._num_episodes_metric.result().numpy())
        print('Average Return: ', self._average_rtn.result().numpy())


    
