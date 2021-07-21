
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import os
from datetime import datetime

import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

from dqn_agent_driver import AgentDriver


graph_file_name_prefix = os.path.join("visualisations", "AverageReturn_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

def plot_average_returns(average_returns, iteration):
    plt.plot(average_returns)
    plt.title('Average Return of Target per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.savefig(f"{graph_file_name_prefix}__{iteration}.png")

num_iterations = 200
iterations = list(range(0, num_iterations + 1))
average_returns = []
driver = AgentDriver(num_collect_episodes=10, num_eval_episodes= 4, replay_buffer_capacity = 10000, verbose_env=True)


input_bool = False
input_frequency = 10
evaluate_before_train = False


print("Initialising target...")
_, average_return = driver.run_target(verbose=True)
average_returns.append(average_return)

if input_bool:
    input("Initialised, PRESS ENTER to continue")
else:
    print("Initialised, PRESS ENTER to continue")


def pause_input(message:str, iteration:int):
    if input_frequency == 0:
        return
    if iteration % input_frequency == 0 and input_bool:
        input(message)



for i in iterations[1:]:
    pause_input("Press ENTER to explore using collect policy", i)
    print("Iteration", i)
    print("Exploring...")
    driver.run_collect(verbose=True)

    if (evaluate_before_train):
        pause_input("Press ENTER to evaluate target before training", i)
        print("Before training, evaluating target...")
        driver.run_target(verbose=True)

    pause_input("Press ENTER to train", i)
    print("Training...")
    interation_losses = driver.train_target(train_steps=100, sample_batch_size=16, verbose=True)
    print()

    pause_input("Press ENTER to evaluate target after training", i)
    print("Evaluating target...")
    num_episodes, average_return = driver.run_target(verbose=True)
    pause_input("Press ENTER continue after the above evaluation", i)

    average_returns.append(average_return)
    plot_average_returns(average_returns, i)


input("Completed")
