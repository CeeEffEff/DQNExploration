
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import os
from datetime import datetime

import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

from dqn_agent_driver import AgentDriver

import optuna


# Setup
VISUALISATIONS_DIR = "visualisations"

if not os.path.exists(VISUALISATIONS_DIR):
    os.makedirs(VISUALISATIONS_DIR)
visual_subdir = os.path.join(VISUALISATIONS_DIR, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
if not os.path.exists(visual_subdir):
    os.makedirs(visual_subdir)
graph_file_name_prefix = os.path.join(visual_subdir, "AverageReturn_")

# Debug
input_bool = False
input_frequency = 0
evaluate_before_train = False


def pause_input(message:str, iteration:int):
    if input_frequency == 0:
        return
    if iteration % input_frequency == 0 and input_bool:
        input(message)


def plot_average_returns(average_returns, iteration):
    plt.plot(average_returns)
    plt.title('Average Return of Target per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.savefig(f"{graph_file_name_prefix}__{iteration}.png")


def objective(trail:optuna.Trial):
    # HYPER_PARAMETERS
    params = {
        #num_iterations = trail.suggest_int("num_iterations", 30, 100)
        "num_iterations" : 30,
    
        #num_collect_episodes = 10
        "num_collect_episodes" : trail.suggest_int("num_collect_episodes", 2, 20),
    
        #num_eval_episodes = 4
        "num_eval_episodes" : trail.suggest_int("num_eval_episodes", 2, 10),

        #replay_buffer_capacity = 10000
        "replay_buffer_capacity" : trail.suggest_int("replay_buffer_capacity", 1000, 10000, step = 1000),

        #learning_rate = 0.0001
        "learning_rate" : trail.suggest_float("learning_rate", 1e-6, 1e-1, log=True),

        #train_steps = 100
        "train_steps" : trail.suggest_int("train_steps", 50, 500, step = 10),

        #sample_batch_size = 16
        "sample_batch_size" : trail.suggest_int("sample_batch_size", 4, 128)

    }
    return run_test(**params)



def run_test(num_iterations, num_collect_episodes, num_eval_episodes, replay_buffer_capacity, learning_rate, train_steps, sample_batch_size):
    iterations = list(range(0, num_iterations + 1))
    average_returns = []
    driver = AgentDriver(
        num_collect_episodes=num_collect_episodes,
        num_eval_episodes=num_eval_episodes,
        replay_buffer_capacity=replay_buffer_capacity,
        learning_rate=learning_rate,
        verbose_env=True,
        show_summary=input_bool
    )


    print("Initialising target...")
    _, average_return = driver.run_target(verbose=True)
    average_returns.append(average_return)

    if input_bool:
        input("Initialised, PRESS ENTER to continue")
    else:
        print("Initialised, PRESS ENTER to continue")


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
        iteration_losses = driver.train_target(train_steps=train_steps, sample_batch_size=sample_batch_size, verbose=True)
        print()

        pause_input("Press ENTER to evaluate target after training", i)
        print("Evaluating target...")
        num_episodes, average_return = driver.run_target(verbose=True)
        pause_input("Press ENTER continue after the above evaluation", i)

        average_returns.append(average_return)
        plot_average_returns(average_returns, i)

    return average_return



study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=200)

print(f"Optimised average return: {study.best_value}")

input("Complete")

