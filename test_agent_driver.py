
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

from dqn_agent_driver import AgentDriver

driver = AgentDriver(num_collect_episodes=10, num_eval_episodes= 4, replay_buffer_capacity = 10000, verbose_env=True)


input_bool = True
input_frequency = 10
evaluate_before_train = True


print("Initialising target...")
driver.run_target(verbose=True)

if input_bool:
    print("Initialised")
else:
    input("Initialised, PRESS ENTER to continue")

# for i in range(10):
#     print("Iteration", i, end="\r")
#     driver.run_collect(verbose=False)
#     driver.train_target(train_steps=200, sample_batch_size=2, verbose=False)
# print()
# driver.run_target(verbose=True)
    #input()


def pause_input(message:str, iteration:int):
    if input_frequency == 0:
        return
    if iteration % input_frequency == 0 and input_bool:
        input(message)

for i in range(200):
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
    #driver.train_target(train_steps=40, sample_batch_size=10, verbose=True)
    #driver.train_target(train_steps=100, sample_batch_size=64, verbose=True)
    driver.train_target(train_steps=100, sample_batch_size=16, verbose=True)
    #driver.train_target_on_all(verbose=True)
    print()

    pause_input("Press ENTER to evaluate target after training", i)
    print("Evaluating target...")
    driver.run_target(verbose=True)
    pause_input("Press ENTER continue after the above evaluation", i)

input("Completed")
