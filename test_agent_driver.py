
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

from dqn_agent_driver import AgentDriver

driver = AgentDriver(num_collect_episodes=1000, num_eval_episodes= 1, verbose_env=True)

print("Initialising target")
driver.run_target(verbose=True)


# for i in range(10):
#     print("Iteration", i, end="\r")
#     driver.run_collect(verbose=False)
#     driver.train_target(train_steps=200, sample_batch_size=2, verbose=False)
# print()
# driver.run_target(verbose=True)
    #input()

input_bool = False
def pause_input():
    if input_bool:
        input()

for i in range(1):
    print("Iteration", i)
    print("Exploring...")
    driver.run_collect(verbose=True)

    pause_input()
    print("Before training, evaluating target...")
    driver.run_target(verbose=True)
    pause_input()
    print("Training...")
    #driver.train_target(train_steps=40, sample_batch_size=10, verbose=True)
    driver.train_target(train_steps=100, sample_batch_size=2, verbose=True)
    # driver.train_target_on_all(verbose=True)
    print()
    pause_input()
    print("Evaluating target...")
    driver.run_target(verbose=True)
    pause_input()

input()
