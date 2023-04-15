
import numpy as np
import sys

import model
import env
import pandas as pd
import time
import os

from copy import deepcopy

# hyper parameter
lr = 0.1
dr = 0.9
aleph = 6.5

tau_alpha = 0.1
tau_gamma = 0.9
EG_gamma =0.9

SIMULATION_TIMES = 1000
EPISODE_TIMES = 1000

env = env.SuboptimaWorld(row=9, col=9, start=int(9*9/2))
df_reward = [pd.DataFrame()]

time = time.asctime()

def make_output_path():
    path = "output/GRC/"
    os.makedirs(path, exist_ok=True)
    return path

def train(env, SIMULATION_TIMES, EPISODE_TIMES, output_path):
    player=model.GRC(lr, dr, aleph, tau_alpha, tau_gamma, EG_gamma, env.num_state, num_action=env.num_action)
    reward_graph = np.zeros((SIMULATION_TIMES, EPISODE_TIMES))
    step_graph = np.zeros((SIMULATION_TIMES, EPISODE_TIMES))
    for n_simu in range(SIMULATION_TIMES):
        
        player.init_params()
        
        sys.stdout.write("\r%s/%s" % (str(n_simu), str(SIMULATION_TIMES-1)))
        # GRCのトレーニング
        for n_epi in range(EPISODE_TIMES):
            # print("n_epi:{}".format(n_epi))
#             sys.stdout.write("\r%s/%s" % (str(n_epi), str(EPISODE_TIMES-1)))
            current_state = deepcopy(env.start_state)
            step = 0
            while True:
                    # sys.stdout.write("\r%s" % str(step))
                current_action = player.get_select_action(current_state)
                next_state, reward, done = env.step(current_state, current_action)
                player.update(current_state, current_action, reward, next_state)

                current_state = next_state
                step += 1

                if done is True:
                    break
            player.update_GRC_params(reward)
            reward_graph[n_simu, n_epi] = reward
    reward_graph = pd.DataFrame(reward_graph.T)
    reward_graph.to_csv(output_path+"/aleph_{}.csv" .format(aleph))
    return reward_graph, step_graph

if __name__ == "__main__":
    output_path = make_output_path()
    train(env, SIMULATION_TIMES, EPISODE_TIMES, output_path)
