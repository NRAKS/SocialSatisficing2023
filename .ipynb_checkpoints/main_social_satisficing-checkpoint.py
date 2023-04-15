
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
aleph = 0.0

tau_alpha = 0.1
tau_gamma = 0.9
EG_gamma =0.9

N_agent = 1

SIMULATION_TIMES = 1000
EPISODE_TIMES = 1000

env = env.SuboptimaWorld(row=9, col=9, start=int(9*9/2))
df_reward = [pd.DataFrame() for i in range(N_agent)]
df_EG = [pd.DataFrame() for i in range(N_agent)]

time = time.asctime()

def make_output_path(N_agent):
    path = "output/SocialSatisficing/{}_agents" .format( N_agent)
    os.makedirs(path, exist_ok=True)
    return path

def train(env, SIMULATION_TIMES, EPISODE_TIMES, output_path):
    player = []
    for _ in range(N_agent):
        player.append(model.GRC(lr, dr, aleph, tau_alpha, tau_gamma, EG_gamma, env.num_state, num_action=env.num_action))

    for n_simu in range(SIMULATION_TIMES):
        reward_graph = np.zeros((len(player), EPISODE_TIMES))
        EG_graph = np.zeros((len(player), EPISODE_TIMES))
        for n_agent in range(len(player)):
            player[n_agent].init_params()
        sys.stdout.write("\r%s/%s" % (str(n_simu), str(SIMULATION_TIMES-1)))
        # GRCのトレーニング
        for n_epi in range(EPISODE_TIMES):
            # print("n_epi:{}".format(n_epi))
#             sys.stdout.write("\r%s/%s" % (str(n_epi), str(EPISODE_TIMES-1)))
            EG_list=[]
            for n_agent in range(len(player)):
                # print("エージェント{}体目" .format(n_agent))
                current_state = deepcopy(env.start_state)
                step = 0
                while True:
                    # sys.stdout.write("\r%s" % str(step))
                    current_action = player[n_agent].get_select_action(current_state)
                    next_state, reward, done = env.step(current_state, current_action)
                    reward_graph[n_agent, n_epi] += reward

                    player[n_agent].update(current_state, current_action, reward, next_state)

                    current_state = next_state
                    # state_count[n_agent, current_state] += 1
                    step += 1

                    if done is True:
                        break
                player[n_agent].update_GRC_params(reward)
                EG_list.append(player[n_agent].EG)
                EG_graph[n_agent, n_epi] = player[n_agent].EG
            for n in range(len(player)):
                player[n].update_GRC_reference(max(EG_list))
        for n in range(len(player)):
            df_reward[n]["{}".format(n_simu)] = reward_graph[n]
            df_EG[n]["{}".format(n_simu)] = EG_graph[n]
    
    for n in range(N_agent):
        df_reward[n].to_csv(output_path+"/agent_{}.csv" .format(n))
        path = "output/SocialSatisficing/EG/{}_agents" .format(N_agent)
        os.makedirs(path, exist_ok=True)
        df_EG[n].to_csv(path+"/agent_{}.csv" .format(n))

if __name__ == "__main__":
    output_path = make_output_path(N_agent)
    train(env, SIMULATION_TIMES, EPISODE_TIMES, output_path)
