
import numpy as np
import sys
from copy import deepcopy

def train_Q(player, env, SIMULATION_TIMES, EPISODE_TIMES, max_step):
    reward_graph = np.zeros((len(player), EPISODE_TIMES))
    step_graph = np.zeros((len(player), EPISODE_TIMES))
    # Q_heat = np.zeros((len(player), env.num_state, env.num_action))
    # state_count = np.zeros((len(player), env.num_state))

    for n_simu in range(SIMULATION_TIMES):
        for n_agent in range(len(player)):
            player[n_agent].init_params()
        sys.stdout.write("\r%s/%s" % (str(n_simu), str(SIMULATION_TIMES-1)))
        # GRCのトレーニング
        for n_epi in range(EPISODE_TIMES):
            # print("n_epi:{}".format(n_epi))
            sys.stdout.write("\r%s/%s" % (str(n_epi), str(EPISODE_TIMES-1)))
            for n_agent in range(len(player)):
                # print("エージェント{}体目" .format(n_agent))
                current_state = deepcopy(env.start)
                step = 0
                while True:
                    # sys.stdout.write("\r%s" % str(step))
                    # print("curr_st:{}" .format(current_state))
                    current_action = player[n_agent].select_action(current_state)
                    next_state, reward, done = env.step(current_action)
                    # env.evaluate_next_state(current_action, current_state)
                    
                    # next_state = env.get_next_state()

                    # reward = env.evaluate_reward(next_state)

                    reward_graph[n_agent, n_epi] += reward

                    player[n_agent].update(current_state, current_action, reward, next_state)

                    current_state = next_state
                    # state_count[n_agent, current_state] += 1
                    step += 1

                    if done is True:
                        player[n_agent].update_eps()
                        step_graph[n_agent, n_epi] += step
                        break
            
            # print("Q:{}" .format(player[0].value_func.Q))
        # for n_agent in range(len(player)):
        #     Q_heat[n_agent] += player[n_agent].value_func.Q

    reward_graph = reward_graph / SIMULATION_TIMES
    step_graph = step_graph / SIMULATION_TIMES
    # _Q_heat = np.zeros((len(player), env.num_state))
    # state_count = state_count / SIMULATION_TIMES
    # print("Q_heat.shape:{}".format(Q_heat.shape))
    # for n in range(len(player)):
    #     _Q_heat[n] = np.amax(Q_heat[n], axis=1) / SIMULATION_TIMES

    return reward_graph, step_graph
