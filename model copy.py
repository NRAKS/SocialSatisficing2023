
import numpy as np
import math
import sys


import numpy as np
import random


# greedy方策
class Greedy(object):  # greedy方策
    # 行動価値を受け取って行動番号を返す
    def select_action(self, value, env):
        # print("value:{}" .format(value))
        # print(np.argmax(value))
        act = np.asarray([0, 0])
        _act = np.where(value == np.max(value))
        # print(_act)
        select_act = np.random.randint(len(_act[0]))
        print("select_act:\t{}".format(select_act))
        for n in range(len(env.n_act)):
            act[n] = _act[n][select_act]
        print("acr:\t{}" .format(act))
        return act

    def init_params(self):
        pass

    def update_params(self):
        pass


# ε-greedy
class EpsGreedy(Greedy):
    def __init__(self, eps):
        self.eps = eps

    def select_action(self, value, env):
        if np.random.random() < self.eps:
            act = np.asarray([0, 0])
            # print(value[current_state])
            # print("value.shape:{}" .format(value.shape))
            for n in range(len(env.n_act)):
                act[n] = np.random.randint(env.n_act[n])
            return act
        else:
            return np.argmax(value)


# ε減衰
class EpsDecGreedy(EpsGreedy):
    def __init__(self, eps, eps_min, eps_decrease):
        super().__init__(eps)
        self.eps_init = eps
        self.eps_min = eps_min
        self.eps_decrease = eps_decrease

    def init_params(self):
        self.eps = self.eps_init

    def update_params(self):
        self.eps = max(self.eps-self.eps_decrease, self.eps_min)

# 
import numpy as np
from copy import deepcopy

class q_learning(object):

    def __init__(self, learning_rate, discount_rate, n_obs):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        # print("n_obs:{}" .format(n_obs.shape))
        self.Q = deepcopy(n_obs)
        self.init_Q = deepcopy(n_obs)

    def update_Q(self, current_state, current_action, reward, next_state, next_action):
        
        a, b, c, d, e, f = current_state+current_action
        # print("next_state+next_action:{}" .format(next_state+next_action))
        _a, _b, _c, _d, _e, _f = next_state+next_action
        # print(self.Q.shape)
        print("pre_update_Q:\t{}" .format(self.Q[a, b, c, d, e, f]))
        maxQ = self.Q[_a, _b, _c, _d, _e, _f]
        # print("maxQ:{}" .format(maxQ))
        TD = (reward
              + self.discount_rate
              * maxQ
              - self.Q[a, b, c, d, e, f])
        self.Q[a, b, c, d, e, f] += self.learning_rate * TD
        print("updated_Q:\t{}" .format(self.Q[a, b, c, d, e, f]))

    def init_params(self):
        self.Q = deepcopy(self.init_Q)


class agent(object):
    def __init__(self, env, learning_rate, discount_rate, eps, eps_min, eps_decrease):
        self.env = env
        # self.state =self.env.st

        self.value_func = q_learning(learning_rate, discount_rate, env.Q_table)
        # print("eps:{}, eps_min:{}, eps_dec:{}" .format(eps, eps_min, eps_decrease))

        # self.policy = EpsDecGreedy(eps, eps_min, 1/eps_decrease)
        self.policy = Greedy()

    def select_action(self, state):
        # print("state:{}" .format(state))
        a, b, c, d = state
        value = self.env.get_action_field(state, self.value_func.Q[a, b, c, d])
        # print("value:{}" .format(value))
        return self.policy.select_action(value, self.env)

    def init_params(self):
        self.value_func.init_params()
        self.policy.init_params()

    def update(self, current_state, current_action, reward, next_state):
        current_action =current_action.tolist()
        next_action = self.select_action(next_state).tolist()
        self.value_func.update_Q(current_state, current_action, reward, next_state, next_action)
    
    def update_eps(self):
        self.policy.update_params()

    # 各行動についてmax Q(s, a)を共有
    def share_Q(self, agents):
        s, a = agents[0].value_func.Q.shape
        value = np.zeros((len(agents), s, a))
        for n in range(len(agents)):
            value[n] = agents[n].value_func.Q
        
        self.value_func.Q = np.max(value, axis=0)
