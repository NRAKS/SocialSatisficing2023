import numpy as np

class GlidWorld(object):
    # 簡単なステージを作る
    def __init__(self, row, col, start, goal=None):
        self.row = row
        self.col = col
        self.start_state = start
        self.goal_state = goal
        self.num_state = row * col
        self.num_action = 5
        self.reward = 0
        self.done = False

    # 座標に変換
    def coord_to_state(self, row, col):
        return ((row * self.col) + col)

    # 座標からx軸を算出
    def state_to_row(self, state):
        return ((int)(state / self.col))

    # 座標からy軸を算出
    def state_to_col(self, state):
        return (state % self.col)

    # 次の座標を算出
    def evaluate_next_state(self, state, action):
        UPPER = 0
        LOWER = 1
        LEFT = 2
        RIGHT = 3
        STOP = 4

        row = self.state_to_row(state)
        col = self.state_to_col(state)

        if action == UPPER:
            if (row) > 0:
                row -= 1
        elif action == LOWER:
            if (row) < (self.row-1):
                row += 1
        elif action == RIGHT:
            if (col) < (self.col-1):
                col += 1
        elif action == LEFT:
            if (col) > 0:
                col -= 1
        elif action == STOP:
            pass

        self.next_state = self.coord_to_state(row, col)

    # 報酬判定
    def evaluate_reward(self, state):
        if state == self.goal_state:
            self.reward = 1
        else:
            self.reward = 0

    def evaluate_done(self):
        if self.reward > 0:
            self.done = True
        else:
            self.done = False

    # 行動
    def step(self, state, action):
        self.evaluate_next_state(state, action)
        self.evaluate_reward(self.next_state)
        self.evaluate_done()

        return self.next_state, self.reward, self.done
    

class SuboptimaWorld(GlidWorld):
    def __init__(self, row, col, start, goal=None):
        super.__init__(row, col, start, goal=None)

        self.world = np.zeros([row, col])
        list_reward = [1, 2, 3, 4, 5, 6, 7, 8]

        self.world[0, 2] = list_reward[7]
        self.world[0, row-3] = list_reward[0]
        self.world[2, 0] = list_reward[2]
        self.world[2, row-1] = list_reward[4]
        self.world[col-3, 0] = list_reward[5]
        self.world[col-3, row-1] = list_reward[3]
        self.world[col-1, 2] = list_reward[1]
        self.world[col-1, row-3] = list_reward[6]

    def evaluate_reward(self, state):
        row = self.state_to_row(state)
        col = self.state_to_col(state)
        self.reward = self.world(row, col)
        return self.reward
