# %%
# ライブラリのインポート
import torch
from copy import deepcopy
from tqdm import tqdm
import random
import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %%
class State():
    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


# %%
from typing import Optional, List, Tuple, Literal


def clone_states_list(states_list: List[State]):
    return [s.clone() for s in states_list]


class SuikawariEnv:
    """ Suikawari Grid World Environment for Multi-agents
    The background of the field is represented as '0' (White)
    Suikas are represented as '1' (Green)
    The agents are represented as '2' or Int 
        greater than 2 and up to 5. (Red, Blue, Yellow, Magenta)
    """
    UP = 0
    DOWN = 2
    LEFT = 1
    RIGHT = 3
    ACTIONS = [UP, DOWN, LEFT, RIGHT]

    FIELD = 0
    SUIKA = 1
    AGENT = 2

    def __init__(self,
                 n_suika: int,
                 n_agents: int,
                 seed: Optional[int] = None,
                 grid_size: int = 9,
                 default_reward: float = -0.04):
        if n_agents > 4:
            raise ValueError('エージェントは4つまで')

        self.config_dict = {
            'n_agents': n_agents,
            'n_suika': n_suika,
            'seed': seed,
            'grid_size': grid_size,
            'default_reward': default_reward,
        }

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Make Suikawari-Grid
        self.n_suika = n_suika
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype='int')
        # Set suikas' position
        _pos_col_cands = list(range(grid_size))
        _pos_row_cands = list(range(grid_size))
        self.suika_list = []
        for _ in range(self.n_suika):
            suika_pos_col = random.sample(_pos_col_cands, k=1)[0]
            suika_pos_row = random.sample(_pos_row_cands, k=1)[0]
            self.suika_list.append(State(suika_pos_col, suika_pos_row))
            self.grid[suika_pos_col][suika_pos_row] = self.SUIKA

        self.n_got_suika = 0
        self.states_list = []
        for _ in range(self.n_agents):
            start_row = np.random.randint(grid_size)
            start_col = np.random.randint(grid_size)
            self.states_list.append(State(start_row, start_col))

        self.defaul_reward = default_reward

    def step(self, action: Literal[0, 1, 2, 3],
             agent_id: int) -> Tuple[List[State], float, bool]:
        if agent_id < self.n_agents: raise ValueError("エージェントIDがエージェント数より多い")
        state = self.states_list[agent_id]
        next_state = self._move(state, action)
        if next_state is not None:
            self.states_list[agent_id] = next_state
        reward, done = self._reward_func(next_state)
        return self.states_list, reward, done

    def _move(self, state: State, action) -> State:
        next_state = state.clone()
        # Execute an action (move).
        if action == self.UP:
            next_state.row -= 1
        elif action == self.DOWN:
            next_state.row += 1
        elif action == self.LEFT:
            next_state.column -= 1
        elif action == self.RIGHT:
            next_state.column += 1

        if next_state in self.states_list:
            # 同じ座標のエージェントがすでに存在する場合は移動しない。
            return state
        # 端に到達した時はそれ以上移動しない。
        next_state.row = max(next_state.row, 0)
        next_state.row = min(next_state.row, self.grid_size - 1)
        next_state.column = max(next_state.column, 0)
        next_state.column = min(next_state.column, self.grid_size - 1)
        return next_state

    def _reward_func(self, state: State) -> Tuple[float, bool]:
        """スイカ割りの進捗に応じて報酬を与える。
        4個中1つ割ったら0.25、もひとつ割ったら0.5のような。
        全て割ったら終了。
        スイカは割られたら消滅する。(セルの値をFIELDに書き換える)
        """
        reward = self.defaul_reward
        done = False

        attribute = self.grid[state.row][state.column]
        if attribute == self.SUIKA:
            self.n_got_suika += 1
            reward = self.n_got_suika / self.n_suika
            self.grid[state.row][state.column] = self.FIELD

        if self.n_got_suika == self.n_suika:
            done = True
        return reward, done

    def reset(self) -> Tuple[np.ndarray, List[State]]:
        self.__init__(**self.config_dict)
        return self.grid, self.states_list

    def observation(self,
                    agent_id: Optional[int] = None,
                    partial: bool = False) -> torch.Tensor:
        assert agent_id is not None if partial else True  #部分観測を得る場合は必ずエージェントを指定する。
        grid = torch.tensor(self.grid)

        # position of reward(suika) cell
        pos_reward = (grid == self.SUIKA)

        # initalize image (shape = (3 (r,g,b), *grid.shape))
        grid_img = torch.zeros((*grid.shape, 3)).float()

        # color
        grid_img[:, :, 1] += pos_reward
        colors = (torch.Tensor([1.0, 0, 0]), torch.Tensor([0, 0, 1.0]),
                  torch.Tensor([1.0, 1.0, 0]), torch.Tensor([1.0, 0, 1.0]))

        for color, state in zip(colors, self.states_list):
            row = state.row
            col = state.column
            grid_img[row, col] = color

        if partial and agent_id is not None:
            state = self.states_list[agent_id]
            row = state.row
            col = state.column
            row_l = max(row - 1, 0)
            row_u = min(row + 2, self.grid_size)
            col_l = max(col - 1, 0)
            col_u = min(col + 2, self.grid_size)
            print(row_l, row_u, col_l, col_u)
            mask = torch.zeros_like(grid_img)
            mask[row_l:row_u, col_l:col_u, :] = 1.0
            print(row, col, mask.shape)
            # return mask
            grid_img *= mask

        return grid_img.float()


# %%
env = SuikawariEnv(n_suika=2, n_agents=4, seed=10)


def one_step(dirction, id):
    states, reward, done = env.step(dirction, agent_id=id)
    # im = env.observation(partial=True, agent_id=id)
    im = env.observation()
    plt.imshow(im)
    plt.show()
    print(reward)


one_step(2, 0)
one_step(2, 0)
one_step(2, 0)
one_step(1, 3)
one_step(1, 3)
one_step(1, 3)
env.reset()
one_step(2, 0)
one_step(2, 0)
one_step(2, 0)
one_step(1, 3)
one_step(1, 3)
one_step(1, 3)

# %%



