# 03230235_gridworld.py
# GridWorld Environment

import numpy as np

class GridWorld:
    def __init__(self):
        self.grid_size = 5
        self.walls = [(1,1), (1,3), (2,3), (3,1)]
        self.start = (0, 0)
        self.goal = (4, 4)
        self.current_pos = self.start

    def reset(self):
        self.current_pos = self.start
        return self.state_to_index(self.current_pos)

    def step(self, action):
        row, col = self.current_pos
        if action == 0:  # UP
            new_pos = (row-1, col)
        elif action == 1:  # DOWN
            new_pos = (row+1, col)
        elif action == 2:  # LEFT
            new_pos = (row, col-1)
        elif action == 3:  # RIGHT
            new_pos = (row, col+1)
        else:
            raise ValueError("Invalid action")

        # Check boundaries and walls
        if (0 <= new_pos[0] < self.grid_size and
            0 <= new_pos[1] < self.grid_size and
            new_pos not in self.walls):
            self.current_pos = new_pos
            reward = -0.1
        else:
            reward = -1  # hit wall

        done = self.current_pos == self.goal
        if done:
            reward = 10

        return self.state_to_index(self.current_pos), reward, done

    def state_to_index(self, state):
        """Convert (row, col) to unique state index"""
        row, col = state
        return row * self.grid_size + col

    def index_to_state(self, index):
        """Convert state index back to (row, col)"""
        row = index // self.grid_size
        col = index % self.grid_size
        return (row, col)
