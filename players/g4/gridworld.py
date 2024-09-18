import constants
from constants import WAIT, LEFT, UP, RIGHT, DOWN

class GridWorld:
    def __init__(self, state, maze_state, goal, is_end_visible):
        self.maze_state = maze_state
        self.state = state
        self.goal = goal
        self.is_end_visible = is_end_visible

    def is_goal(self, state):
        return state == self.goal
    
    # def update_goal(self, goal):
    #     self.goal = goal

    # def update_params(self, timingmazestate, maze_state):
    #     self.maze_state = maze_state
    #     self.is_end_visible = timingmazestate.is_end_visible
    #     if timingmazestate.is_end_visible:
    #         self.goal = (timingmazestate.end_x, timingmazestate.end_y)
    
    def get_next_state(self, state, action):
        # get next state, but check if the move is actually possible
        x, y = state
        cell_states = sorted(self.maze_state[(x,y)], key=lambda x : x[2])
        if action == LEFT and cell_states[action][-1] == constants.OPEN:
            adj_cell = sorted(self.maze_state[(x-1,y)], key=lambda x : x[2])
            if adj_cell[RIGHT][-1] == constants.OPEN:
                x -= 1
        elif action == UP and cell_states[action][-1] == constants.OPEN:
            adj_cell = sorted(self.maze_state[(x,y+1)], key=lambda x : x[2])
            if adj_cell[DOWN][-1] == constants.OPEN:
                y -= 1
        elif action == RIGHT and cell_states[action][-1] == constants.OPEN:
            adj_cell = sorted(self.maze_state[(x+1,y)], key=lambda x : x[2])
            if adj_cell[LEFT][-1] == constants.OPEN:
                x += 1
        elif action == DOWN and cell_states[action][-1] == constants.OPEN:
            adj_cell = sorted(self.maze_state[(x,y-1)], key=lambda x : x[2])
            if adj_cell[UP][-1] == constants.OPEN:
                y += 1
        
        return (x, y)
    
    def step(self, action):
        next_state = self.get_next_state(self.state, action)
        self.state = next_state
        
        if self.is_goal(self.state) and self.is_end_visible:
            return next_state, 100, True
        elif self.is_goal(self.state) and not self.is_end_visible:
            return next_state, 1, True
        else:
            return next_state, -0.01, False
