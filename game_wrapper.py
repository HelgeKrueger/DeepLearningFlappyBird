import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import cv2
import numpy as np

class Game():
    def __init__(self):
        self.game_state = game.GameState()

    def run_action(self, action):
        a_t = np.zeros([2])
        a_t[action] = 1

        x_t, r_0, terminal = self.game_state.frame_step(a_t)
        x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)

        return r_0, x_t, terminal

