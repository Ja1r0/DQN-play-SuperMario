# -*- coding: utf-8 -*-

from data import env
import cv2

env = env.Env()


dis = 0.9
REPLAY_MEMORY = 50000


def main():

    max_episodes = 1500
    for episode in range(max_episodes):
        done = False
        step_count = 0
        env.reset()
        obs,_,_,_,_,_,_=env.step(0)
        cv2.imshow('mario',obs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        while not done:
            state, reward, done,s4,s5,s6,s7 = env.step(11)
            # 0 next_state:{ndarray} shape (90,90)
            # 1 reward:{int}
            # 2 done:{bool}
            # 3 state_clear:{bool}
            # 4 max_x:{int}
            # 5 time_out:{bool}
            # 6 now_x:{int}
            step_count += 1



if __name__ == "__main__":
    main()
