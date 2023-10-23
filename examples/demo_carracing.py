"""
Code adapted from gym.envs.box2d.car_racing.py
"""

from typing import Any
import numpy as np
import gymnasium as gym
import time
import pygame
from carl.envs.box2d.carl_vehicle_racing import CARLVehicleRacingEnv, VEHICLE_NAMES

if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])

    def register_input():
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    global restart
                    restart = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

    contexts = {i: {"VEHICLE": i} for i in range(len(VEHICLE_NAMES))}
    env = CARLVehicleRacingEnv(contexts=contexts)
    env.render()
    record_video = False
    if record_video:
        from gymnasium.wrappers.record_video import RecordVideo

        env = RecordVideo(
            env=env, video_folder="/tmp/video-test", name_prefix="CARLVehicleRacing"
        )

    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, done, info = env.step(a)
            time.sleep(0.025)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or not isopen:
                break
    env.close()
