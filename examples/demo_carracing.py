from typing import Any
import numpy as np
import gym
import time
from carl.envs.box2d.carl_vehicle_racing import CARLVehicleRacingEnv, VEHICLE_NAMES


if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k: int, mod: Any) -> None:
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.SPACE:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +1.0

    def key_release(k: int, mod: Any) -> None:
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    contexts = {i: {"VEHICLE": i} for i in range(len(VEHICLE_NAMES))}
    env = CARLVehicleRacingEnv(contexts=contexts)
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.record_video import RecordVideo

        env = RecordVideo(env=env, video_folder="/tmp/video-test", name_prefix="CARLVehicleRacing")

    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
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
