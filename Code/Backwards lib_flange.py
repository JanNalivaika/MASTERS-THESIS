#!/usr/bin/env python3

from visual_kinematics.RobotSerial import *

import numpy as np
from math import pi
import math
import os
import shutil
import time

d = [600, 0, 0, 800, 0, 200]
a = [200, 900, 150, 0, 0, 0]
alpha = [90, 0, 90, -90, 90, 0]

theta = [0, 0, 0, 0, 0, 0]

dh_params = np.array(np.transpose(np.vstack((d, a, np.radians(alpha), theta))))

np.set_printoptions(precision=5, suppress=True)

C = 30
T = 25
xyz = np.load(f"Toolpaths/path_1_rot_0_tilt_{T}.npy")

sol = [0, 0, 0, 0, 0, 0]
pos = []


xyz[0] += 1000
xyz[1] += 0-350*np.sin(np.radians(T))
xyz[2] += 600+350*np.cos(np.radians(T))



for iter in range(0, len(xyz[0]), 10):

    robot = RobotSerial(dh_params)

    if iter % 100 == 0: print(f"{iter} / {len(xyz[0])}")
    # abc = np.array([c_axis,-np.pi*1.5,-np.deg2rad(kipp_winkel)])
    abc = np.array(np.radians([90+T, 90+C,90]))

    tcp = np.array([[xyz[0, iter]], [xyz[1, iter]], [xyz[2, iter]]])

    end = Frame.from_euler_3(abc, tcp)
    robot.inverse(end, sol)
    sol = robot.axis_values
    pos.append(sol)


np.save(f"wrong0.npy",pos)
