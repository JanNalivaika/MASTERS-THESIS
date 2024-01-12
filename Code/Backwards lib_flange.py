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

for selection in [1, 2, 3]:
    for kipp_winkel in range(1):  # -45,46,2
        for c_axis in range(-135, 140, 5):  # -135,140,5 [-120]:#

            if os.path.exists(
                    f"XXXXXXXXJoint_angles_lowres_flange/path_{selection}_rot_0_tilt_{kipp_winkel}_C_{c_axis}.npy"):
                print("exists")
            else:
                xyz = np.load(f"Toolpaths/path_{selection}_rot_0_tilt_{kipp_winkel}.npy")

                xyz[0] += 900
                xyz[1] += 0 - 350 * np.sin(np.radians(kipp_winkel))
                xyz[2] += 600 + 350 * np.cos(np.radians(kipp_winkel))
                if selection == 1:
                    sol = [-12, 7658, -2572, -18, 44, 13]
                if selection == 2:
                    sol = [-6, 7e+03, -2e+03, -1e+01, 3.e+01, 9e+00]
                if selection == 3:
                    sol = [-9, 7674, -2587, -13, 44, 9]
                pos = []

                for iter in range(0, len(xyz[0]), 1):
                    robot = RobotSerial(dh_params)

                    # if iter % 100 == 0: print(f"{iter} / {len(xyz[0])}")
                    # abc = np.array([c_axis,-np.pi*1.5,-np.deg2rad(kipp_winkel)])
                    abc = np.array(np.radians([90 + kipp_winkel, 90 + c_axis, 90]))

                    tcp = np.array([[xyz[0, iter]], [xyz[1, iter]], [xyz[2, iter]]])

                    end = Frame.from_euler_3(abc, tcp)
                    robot.inverse(end, sol)
                    sol = robot.axis_values
                    pos.append(sol)

                # np.save(f"Joint_angles_lowres_flange/path_{selection}_rot_0_tilt_{kipp_winkel}_C_{c_axis}.npy", pos)
                np.save(f"Joint_angles_flange/path_{selection}_rot_0_tilt_{kipp_winkel}_C_{c_axis}.npy", pos)
                print(f"path_{selection}_rot_0_tilt_{kipp_winkel}_C_{c_axis}.npy")
