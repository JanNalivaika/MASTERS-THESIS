#!/usr/bin/env python3

from visual_kinematics.RobotSerial import *

import numpy as np
from math import pi
import math
import os
import shutil
import time


def rotate_x_axis(x, y, z, angle, origin_x=0, origin_y=0, origin_z=0):
    # Adjust coordinates relative to the origin point
    x -= origin_x
    y -= origin_y
    z -= origin_z

    # Convert angle to radians
    angle_rad = math.radians(angle)

    # Apply rotation formulas
    y_rotated = (y * math.cos(angle_rad)) - (z * math.sin(angle_rad))
    z_rotated = (y * math.sin(angle_rad)) + (z * math.cos(angle_rad))

    # Add back the origin point coordinates
    x_rotated = x + origin_x
    y_rotated += origin_y
    z_rotated += origin_z

    return x_rotated, y_rotated, z_rotated


d = [400, 0, 0, 600, 0, 200]
a = [200, 800, 150, 0, 0, 0]
alpha = [90, 0, 90, -90, 90, 0]

theta = [0, 0, 0, 0, 0, 0]

dh_params = np.array(np.transpose(np.vstack((d, a, np.radians(alpha), theta))))

np.set_printoptions(precision=5, suppress=True)

for selection in [1,2,3]:
    for kipp_winkel in range(-45,46,2):  # -45,46,2
        for c_axis in range(-135,140,5):  # -135,140,5 [-120]:#

            if os.path.exists(
                    f"XXJoint_angles_lowres_flange/path_{selection}_rot_0_tilt_{kipp_winkel}_C_{c_axis}.npy"):
                print("exists")
            else:
                if selection ==4 :
                    xyz = np.load(f"RealG.npy")
                else:
                    xyz = np.load(f"Toolpaths/path_{selection}_rot_0_tilt_{kipp_winkel}.npy")

                xyz[0] += 800
                xyz[1] += 0 - 350 * np.sin(np.radians(kipp_winkel))
                xyz[2] += 400 + 350 * np.cos(np.radians(kipp_winkel))
                sol = [0,0,0,0,0,0]
                if selection == 1:
                    sol = [ 0,  7.0e+01, -3.8e+01,  0,  5.5e+01, 0]
                if selection == 2:
                    sol = [ 0,  7.0e+01, -3.8e+01,  0,  5.5e+01, 0]
                if selection == 3:
                    sol = [ 0,  7.0e+01, -3.8e+01,  0,  5.5e+01, 0]
                pos = []

                for iter in range(0, len(xyz[0]), 3):
                    robot = RobotSerial(dh_params)

                    #if iter % 1000 == 0: print(f"{iter} / {len(xyz[0])}")
                    # abc = np.array([c_axis,-np.pi*1.5,-np.deg2rad(kipp_winkel)])
                    abc = np.array(np.radians([90 + kipp_winkel, 90 + c_axis, 90]))
                    tcp = np.array([[xyz[0, iter]], [xyz[1, iter]], [xyz[2, iter]]])

                    if selection == 4:
                        xyz[0, iter], xyz[1, iter], xyz[2, iter] = rotate_x_axis(xyz[0, iter], xyz[1, iter],
                                                                                 xyz[2, iter], xyz[6, iter],
                                                                                 origin_x=800, origin_y=0,
                                                                                 origin_z=400+350)
                        #print((xyz[6,iter]+xyz[3,iter]))
                        tcp = np.array([[xyz[0, iter]], [xyz[1, iter]], [xyz[2, iter]]])






                    end = Frame.from_euler_3(abc, tcp)
                    robot.inverse(end, sol)
                    sol = robot.axis_values
                    pos.append(sol)

                #np.save(f"Joint_angles_lowres_flange/path_{selection}_rot_0_tilt_{kipp_winkel}_C_{c_axis}.npy", pos)
                #np.save(f"Joint_angles_flange/path_{selection}_rot_0_tilt_{kipp_winkel}_C_{c_axis}.npy", pos)
                #np.save(f"Joint_angles_lowres_flange/path_{selection}_rot_0_tilt_{kipp_winkel}_C_{c_axis}.npy", pos)
                print(f"path_{selection}_rot_0_tilt_{kipp_winkel}_C_{c_axis}.npy")
