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


def rotate_z_axis(x, y, z, angle, origin_x, origin_y, origin_z):
    # Adjust coordinates relative to the origin point
    x -= origin_x
    y -= origin_y
    z -= origin_z

    # Convert angle to radians
    angle_rad = math.radians(angle)

    # Apply rotation formulas
    x_rotated = (x * math.cos(angle_rad)) - (y * math.sin(angle_rad))
    y_rotated = (x * math.sin(angle_rad)) + (y * math.cos(angle_rad))

    # Add back the origin point coordinates
    x_rotated += origin_x
    y_rotated += origin_y
    z_rotated = z + origin_z

    return x_rotated, y_rotated, z

"""
Working but "wrong"
d =     [600,       0,      0,   800,    0,     -100]
a =     [200,    900,    150,     0,    150,       150]
alpha = [90,       0,    90,    -90,    90,      00]"""

d =     [600,       0,      0,   800,    0,     200]
a =     [200,    900,    150,     0,    150,       0]
alpha = [90,       0,    90,    -90,    90,      -90]

theta = [0,0,0,0,0,0]


dh_params = np.array(np.transpose(np.vstack((d,a,np.radians(alpha),theta))))

np.set_printoptions(precision=5, suppress=True)



#shutil.rmtree("Joint_angles", ignore_errors=True)
#os.mkdir("Joint_angles")


for selection in [4]:
    for rot_winkel in range(1):
        for kipp_winkel in range(1): #-45,46,2
            for c_axis in range(-135,140,5): # -135,140,5

                if os.path.exists(f"Joint_angles_lowres/path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}_C_{c_axis}.npy"):
                    print("exists")
                    #pass
                else:
                    xyz = np.load(f"RealG.npy")
                    print("Started")
                    t1 = time.time()
                    if selection == 1:
                        sol =  [0, 50, 0, 100, 100, -100]
                    if selection == 2:
                        sol = [0, 50, -50, 100, 100, 220]
                    else:
                        sol = [25, 75, -30, -75, -80, 75]

                    sol = np.deg2rad(sol)
                    pos = []

                    #xyz = np.load(f"Toolpaths/path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}.npy")


                    xyz[0]+= 1000
                    xyz[1] += 0
                    xyz[2] += 600



                    #600*600 fläche


                    #for iter in range(len(xyz[0])):

                    for iter in range(0,len(xyz[0]),1):

                        xyz[0,iter], xyz[1,iter], xyz[2,iter] = rotate_x_axis(xyz[0,iter], xyz[1,iter], xyz[2,iter], xyz[6,iter], origin_x=1000, origin_y=0, origin_z=600)

                        robot = RobotSerial(dh_params)

                        if iter%5000 == 0: print(f"{iter} / {len(xyz[0])}")
                        #abc = np.array([c_axis,-np.pi*1.5,-np.deg2rad(kipp_winkel)])
                        abc = np.array([np.deg2rad(xyz[6,iter]+xyz[3,iter]),0,np.deg2rad(c_axis)])

                        tcp = np.array([[xyz[0,iter]],[xyz[1,iter]],[xyz[2,iter]]])

                        end = Frame.from_euler_3(abc, tcp)
                        robot.inverse(end,sol)
                        sol = robot.axis_values
                        pos.append(sol)

                    #np.save(f"Joint_angles/path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}_C_{c_axis}.npy", pos)
                    #np.save(f"Joint_angles_lowres/path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}_C_{c_axis}.npy", pos)
                    np.save(f"RealG_angles/path_4_rot_0_tilt_conti._C_{c_axis}.npy",pos)

                    print(f"DONE: path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}_C_{c_axis}   TIME: {np.ceil(time.time()-t1)}s")





