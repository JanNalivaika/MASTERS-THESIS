#!/usr/bin/env python3

from visual_kinematics.RobotSerial import *


import numpy as np
from math import pi
import math
import os
import shutil
import time


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


for selection in [1,2,3]:
    for rot_winkel in range(1):
        for kipp_winkel in range(-45,46,2):
            for c_axis in range(-135,140,5):

                if os.path.exists(f"Joint_angles_lowres/path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}_C_{c_axis}.npy"):
                    print("exists")
                    pass
                else:

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

                    xyz = np.load(f"Toolpaths/path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}.npy")

                    xyz[0]+= 1000
                    xyz[1] += 0
                    xyz[2] += 600

                    #600*600 fläche


                    #for iter in range(len(xyz[0])):

                    for iter in range(0,len(xyz[0]),3):



                        robot = RobotSerial(dh_params)

                        #if iter%300 == 0: print(iter)
                        #abc = np.array([c_axis,-np.pi*1.5,-np.deg2rad(kipp_winkel)])
                        abc = np.array([np.deg2rad(kipp_winkel),0,np.deg2rad(c_axis)])

                        tcp = np.array([[xyz[0,iter]],[xyz[1,iter]],[xyz[2,iter]]])

                        end = Frame.from_euler_3(abc, tcp)
                        robot.inverse(end,sol)
                        sol = robot.axis_values
                        pos.append(sol)

                    #np.save(f"Joint_angles/path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}_C_{c_axis}.npy", pos)
                    np.save(f"Joint_angles_lowres/path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}_C_{c_axis}.npy", pos)
                    print(f"DONE: path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}_C_{c_axis}   TIME: {np.ceil(time.time()-t1)}s")





