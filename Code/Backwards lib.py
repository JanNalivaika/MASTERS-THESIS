#!/usr/bin/env python3

from visual_kinematics.RobotSerial import *


import numpy as np
from math import pi
import math
import os
import shutil
import time


d =     [570,       0,      0,   935,    0,     0]
a =     [175,    890,    100,     0,    140,       50]
alpha = [90,       0,    90,    -90,    90,      00]

theta = [0,0,0,0,0,0]


dh_params = np.array(np.transpose(np.vstack((d,a,np.radians(alpha),theta))))

np.set_printoptions(precision=5, suppress=True)



shutil.rmtree("Joint_angles", ignore_errors=True)
os.mkdir("Joint_angles")


for selection in range(1,4):
    for rot_winkel in range(1): #-25, 26
        for kipp_winkel in range(1): #-25, 26 ,10
            for c_axis in list(np.arange(-0.6,0.7,0.2)):  #list(np.arange(-0.6,0.7,0.2)) range(1)
                c_axis = np.round(c_axis,2)

                print("Started")
                t1 = time.time()
                pos = []

                xyz = np.load(f"Toolpaths/path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}.npy")

                xyz[0]+= 1300
                xyz[1] += 0
                xyz[2] += 800

                #600*600 fläche


                #for iter in range(len(xyz[0])):

                for iter in range(1000):



                    robot = RobotSerial(dh_params)

                    if iter%1000 == 0: print(iter)
                    #abc = np.array([c_axis,-np.pi*1.5,-np.deg2rad(kipp_winkel)])
                    abc = np.array([np.deg2rad(kipp_winkel),0,c_axis])

                    tcp = np.array([[xyz[0,iter]],[xyz[1,iter]],[xyz[2,iter]]])

                    end = Frame.from_euler_3(abc, tcp)
                    robot.inverse(end)
                    pos.append(robot.axis_values)

                np.save(f"Joint_angles/path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}_C_{np.round(c_axis,2)}", pos)
                print(f"working on element: path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}_C_{c_axis}   TIME: {np.ceil(time.time()-t1)}s")





