#!/usr/bin/env python3

from visual_kinematics.RobotSerial import *


import numpy as np
from math import pi
import math
import os
import shutil
import time


d =     [600,       0,      0,   800,    0,     200]
a =     [200,    900,    150,     0,    150,       0]
alpha = [90,       0,    90,    -90,    90,      -90]

theta = [0,0,0,0,0,0]


dh_params = np.array(np.transpose(np.vstack((d,a,np.radians(alpha),theta))))

np.set_printoptions(precision=5, suppress=True)



#shutil.rmtree("Joint_angles", ignore_errors=True)
#os.mkdir("Joint_angles")


selection = 1
rot_winkel = 0
kipp_winkel = 0
c_axis = 0

print("Started")
t1 = time.time()
sol =  [0, 80, -27, 180, -128, 0]
#sol =  [0, 00, 0, 180, 0, 0]
#sol=  [0, 80, 0, 180, 0, 0]
sol = np.deg2rad(sol)
pos = []

xyz = np.load(f"Toolpaths/path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}.npy")

xyz[0]+= 1000
xyz[1] += 0
xyz[2] += 600

#600*600 fläche


#for iter in range(len(xyz[0])):

for iter in range(0,len(xyz[0]),1):



    robot = RobotSerial(dh_params)

    if iter%300 == 0: print(iter)

    abc = np.array([np.deg2rad(kipp_winkel),0,np.deg2rad(c_axis)])

    tcp = np.array([[xyz[0,iter]],[xyz[1,iter]],[xyz[2,iter]]])

    end = Frame.from_euler_3(abc, tcp)
    robot.inverse(end,sol)
    sol = robot.axis_values
    pos.append(sol)
np.save(f"path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}_C_{c_axis}.npy", pos)
