#!/usr/bin/env python3

from visual_kinematics.RobotSerial import *
import numpy as np
from math import pi


a = [180, 600, 120, 0, 0, 0]
alpha = [90, 0, 90, 90, 90, 0]
d = [400, 0, 0, 620, 0, 115]

def main():
    np.set_printoptions(precision=3, suppress=True)

    dh_params = np.array([[400,   180.,     0.5 * pi,   0.],
                          [0.,      600,  0,         0],
                          [0.,      120, 0.5 * pi,         0.],
                          [620,  0.,     0.5 * pi,  0],
                          [0,   0.,     0.5 * pi,   0.],
                          [115,   0.,     0.,         0.]])
    robot = RobotSerial(dh_params)

    # =====================================
    # forward
    # =====================================

    theta = np.array([0., 0., 0, 0., 0., 0.])
    f = robot.forward(theta)

    print("-------forward-------")
    print("end frame t_4_4:")
    print(f.t_4_4)
    print("end frame xyz:")
    print(f.t_3_1.reshape([3, ]))
    print("end frame abc:")
    print(f.euler_3)
    print("end frame rotational matrix:")
    print(f.r_3_3)
    print("end frame quaternion:")
    print(f.q_4)
    print("end frame angle-axis:")
    print(f.r_3)

    robot.show()


if __name__ == "__main__":
    main()