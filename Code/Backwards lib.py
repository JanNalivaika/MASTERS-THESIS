#!/usr/bin/env python3

from visual_kinematics.RobotSerial import *
import numpy as np
from math import pi


def main(reps):
    np.set_printoptions(precision=2, suppress=True)
    pos = []
    sol = [0,0,0,0,0,0]

    for iter in range(360):
        plt.close()
        dh_params = np.array([[400,   180.,     0.5 * pi,   0],
                              [0.,      600,  0,         0],
                              [0.,      120, 0.5 * pi,         0],
                              [620,  0.,     0.5 * pi,  0],
                              [0,   0.,     0.5 * pi,   0],
                              [115,   0.,     0.,         0]])

        robot = RobotSerial(dh_params)

        # =====================================
        # inverse
        # =====================================

        xyz = np.array([[np.sin(np.deg2rad(iter))*400+0], [np.cos(np.deg2rad(iter))*200+500], [100+iter*2]])
        #if iter%100 == 0: print(iter)
        abc = np.array([0,reps,  0])
        end = Frame.from_euler_3(abc, xyz)
        robot.inverse(end)

        #print(xyz)
        #print("inverse is successful: {0}".format(robot.is_reachable_inverse))
        #print("axis values: \n{0}".format(robot.axis_values))
        sol = robot.axis_values
        pos.append(robot.axis_values)
        #print(robot.axis_values)
        #robot.show()
    print(reps)
    np.save(f"{reps}_angles", pos)


    # example of unsuccessful inverse kinematics
    #xyz = np.array([[2.2], [0.], [1.9]])
    #end = Frame.from_euler_3(abc, xyz)
    #robot.inverse(end)

    #print("inverse is successful: {0}".format(robot.is_reachable_inverse))
    #robot.show()

if __name__ == "__main__":
    for reps in range(100):
        main(reps)