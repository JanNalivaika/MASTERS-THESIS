#!/usr/bin/env python3

from visual_kinematics.RobotSerial import *
import numpy as np
from math import pi
import math


def tilting_spiral(num_points, radius, angle_increment, tilt_angle, rise_increment):
    coordinates = []
    current_angle = 0
    current_height = 0

    for _ in range(num_points):
        x = radius * math.cos(math.radians(current_angle))
        y = radius * math.sin(math.radians(current_angle)) * math.cos(math.radians(tilt_angle))
        z = current_height
        coordinates.append((x, y, z))
        current_angle += angle_increment
        current_height += rise_increment

    return coordinates

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

#spiral_coords  = tilting_spiral(1000, 100, 10, 30, 1)
def main(reps):
    np.set_printoptions(precision=5, suppress=True)
    pos = []
    sol = [0,0,0,0,0,0]

    for iter in range(1000):
        plt.close()
        dh_params = np.array([[400,   180.,     0.5 * pi,   0],
                              [0.,      600,  0,         0],
                              [0.,      120, 0.5* pi,         0],
                              [620,  0.,     0.5 * pi,  0],
                              [0,   0.,     0.5 * pi,   0],
                              [115,   0.,     0.,         0]])

        robot = RobotSerial(dh_params)

        # =====================================
        # inverse
        # =====================================


        x = np.cos(np.deg2rad(iter))*(500 - iter/3) + 700
        y = np.sin(np.deg2rad(iter))*(500 - iter/3)
        z = 50+iter/3
        #[x,y,z] = spiral_coords[iter]
        kippen = 10*reps

        x, y,z = rotate_x_axis(x, y, z, kippen, 700, 0, 50)

        xyz = np.array([[x], [y], [z]])




        if iter%10 == 0: print(iter)
        abc = np.array([0,-np.pi,-np.deg2rad(kippen)])
        end = Frame.from_euler_3(abc, xyz)
        robot.inverse(end)

        #print(xyz)
        #print("inverse is successful: {0}".format(robot.is_reachable_inverse))
        #print("axis values: \n{0}".format(robot.axis_values))
        sol = robot.axis_values
        pos.append(robot.axis_values)
        #print(robot.axis_values)
        #robot.show()
    #print(reps)
    np.save(f"{reps}_angles", pos)


    # example of unsuccessful inverse kinematics
    #xyz = np.array([[2.2], [0.], [1.9]])
    #end = Frame.from_euler_3(abc, xyz)
    #robot.inverse(end)

    #print("inverse is successful: {0}".format(robot.is_reachable_inverse))
    #robot.show()


for reps in range(10):
    main(reps)
    print(f"FINISHED {reps}")