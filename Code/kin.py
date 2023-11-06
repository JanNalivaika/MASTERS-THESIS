import numpy as np
from random import random
import matplotlib.pyplot as plt

def forward_kinematics(theta):


    # DH parameters
    a = [180, 600, 120, 0, 0, 0]
    alpha = [0.5 * np.pi, 0, 0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi, 0]
    d = [400, 0, 0, 620, 0, 115]

    # Create transformation matrices for each joint
    T = []
    for i in range(len(theta)):
        ct = np.cos(theta[i])
        st = np.sin(theta[i])
        ca = np.cos(alpha[i])
        sa = np.sin(alpha[i])

        # Denavit-Hartenberg transformation matrix
        Ti = np.array([[ct, -st * ca, st * sa, a[i] * ct],
                       [st, ct * ca, -ct * sa, a[i] * st],
                       [0, sa, ca, d[i]],
                       [0, 0, 0, 1]])

        T.append(Ti)

        # Compute the transformation matrix for the end-effector
    T_total = np.eye(4)
    for Ti in T:
        T_total = np.dot(T_total, Ti)

    # Extract XYZ position from the transformation matrix
    position = T_total[:3, 3]

    return position

def acc_calc(positions, time):
    # Differentiate position data twice to get acceleration data
    position_x = np.asarray(positions)[:,0]
    position_y = np.asarray(positions)[:,1]
    position_z = np.asarray(positions)[:,2]
    #time = np.array(range(len(positions)))  # Time value

    acceleration_x = np.gradient(np.gradient(position_x, time), time)
    acceleration_y = np.gradient(np.gradient(position_y, time), time)
    acceleration_z = np.gradient(np.gradient(position_z, time), time)

    # Calculate the magnitude of acceleration
    acceleration = np.sqrt(acceleration_x ** 2 + acceleration_y ** 2 + acceleration_z ** 2)
    return acceleration


def count_direction_changes(data):
    count = 0

    # Loop through the data starting from the second element
    for i in range(1, len(data)-1):
        # Check if the direction changes from "up" to "down"
        if (data[i - 1] < data[i] > data[i + 1]) or (data[i - 1] > data[i] < data[i + 1]):
            count += 1

    return count


import math


def calculate_total_distance(coordinates):
    total_distance = 0

    for i in range(len(coordinates) - 1):
        x1, y1, z1 = coordinates[i]
        x2, y2, z2 = coordinates[i + 1]

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        total_distance += distance

    return total_distance

momory =[]
for tests in range(80):
    pos = []
    ang = []

    all_pos = np.load(f"{tests}_angles.npy")

    time = np.linspace(0, 2.0, num=len(all_pos))
    # Calculate forward kinematics
    for x in range(len(all_pos)):
        # Example joint angles (in radians)

        #theta = [np.cos(x)*np.sin(x**2-x), np.sin(x * 2), np.cos(x), 3 * np.cos(x), np.sin(x) * np.cos(x), np.sin(x) * np.sin(x)]
        theta = all_pos[x]
        #theta = np.degrees(theta)
        ang.append(list(theta))
        end_effector_pose = forward_kinematics(theta)
        pos.append(end_effector_pose)





    acc = acc_calc(pos,time)

    #time = np.array(range(len(acc)))  # Time value
    jerk = np.gradient(np.gradient(acc, time), time)

    dirchange = count_direction_changes(np.asarray(ang)[:,0])
    dist = calculate_total_distance(pos)

    total_changes = 0
    for x in range(6):
        total_changes += count_direction_changes(np.asarray(ang)[:, x])


    # Print the resulting transformation matrix
    #print("End Effector Pose:")
    #print(pos)


    # Create a figure with three subplots stacked vertically
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    plt.suptitle(f"Changes in Joint 1: {dirchange}, Total changes = {total_changes},  Distance = {dist}")
    # Plot the first time series on the top subplot
    ax1.plot(np.asarray(ang)[:,0])
    ax1.set_ylabel('Angle 1')

    # Plot the second time series on the middle subplot
    ax2.plot(np.asarray(pos)[:,0])
    ax2.set_ylabel('X pos')

    # Plot the third time series on the bottom subplot
    ax3.plot(np.asarray(acc))
    ax3.set_ylabel('Acc')

    ax4.plot(np.asarray(jerk))
    ax4.set_ylabel('Jerk')

    # Set the x-axis label for the bottom subplot
    ax3.set_xlabel('Time')

    # Adjust the spacing between subplots
    plt.tight_layout()
    momory.append(dirchange)


    # Show the plot
    #plt.show()
    plt.close()


plt.plot(momory)
plt.show()