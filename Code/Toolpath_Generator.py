import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi
import math
import os
import shutil


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


iter = np.arange(3000)

shutil.rmtree("Toolpaths", ignore_errors=True)
os.mkdir("Toolpaths")

for rot_winkel in range(1):
    for kipp_winkel in range(-50, 50):

        for selection in range(1, 4):

            if selection == 1:
                x = np.cos(np.deg2rad(iter)) * (500 - iter / 3)
                y = np.sin(np.deg2rad(iter)) * (500 - iter / 3)
                z = iter / 10
                tit = "Converging-Diverging Spiral"

            if selection == 2:
                x = np.sin(np.deg2rad(iter)) * (400 - iter / 5)
                y = np.sin(np.deg2rad(iter)) * np.cos(np.deg2rad(iter)) * (500 - iter / 6)
                z = iter / 10
                tit = "Converging Loop"

            if selection == 3:
                x = np.sin(np.deg2rad(iter)) * 200
                y = iter / 3 - (2500 / 3 / 2)
                z = np.sin(np.deg2rad(x)) * 100
                tit = "Pendulum Wave"

            x, y, z = rotate_z_axis(x, y, z, rot_winkel, 0, 0, 0)
            x, y, z = rotate_x_axis(x, y, z, kipp_winkel, 0, 0, 0)

            if rot_winkel == 0 and kipp_winkel == 0:
                plt.rcParams.update({'font.size': 15})
                fig = plt.figure(figsize=(9, 9))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, c=z, cmap='viridis')
                # Set axis labels
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                # Set plot title
                # ax.set_title('3D Scatter Plot')
                ax.set_xlim([min(x) * 1.1, max(x) * 1.1])
                ax.set_ylim([min(y) * 1.1, max(y) * 1.1])
                ax.set_zlim([min(z) * 1.1, max(z) * 1.1])
                ax.view_init(elev=20, azim=-150)

                # plt.title(tit,fontsize=20)
                # ax.tick_params(labelsize=15)
                plt.savefig(f'../Latex/figures/path{selection}.png', dpi=500, bbox_inches="tight", pad_inches=0.3)
                # Display the plot
                # plt.show()
                plt.close()

            if rot_winkel == 0 and (kipp_winkel == -25 or kipp_winkel == 0 or kipp_winkel == 25):
                plt.rcParams.update({'font.size': 15})
                fig = plt.figure(figsize=(9, 9))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, c=z, cmap='viridis')
                # Set axis labels
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                # Set plot title
                # ax.set_title('3D Scatter Plot')
                ax.set_xlim([-500, 500])
                ax.set_ylim([-500, 500])
                ax.set_zlim([-200, 300])
                ax.view_init(elev=20, azim=-150)

                # plt.title(tit,fontsize=20)
                # ax.tick_params(labelsize=15)
                plt.savefig(f'../Latex/figures/path{selection}_kipp_{kipp_winkel}_comparison.png', dpi=500, bbox_inches="tight",
                            pad_inches=0.3)
                # Display the plot
                # plt.show()
                plt.close()

            with open(f'Toolpaths/path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}.npy', 'wb') as f:
                np.save(f, np.array([x, y, z]))
            print(f"path_{selection}_rot_{rot_winkel}_tilt_{kipp_winkel}")
