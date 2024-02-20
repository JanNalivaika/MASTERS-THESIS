import numpy as np
import matplotlib.pyplot as plt
import math


def rotate_x_axis(x, y, z, angle, origin_x, origin_y, origin_z):
    # Adjust coordinates relative to the origin point
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

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

    return list(x_rotated), list(y_rotated), list(z_rotated)


# DH parameters
a = [200, 800, 150, 0, 0, 0]
d = [400, 0, 0, 600, 0, 200]
alpha = [90, 0, 90, -90, 90, 0]


# import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 20})

# Homogeneous transformation matrix
def dh_transform(a, alpha, d, theta):
    transform = np.array(
        [[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
         [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
         [0, np.sin(alpha), np.cos(alpha), d],
         [0, 0, 0, 1]])
    return transform


# Forward kinematics
def forward_kinematics(a, alpha, d, theta):
    num_links = len(a)
    T = np.eye(4)
    transformations = []

    for i in range(num_links):
        T_i = dh_transform(a[i], np.radians(alpha[i]), d[i], np.radians(theta[i]))
        T = np.dot(T, T_i)
        transformations.append(T)

    position = T[:3, 3]
    rmatrix = T[:3, :3]
    return transformations, position, rmatrix





x_coords = []
y_coords = []
z_coords = []
angel_before = 0
col = []
fig = plt.figure(figsize=(8, 8), dpi=100)
xyz = np.load(f"RealG.npy")
path = 1
tilt = 0
C_ax = -45
pos = np.load(f"Joint_angles_flange/path_{path}_rot_0_tilt_{tilt}_C_{C_ax}.npy")

for iter in range(0, len(pos), 8):
    plt.clf()

    theta = [2, 75, -45, -88, -91, 61 + np.sin(np.radians(iter)) * 30]

    theta = [0 + iter, 135, -45, 0, 0 + np.sin(np.radians(iter)) * 60, 0]
    theta = [0, 135, -45, 0, 0, 0]
    #theta = [0, 0, 0, 0, 0, 0]
    theta = pos[iter]
    theta = np.degrees(theta)
    #print(theta)

    transformations, [x, y, z], rotM = forward_kinematics(a, alpha, d, theta)

    # axCOORD = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    # Extract position and orientation
    positions = np.array([t[:3, 3] for t in transformations])

    # Plot robot links
    for i in range(len(positions) - 1):  # len(positions) - 1
        ax.plot([positions[i][0], positions[i + 1][0]],
                [positions[i][1], positions[i + 1][1]],
                [positions[i][2], positions[i + 1][2]], 'b')

    ax.plot([0, np.cos(np.radians(theta[0])) * a[0]],
            [0, np.sin(np.radians(theta[0])) * a[0]],
            [0, 0], 'g')

    ax.plot([np.cos(np.radians(theta[0])) * a[0], np.cos(np.radians(theta[0])) * a[0]],
            [np.sin(np.radians(theta[0])) * a[0], np.sin(np.radians(theta[0])) * a[0]],
            [0, d[0]], 'g')

    # Plot robot joints
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o', label="Joints")
    #ax.scatter(0, 0, 0, c='r', marker='o')

    #x_coords.append(x)
    #y_coords.append(y)
    #z_coords.append(z)
    #ax.scatter(x_coords, y_coords, z_coords, c="gray", marker='o', s=2, label="Traversed coordinates")


    # ony robot prog
    ax.set_xlim([-400, 600])
    ax.set_ylim([-500, 500])
    ax.set_zlim([0, 1200])

    # path 1
    ax.set_xlim([0, 1000])
    ax.set_ylim([-500, 500])
    ax.set_zlim([0, 1200])

    # For 45
    #ax.set_xlim([0, 1400])
    #ax.set_ylim([-700, 700])
    #ax.set_zlim([0, 1200])


    if path == 4:
        ax.set_xlim([-100, 100])
        ax.set_ylim([-200, 200])
        ax.set_zlim([300, 700])

    if tilt == 46 or tilt == -46:
        ax.set_xlim([0, 1300])
        ax.set_ylim([-650, 650])
        ax.set_zlim([0, 1300])

    pointer1 = np.dot(rotM, [1, 0, 0])
    pointer2 = np.dot(rotM, [0, 1, 0])
    pointer3 = np.dot(rotM, [0, 0, 1])


    """ax.quiver(x, y, z, pointer1[0], pointer1[1], pointer1[2], length=100, normalize=True, color='r',
              linewidth=1)  # x-axis
    ax.quiver(x, y, z, pointer2[0], pointer2[1], pointer2[2], length=100, normalize=True, color='g',
              linewidth=1)  # y-axis
    ax.quiver(x, y, z, pointer3[0], pointer3[1], pointer3[2], length=100, normalize=True, color='b',
              linewidth=1)  # z-axis"""
    fs = 13


    ax.quiver(0, 0, 0, 1, 0, 0, length=300, normalize=True, color='black', linewidth=1)  # x-axis
    ax.quiver(0, 0, 0, 0, 1, 0, length=300, normalize=True, color='black', linewidth=1)  # y-axis
    ax.quiver(0, 0, 0, 0, 0, 1, length=300, normalize=True, color='black', linewidth=1)  # z-axis

    ax.text(310, 0, 0, "X", color='black', fontsize=fs-5)
    ax.text(0, 310, 0, "Y", color='black', fontsize=fs-5)
    ax.text(0, 0, 310, "Z", color='black', fontsize=fs-5)


    # FOR BASIC IMAGE
    ax.elev = 9  # 35  # Set the elevation angle (vertical rotation)
    ax.azim = -50+iter/100  # -45  # Set the azimuth angle (horizontal rotation

    #FOR 45 Top
    #ax.elev = 73  # 35  # Set the elevation angle (vertical rotation)
    #ax.azim = -56  # -45  # Set the azimuth angle (horizontal rotation

    #ax.elev = 7
    #ax.azim = -22


    # FOR real GCODE
    if path == 4:
        ax.elev = 0  # 35  # Set the elevation angle (vertical rotation)
        ax.azim = 0  # -45  # Set the azimuth angle (horizontal rotation

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # print(iter, x, y, z)

    translated_end_point = np.array([-300, 0, 0])
    transformed_end_point = np.dot(rotM, translated_end_point)
    transformed_end_point += [x, y, z]

    ax.plot([x, transformed_end_point[0]], [y, transformed_end_point[1]], [z, transformed_end_point[2]], 'black',
            label='Spindle', linewidth=6)

    translated_end_point_1 = np.array([-50, 0, 0])
    transformed_end_point_1 = np.dot(rotM, translated_end_point_1)
    transformed_end_point_1 += transformed_end_point
    """
    ax.plot([transformed_end_point[0], transformed_end_point_1[0]],
            [transformed_end_point[1], transformed_end_point_1[1]],
            [transformed_end_point[2], transformed_end_point_1[2]], 'orange',
            label='Endmill', linewidth=7)"""

    ax.scatter([transformed_end_point_1[0]],
            [transformed_end_point_1[1]],
            [transformed_end_point_1[2]], c='orange', marker='o', s=150,
            label='Endmill')



    x_coords.append(transformed_end_point_1[0])
    y_coords.append(transformed_end_point_1[1])
    z_coords.append(transformed_end_point_1[2])


    if path ==4:
        # ax.scatter(x_coords, y_coords, z_coords, c=col, marker='o', s=2)

        angel = xyz[6, iter] - angel_before
        #print(angel)
        x_coords, y_coords, z_coords = rotate_x_axis(x_coords, y_coords, z_coords, angel,
                                                     origin_x=800, origin_y=0, origin_z=400)

        angel_before = xyz[6, iter]

    ax.plot(x_coords, y_coords, z_coords, c="gray",  label="Traversed coordinates")#marker='o', s=15,

    ax.quiver(transformed_end_point_1[0], transformed_end_point_1[1], transformed_end_point_1[2], pointer1[0], pointer1[1], pointer1[2], length=300, normalize=True, color='b',
              linewidth=2)  # x-axis
    ax.quiver(transformed_end_point_1[0], transformed_end_point_1[1], transformed_end_point_1[2], -pointer2[0], -pointer2[1], -pointer2[2], length=300, normalize=False, color='g',
              linewidth=2)  # y-axis
    ax.quiver(transformed_end_point_1[0], transformed_end_point_1[1], transformed_end_point_1[2], pointer3[0], pointer3[1], pointer3[2], length=300, normalize=True, color='r',
              linewidth=2)  # z-axis

    ax.text(transformed_end_point_1[0] + 310*np.cos(C_ax), transformed_end_point_1[1]+310*np.sin(C_ax), transformed_end_point_1[2], "X'", color='r', fontsize=fs)
    ax.text(transformed_end_point_1[0]-310*np.sin(C_ax), transformed_end_point_1[1] + 310*np.cos(C_ax), transformed_end_point_1[2], "Y'", color='g', fontsize=fs)
    ax.text(transformed_end_point_1[0], transformed_end_point_1[1], transformed_end_point_1[2] + 410, "Z'", color='b', fontsize=fs) # was 450

    ax.legend(loc='upper right',  bbox_to_anchor=(1, 0.9), fontsize=10, ncol=2) #bbox_to_anchor=(0, 0.08),

    #plt.show()
    #plt.pause(0.01)
    if iter % 100 == 0: print(iter)
    #print(np.linalg.det(rotM))
    plt.savefig(f'robot_rot/{iter}.png', dpi=200, bbox_inches="tight", pad_inches=0.3)


#plt.savefig(f'../Latex/figures/robotprog.png', dpi=1200, bbox_inches="tight", pad_inches=0.3)
#plt.savefig(f'../Latex/figures/robotANDpath1.png', dpi=1200, bbox_inches="tight", pad_inches=0.3)
#plt.savefig(f'../Latex/figures/robotANDpath1_45.png', dpi=1200, bbox_inches="tight", pad_inches=0.3)
#plt.savefig(f'../Latex/figures/robotANDpath3_45.png', dpi=1200, bbox_inches="tight", pad_inches=0.3)
#plt.savefig(f'../Latex/figures/-45.png', dpi=1200, bbox_inches="tight", pad_inches=0.3)
#plt.show()
plt.close()

