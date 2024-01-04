import numpy as np
import matplotlib.pyplot as plt
import math

# DH parameters

d = [600, 0, 0, 800, 0, 200]
a = [200, 900, 150, 0, 150, 0]
alpha = [90, 0, 90, -90, 90, -90]

#import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 20})

# Homogeneous transformation matrix
def dh_transform(a, alpha, d, theta):
    transform = np.array(
        [[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
         [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
         [0, np.sin(alpha), np.cos(alpha), d],
         [0, 0, 0, 1]])
    return transform


def jacobian(a, alpha, d, theta):
    # Calculate the forward kinematics to get the transformation matrix T
    transformations, _, _ = forward_kinematics(a, alpha, d, theta)
    T = transformations[-1]  # Use the final transformation matrix

    # Calculate the Jacobian matrix
    J = np.zeros((6, len(theta)))  # Assuming a 6-DOF manipulator

    for i in range(len(theta)):
        # Calculate the partial derivatives of the position and orientation
        # with respect to the joint variables
        T_i = transformations[i]
        z_i = T_i[:3, 2]  # Z-axis of the i-th link
        p_i = T_i[:3, 3]  # Position of the i-th link

        J[:3, i] = np.cross(z_i, T[:3, 3] - p_i)
        J[3:, i] = z_i

    return J


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


def rotate_x_axis(x, y, z, angle, origin_x=0, origin_y=0, origin_z=0):
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


x_coords = []
y_coords = []
z_coords = []
EV = []
col = []
fig = plt.figure(figsize=(8, 8), dpi=100)
pos = np.load("Joint_angles/path_3_rot_0_tilt_45_C_0.npy")
#pos = np.load("RealG_angles/path_4_rot_0_tilt_conti._C_-30.npy")

#xyz = np.load(f"RealG.npy")
angel_before = 0
for iter in range(0, len(pos), 1):
    plt.clf()
    theta = pos[iter]
    theta = np.degrees(theta)
    # print(theta)
    # theta =     [0, 80, -27, 180, -128, 0]

    # print(theta)
    # theta = [0, 135, -45, 0, 0, 0]
    # theta = [0, 0, -0, 0, 0, 0]
    # theta = [2, 75, -45, -88, -91, 61]
    # explain =  [D,  K,   K, D,  K, D]

    transformations, [x, y, z], rotM = forward_kinematics(a, alpha, d, theta)

    det_jacob = np.linalg.det(jacobian(a, alpha, d, theta))
    # print(det_jacob)
    if det_jacob == 0:
        col.append("red")
    else:
        col.append("silver")
    EV.append(det_jacob)

    # axCOORD = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    # Extract position and orientation
    positions = np.array([t[:3, 3] for t in transformations])

    # plt.show()

    orientations = np.array([t[:3, :3] for t in transformations])

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
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o')
    ax.scatter(0, 0, 0, c='r', marker='o')

    x_coords.append(x)
    y_coords.append(y)
    z_coords.append(z)
    """# ax.scatter(x_coords, y_coords, z_coords, c=col, marker='o', s=2)

    angel = xyz[6, iter] - angel_before
    # print(angel)
    x_coords, y_coords, z_coords = rotate_x_axis(x_coords, y_coords, z_coords, angel,
                                                 origin_x=1000, origin_y=0, origin_z=600)

    angel_before = xyz[6, iter]"""
    ax.scatter(x_coords, y_coords, z_coords, c=col, marker='o', s=2)
    # ax.scatter(x_coords_tilt, y_coords_tilt, z_coords_tilt, c=col, marker='o', s=2)

    # Set plot limits
    ax.set_xlim([-100, 1500])
    ax.set_ylim([-800, 800])
    ax.set_zlim([0, 1600])

    pointer1 = np.dot(rotM, [1, 0, 0])
    pointer2 = np.dot(rotM, [0, 1, 0])
    pointer3 = np.dot(rotM, [0, 0, 1])

    # JOINTS HERE
    ax.scatter(0, -600, 0, c='r', marker='o')
    ax.text(30, -600, -10, "= Joints", color='Black', fontsize=12)


    # Plot robot end-effector
    ax.scatter(x, y, z, c='black', marker='o', s=20)

    ax.quiver(x, y, z, pointer1[0], pointer1[1], pointer1[2], length=300, normalize=True, color='r',
              linewidth=3)  # x-axis
    ax.quiver(x, y, z, pointer2[0], pointer2[1], pointer2[2], length=300, normalize=True, color='g',
              linewidth=3)  # y-axis
    ax.quiver(x, y, z, pointer3[0], pointer3[1], pointer3[2], length=300, normalize=True, color='b',
              linewidth=3)  # z-axis
    fs = 15
    ax.text( x+310, y, z, "X'", color='r', fontsize = fs)
    ax.text( x, y+310, z, "Y'", color='g', fontsize = fs)
    ax.text( x, y, z+310, "Z'", color='b', fontsize = fs)

    ax.quiver(0, 0, 0, 1, 0, 0, length=300, normalize=True, color='black', linewidth=1)  # x-axis
    ax.quiver(0, 0, 0, 0, 1, 0, length=300, normalize=True, color='black', linewidth=1)  # y-axis
    ax.quiver(0, 0, 0, 0, 0, 1, length=300, normalize=True, color='black', linewidth=1)  # z-axis


    ax.text(310, 0, 0, "X", color='black')
    ax.text(0, 310, 0, "Y", color='black')
    ax.text(0, 0, 310, "Z", color='black')

    # ax.view_init(elev=30, azim=45)
    # FOR BASIC IMAGE
    ax.elev = 9  # 35  # Set the elevation angle (vertical rotation)
    ax.azim = -50  # -45  # Set the azimuth angle (horizontal rotation

    #robotANDpath1_
    #ax.elev = 70  # 35  # Set the elevation angle (vertical rotation)
    #ax.azim = -50  # -45  # Set the azimuth angle (horizontal rotation


    """ax.elev = 0
    ax.azim = 0
    ax.set_ylim([-300, 300])
    ax.set_zlim([500, 1000])
    """

    # Set plot labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # print(iter, x, y, z)

    # Show plot
    # if iter%100==0: plt.savefig(f"robot/robot{iter}.png", dpi=200, bbox_inches="tight", pad_inches=0.3)
    #plt.pause(0.01)
    # print(theta[3]+theta[5])
    # print(pointer1)
    # plt.show()
    # plt.close()

    if iter % 100 == 0: print(iter)

#plt.setp(ax.get_xticklabels(), visible=False)
#plt.setp(ax.get_yticklabels(), visible=False)
#plt.setp(ax.get_zticklabels(), visible=False)
# plt.savefig(f"../Latex/figures/robotANDpath3_45.png", dpi=500, bbox_inches="tight", pad_inches=0.1)
# plt.savefig(f"../Latex/figures/robotprog.png", dpi=500, bbox_inches="tight", pad_inches=0.1)
# plt.savefig(f"../Latex/figures/robotANDpath1.png", dpi=500, bbox_inches="tight", pad_inches=0.1)
#plt.savefig(f"../Latex/figures/robotANDpath1_45.png", dpi=500, bbox_inches="tight", pad_inches=0.1)
plt.savefig(f"../Latex/figures/robotANDpath3_45.png", dpi=500, bbox_inches="tight", pad_inches=0.1)

plt.show()
# plt.plot(EV)
plt.show()
