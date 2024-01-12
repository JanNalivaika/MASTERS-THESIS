import numpy as np
import matplotlib.pyplot as plt
import math

# DH parameters

d = [600, 0, 0, 800, 0, 200]
a = [200, 900, 150, 0, 0, 0]
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

col = []
fig = plt.figure(figsize=(8, 8), dpi=100)
pos = np.load(f"Joint_angles_flange/path_{3}_rot_0_tilt_{0}_C_{-130}.npy")
#pos = np.load(f"wrong0.npy")
for iter in range(0, len(pos), 10):
    plt.clf()

    theta = [2, 75, -45, -88, -91, 61 + np.sin(np.radians(iter)) * 30]

    theta = [0 + iter, 135, -45, 0, 0 + np.sin(np.radians(iter)) * 60, 0]
    theta = [0, 135, -45, 0, 0, 0]
    theta = pos[iter]
    theta = np.degrees(theta)
    print(theta)

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


    # Set plot limits
    ax.set_xlim([-100, 1500])
    ax.set_ylim([-800, 800])
    ax.set_zlim([0, 1600])

    pointer1 = np.dot(rotM, [1, 0, 0])
    pointer2 = np.dot(rotM, [0, 1, 0])
    pointer3 = np.dot(rotM, [0, 0, 1])

    # JOINTS HERE
    # ax.scatter(0, -600, 0, c='r', marker='o')
    # ax.text(30, -600, -10, "= Joints", color='Black', fontsize=12)

    # Plot robot end-effector
    ax.scatter(x, y, z, c='black', marker='o', s=30, label="TCP")

    ax.quiver(x, y, z, pointer1[0], pointer1[1], pointer1[2], length=100, normalize=True, color='r',
              linewidth=1)  # x-axis
    ax.quiver(x, y, z, pointer2[0], pointer2[1], pointer2[2], length=100, normalize=True, color='g',
              linewidth=1)  # y-axis
    ax.quiver(x, y, z, pointer3[0], pointer3[1], pointer3[2], length=100, normalize=True, color='b',
              linewidth=1)  # z-axis
    fs = 15
    #ax.text( x+310, y, z, "X'", color='r', fontsize = fs)
    #ax.text( x, y+310, z, "Y'", color='g', fontsize = fs)
    #ax.text( x, y, z+310, "Z'", color='b', fontsize = fs)

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

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # print(iter, x, y, z)

    translated_end_point = np.array([-300, 0, 0])
    transformed_end_point = np.dot(rotM, translated_end_point)
    transformed_end_point += [x, y, z]

    ax.plot([x, transformed_end_point[0]], [y, transformed_end_point[1]], [z, transformed_end_point[2]], 'black',
            label='Spindle', linewidth=10)

    translated_end_point_1 = np.array([-50, 0, 0])
    transformed_end_point_1 = np.dot(rotM, translated_end_point_1)
    transformed_end_point_1 += transformed_end_point

    ax.plot([transformed_end_point[0], transformed_end_point_1[0]],
            [transformed_end_point[1], transformed_end_point_1[1]],
            [transformed_end_point[2], transformed_end_point_1[2]], 'red',
            label='Endmill', linewidth=5)



    x_coords.append(transformed_end_point_1[0])
    y_coords.append(transformed_end_point_1[1])
    z_coords.append(transformed_end_point_1[2])
    ax.scatter(x_coords, y_coords, z_coords, c="gray", marker='o', s=2, label="Traversed coordinates")

    ax.quiver(transformed_end_point_1[0], transformed_end_point_1[1], transformed_end_point_1[2], pointer1[0], pointer1[1], pointer1[2], length=300, normalize=True, color='b',
              linewidth=2)  # x-axis
    ax.quiver(transformed_end_point_1[0], transformed_end_point_1[1], transformed_end_point_1[2], -pointer2[0], -pointer2[1], -pointer2[2], length=300, normalize=True, color='g',
              linewidth=2)  # y-axis
    ax.quiver(transformed_end_point_1[0], transformed_end_point_1[1], transformed_end_point_1[2], pointer3[0], pointer3[1], pointer3[2], length=300, normalize=True, color='r',
              linewidth=2)  # z-axis


    ax.legend(loc='center left', bbox_to_anchor=(0, 0.08), fontsize=10)

    #plt.show()
    plt.pause(0.01)
    if iter % 100 == 0: print(iter)
    #print(np.linalg.det(rotM))

plt.show()
