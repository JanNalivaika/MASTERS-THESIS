import numpy as np
import matplotlib.pyplot as plt

# DH parameters
a = [180, 600, 120, 0, 0, 0]
alpha = [90, 0, 0, 90, 90, 0]
d = [400, 0, 0, 620, 0, 115]


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
    return transformations, position,rmatrix


pos = np.load("0_angles.npy")

fig = plt.figure(figsize=(8, 8), dpi=100)


for iter in range(360):
    plt.clf()

    theta = [iter, 45, -90, 0, 90, 0]
    theta = [iter, iter+45, iter-90, iter, iter+180, iter-45]
    #theta = pos[iter]
    #theta = np.deg2rad(theta)
    # Compute transformations
    transformations, [x, y, z], rotM = forward_kinematics(a, alpha, d, theta)

    # axCOORD = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    # Extract position and orientation
    positions = np.array([t[:3, 3] for t in transformations])

    # plt.show()

    orientations = np.array([t[:3, :3] for t in transformations])

    # Plot robot links
    for i in range(len(positions) - 1):
        ax.plot([positions[i][0], positions[i + 1][0]],
                [positions[i][1], positions[i + 1][1]],
                [positions[i][2], positions[i + 1][2]], 'b')

    ax.plot([0, np.cos(np.radians(theta[0]))*a[0]],
            [0, np.sin(np.radians(theta[0]))*a[0]],
            [0, 0], 'g')

    ax.plot([np.cos(np.radians(theta[0])) * a[0], np.cos(np.radians(theta[0])) * a[0]],
            [np.sin(np.radians(theta[0])) * a[0], np.sin(np.radians(theta[0])) * a[0]],
            [0, d[0]], 'g')





    # Plot robot joints
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o')
    ax.scatter(0, 0, 0, c='r', marker='o')

    # Plot robot end-effector
    ax.scatter(positions[-1][0], positions[-1][1], positions[-1][2], c='g', marker='o')

    # Set plot limits
    ax.set_xlim([-1000, 1000])
    ax.set_ylim([-1000, 1000])
    ax.set_zlim([-1000, 1000])

    pointer1 = np.dot(rotM,[1, 0, 0])
    pointer2 = np.dot(rotM, [0, 1, 0])
    pointer3 = np.dot(rotM, [0, 0, 1])


    ax.quiver(x, y, z, pointer1[0], pointer1[1], pointer1[2], length=300, normalize=True, color='r', linewidth=3)  # x-axis
    ax.quiver(x, y, z, pointer2[0], pointer2[1], pointer2[2], length=300, normalize=True, color='g', linewidth=3)  # y-axis
    ax.quiver(x, y, z, pointer3[0], pointer3[1], pointer3[2], length=300, normalize=True, color='b', linewidth=3)  # z-axis
    # ax.view_init(elev=30, azim=45)
    # ax.elev = 30  # Set the elevation angle (vertical rotation)
    # ax.azim = 45  # Set the azimuth angle (horizontal rotation
    # Set plot labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #print(iter, x, y, z)

    # Show plot

    plt.pause(0.001)
    #print(pointer1)
    #plt.show()

plt.show()