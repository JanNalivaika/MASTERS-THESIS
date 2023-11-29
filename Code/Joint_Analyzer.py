import numpy as np
import matplotlib.pyplot as plt

# Load joint positions from .npy file
joint_positions1 = np.degrees(np.load('Joint_angles/path_1_rot_0_tilt_0_C_0.0.npy')[:,4])
for x in joint_positions1:
    print(x)
#joint_positions2 = np.load('Joint_angles/path_1_rot_-25_tilt_-20_C_-0.6.npy')[:,3]
# Calculate velocities
joint_velocities = np.gradient(joint_positions1)

# Calculate accelerations
joint_accelerations = np.gradient(joint_velocities)

# Calculate jerks
joint_jerks = np.gradient(joint_accelerations)

# Create time array
time = np.arange(len(joint_positions1))

# Plot joint positions
#plt.subplot(4, 1, 1)
plt.scatter(time, joint_positions1)
#plt.scatter(time, joint_positions2)
plt.title('Joint Positions')
plt.xlabel('Time')
plt.ylabel('Position')

# Plot joint velocities
#plt.subplot(4, 1, 2)
#plt.plot(time, joint_velocities)
#plt.title('Joint Velocities')
#plt.xlabel('Time')
#plt.ylabel('Velocity')

# Plot joint accelerations
#plt.subplot(4, 1, 3)
#plt.plot(time, joint_accelerations)
#plt.title('Joint Accelerations')
#plt.xlabel('Time')
#plt.ylabel('Acceleration')

# Plot joint jerks
#plt.subplot(4, 1, 4)
#plt.plot(time, joint_jerks)
#plt.title('Joint Jerks')
#plt.xlabel('Time')
#plt.ylabel('Jerk')

plt.tight_layout()
plt.show()
