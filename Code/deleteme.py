import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def helix_function(t, r, a, k):
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = a * t
    return x, y, z


t = np.linspace(0, 10 * np.pi, 1000)  # Input values

r = 1.0  # Radius of the helix
a = 0.5  # Vertical spacing between each revolution
k = 5.0  # Controls the frequency of the helix twists

x, y, z = helix_function(t, r, a, k)

# Plotting the 3D helix
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Helix Function')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
