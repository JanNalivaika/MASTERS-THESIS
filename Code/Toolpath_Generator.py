import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate data
np.random.seed(0)
iter = np.arange(2500)

selection = 3
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')

if selection == 1:
    x = np.cos(np.deg2rad(iter)) * (500 - iter / 3)
    y = np.sin(np.deg2rad(iter)) * (500 - iter / 3)
    z = iter / 10
    tit = "Converging-Diverging Spiral"

if selection == 2:
    x = np.sin(np.deg2rad(iter)) * (800-iter / 4)
    y = np.sin(np.deg2rad(iter)) * np.cos(np.deg2rad(iter)) * (800-iter / 4)
    z = iter / 10
    tit = "Converging Loop"

if selection == 3:
    x = np.sin(np.deg2rad(iter)) * 200
    y = iter / 3 - (2500/3/2)
    z = np.sin( np.deg2rad(x))*50
    tit = "Pendulum Wave"


# Create a 3D scatter plot

ax.scatter(x, y, z, c=z, cmap='viridis')





# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set plot title
ax.set_title('3D Scatter Plot')
ax.set_xlim([-600, 600])
ax.set_ylim([-600, 600])
ax.set_zlim([-100, 300])
ax.view_init(elev=20, azim=-150)
plt.title(tit,fontsize=30)
plt.savefig(f'../Latex/figures/path{selection}.png', dpi=500,bbox_inches='tight')
# Display the plot
#plt.show()