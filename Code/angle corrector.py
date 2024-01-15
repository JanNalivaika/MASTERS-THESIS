import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib import ticker



def simplify_angle(angles):
    angles = np.array(angles)
    #angle = np.round(angle,2)
    while all(i > 180 for i in angles):
        angles -= 2 * 180
    while all(i < -180 for i in angles):
        angles += 2* 180
    return angles

tp = 1
T = 1
C = 0
joint_positions_all = np.degrees(np.load(f'Joint_angles_lowres_flange/path_{tp}_rot_0_tilt_{T}_C_{C}.npy'))
for joint in range(6):
    print(joint)

    joint_positions = simplify_angle(joint_positions_all[:,joint])
    plt.plot(np.round(joint_positions, 2), label=f"Joint {joint+1}", lw=3)


    plt.show()

print("FINISH")
#np.save(f'Joint_angles_lowres_flange/path_{tp}_rot_0_tilt_{T}_C_{C}.npy',joint_positions_all)
