import mpmath
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import glob

import pylab as pl
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()


def simplify_angle(angle):
    angle = np.round(angle,2)
    while angle > 180:
        angle -= 2 * 180
    while angle < -180:
        angle += 2 * 180
    return angle



def basicplot():
    fig = plt.figure(figsize=(10, 6), dpi=200)
    for joint in range(6):

        joint_positions = np.degrees(np.load('Joint_angles/path_1_rot_0_tilt_0_C_0.npy')[:, joint])

        for i in range(len(joint_positions)):
            joint_positions[i] = simplify_angle(joint_positions[i])
        time = np.arange(len(joint_positions)*0.1,step = 0.1)
        plt.plot(time, np.round(joint_positions, 2), label=f"Joint {joint+1}", lw=3)


    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title(f'Toolpath 1, A=B=C=0')
    plt.xlabel('Time [s]')
    plt.ylabel('Position in Degrees [°] ')

    plt.tight_layout()
    plt.savefig(f"../Latex/figures/TP1ABC0.png")
    #plt.show()
    plt.close()


def count_direction_changes(time_series):
    direction_changes = 0
    remeber = 1
    for i in range(1, len(time_series)):
        if time_series[i-1] < time_series[i] and remeber == -1:
            direction_changes += 1
            remeber = 1
        if time_series[i-1] > time_series[i] and remeber == 1:
            direction_changes += 1
            remeber = -1

    return direction_changes




def basicscore():
    for path in [1,2,3]:
        plt.figure(figsize=(10, 4), dpi=200)
        files = glob.glob(f'Joint_angles/*path_{path}_rot_0_tilt_0*')
        DC_tracker = []
        Travel_tracker = []
        Acc_tracker = []
        X_ax = []
        for idx, file in enumerate(files):
            C_val = np.float64(file.split("_")[-1].split(".npy")[0])
            tilt_val =  np.float64(file.split("_")[6])
            X_ax.append(C_val)
            DC_total = 0
            total_travel = 0
            accel_sore = 0
            for joint in range(6):

                joint_positions = np.degrees(np.load(file)[:, joint])
                for i in range(len(joint_positions)):
                    joint_positions[i] = simplify_angle(joint_positions[i])
                DC =count_direction_changes(joint_positions)
                DC_total+=DC

                for x in range(1,len(joint_positions)):
                    total_travel += abs(joint_positions[i-1]-joint_positions[i])


                if joint == 0:
                    joint_velocity = np.gradient(joint_positions, np.arange(len(joint_positions)*0.1,step = 0.1))
                    joint_acceleration = np.gradient(joint_velocity, np.arange(len(joint_positions) * 0.1, step=0.1))

                    accel_sore += np.sum(np.square(joint_acceleration))

            DC_tracker.append(DC_total)
            Travel_tracker.append(total_travel)
            Acc_tracker.append(accel_sore)
            # print(DC_total,total_travel,accel_sore)

        scaled_DC_tracker = min_max_scaler.fit_transform(-np.array(DC_tracker).reshape(-1, 1))*100*0.4
        scaled_Travel_tracker = min_max_scaler.fit_transform(-np.array(Travel_tracker).reshape(-1, 1))*100*0.4
        scaled_Acc_tracker = min_max_scaler.fit_transform(-np.array(Acc_tracker).reshape(-1, 1))*100*0.2

        X_ax, scaled_DC_tracker, scaled_Travel_tracker, scaled_Acc_tracker = zip(*sorted(zip(X_ax, scaled_DC_tracker, scaled_Travel_tracker, scaled_Acc_tracker)))

        plt.plot(X_ax,scaled_DC_tracker, lw = 0.5, label="Direction Canges",linestyle='dashed', marker='o')
        plt.plot(X_ax,scaled_Travel_tracker, lw = 0.5, label="Travel",linestyle='dashed', marker='o')
        plt.plot(X_ax,scaled_Acc_tracker, lw = 0.5,  label="Acceleration",linestyle='dashed', marker='o')

        plt.xlabel('Rotation around Z in degrees')
        plt.ylabel('Local Score')
        plt.ylim((-10,110))
        #plt.title(f"Toolpath {path}", y=1.2)
        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3)
        plt.savefig(f"../Latex/figures/LocalScores_{path}.png", bbox_inches='tight')
        #plt.close()
        #plt.figure(figsize=(10, 4))

        SCORE = np.array(scaled_DC_tracker)+ np.array(scaled_Travel_tracker) + np.array(scaled_Acc_tracker)

        max_value = np.max(SCORE)
        max_index = X_ax[int(np.where(SCORE == max_value)[0])]

        print(max_value, max_index)


        plt.plot(X_ax,SCORE, lw = 0.5, c="red", label = "Global score",linestyle='dashed', marker='o')
        plt.scatter(max_index, max_value, s = 200, c="green", label="optimal Bandary condition", marker = "2")
        plt.ylim((-10, 110))
        plt.xlabel('Rotation around Z in degrees')
        plt.ylabel('Score')
        #plt.title(f"Toolpath {path}", y=1.2)
        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=2)
        #plt.savefig(f"../Latex/figures/best_c_{path}.png",bbox_inches='tight')
        plt.savefig(f"../Latex/figures/best_c_{path}_combi.png", bbox_inches='tight')
        #plt.show()
        plt.close()



def TWODplot():

    DC_tracker234 = []
    DC_tracker1 = []
    V_tracker = []
    Acc_tracker = []

    toolpath = 3
    matrix = np.zeros((51,51))
    tracker = 0
    for kipp_winkel in range(-25, 26, 1):  # -25, 26 ,10
        for c_axis in range(-25, 26, 1):
            try:
                if kipp_winkel==-15 and c_axis==2:
                    print("panic")
                DC234 = 0
                DC1 = 0
                accel_sore = 0
                v_score = 0

                joints = np.load(f"Joint_angles/path_{toolpath}_rot_0_tilt_{kipp_winkel}_C_{c_axis}.npy")

                for j in range(6):
                    joint = joints[:,j]
                    joint = np.degrees(joint)


                    if j==0:
                        DC1 = count_direction_changes(joint)
                        joint_velocity = np.gradient(joint, np.arange(len(joint) * 0.1, step=0.1))
                        joint_acceleration = np.gradient(joint_velocity,np.arange(len(joint) * 0.1, step=0.1))

                        accel_sore = np.sum(np.square(joint_acceleration))

                    if j == 1 or j==2 or j==4:
                        DC234+=count_direction_changes(joint)

                    if j==5:

                        joint_velocity = np.gradient(joint, np.arange(len(joint) * 0.1, step=0.1))
                        v_score = np.sum(np.square(joint_velocity))

                DC_tracker234.append(DC234)
                DC_tracker1.append(DC1)
                Acc_tracker.append(accel_sore)
                V_tracker.append(v_score)

            except:
                #print("asdasdasasdasdasdasdasdasdafdafggfdhjjjsdfjghsfd")

                DC_tracker234.append(np.average(DC_tracker234))
                DC_tracker1.append(np.average(DC_tracker1))
                Acc_tracker.append(np.average(Acc_tracker))
                V_tracker.append(np.average(V_tracker))

    DC_tracker234 = min_max_scaler.fit_transform(-np.array(DC_tracker234).reshape(-1, 1)) * 100 * 0.3
    DC_tracker1 = min_max_scaler.fit_transform(-np.array(DC_tracker1).reshape(-1, 1)) * 100 * 0.25
    Acc_tracker = min_max_scaler.fit_transform(-np.array(Acc_tracker).reshape(-1, 1)) * 100 * 0.25
    V_tracker = min_max_scaler.fit_transform(-np.array(V_tracker).reshape(-1, 1)) * 100 * 0.2


    score = np.array(DC_tracker234) + np.array(DC_tracker1)+ np.array(V_tracker)+ np.array(Acc_tracker)


    matrix = np.reshape(score, (-1, 51))


    #plt.rcParams["figure.figsize"] = [7.00, 3.50]
    #plt.rcParams["figure.autolayout"] = True
    plt.figure(figsize=(9, 9))
    plt.imshow(matrix)
    # Set xticks and yticks
    plt.xticks(range(0,51,2), range(-25, 26 ,2))
    plt.xlabel("C in Degrees [°] ")
    plt.ylabel("Tilting in Degrees [°] ")
    plt.yticks(range(0,51,2), range(-25, 26 ,2))
    plt.colorbar()
    plt.grid(color='black', linewidth=0.1)
    plt.savefig(f"../Latex/figures/best_2D_{toolpath}.png", bbox_inches='tight',dpi=1000)
    #plt.show()
    plt.close()

#basicplot()
#basicscore()
TWODplot()