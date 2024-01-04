import mpmath
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib import ticker

import pylab as pl
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()


def simplify_angle(angles):
    angles = np.array(angles)
    #angle = np.round(angle,2)
    while all(i > 180 for i in angles):
        angles -= 2 * 180
    while all(i < -180 for i in angles):
        angles += 2* 180
    return angles



def basicplot():

    for tp in [1,2,3]:
        for C in [0,45]:
            import matplotlib.pyplot as plt
            plt.rcParams.update({'font.size': 12})

            plt.figure(figsize=(10, 4), dpi=200)
            for joint in range(6):
                joint_positions = np.degrees(np.load(f'Joint_angles/path_{tp}_rot_0_tilt_0_C_{C}.npy')[:, joint])

                for i in range(6):
                    joint_positions = simplify_angle(joint_positions)

                time = np.arange(len(joint_positions)*0.1,step = 0.1)
                plt.plot(time, np.round(joint_positions, 2), label=f"Joint {joint+1}", lw=3)


            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.title(f'Toolpath {tp}, A=0° B=0° C={C}°')
            plt.xlabel('Time in seconds [s]')
            plt.ylabel('Position in degrees [°] ')
            plt.ylim((-180,180))
            plt.tight_layout()
            plt.savefig(f"../Latex/figures/TP{tp}ABC{C}.png",dpi=1200)
            print(f"TP{tp}ABC{C}.png")
            #plt.show()
            plt.close()


def count_direction_changes(time_series):
    direction_changes = 0
    remeber = 1
    time_series = np.round(time_series,3)
    for i in range(1, len(time_series)):
        if time_series[i-1] < time_series[i] and remeber == -1 and abs(time_series[i-1] - time_series[i]) > 0.01:
            direction_changes += 1
            remeber = 1
        if time_series[i-1] > time_series[i] and remeber == 1 and abs(time_series[i-1] - time_series[i]) > 0.01:
            direction_changes += 1
            remeber = -1

    return direction_changes




def basicscore():
    for path in [1,2,3]:

        files = glob.glob(f'Joint_angles/*path_{path}_rot_0_tilt_0*')

        DC_tracker = []
        Travel_tracker = []
        Acc_tracker = []
        X_ax = []
        for idx, file in enumerate(files):
            C_val = np.float64(file.split("_")[-1].split(".npy")[0])

            X_ax.append(C_val)
            DC_total = 0
            total_travel = 0
            accel_sore = 0
            for joint in range(6):

                joint_positions = np.degrees(np.load(file)[:, joint])
                joint_positions = simplify_angle(joint_positions)


                DC =count_direction_changes(joint_positions)
                DC_total += DC

                for x in range(1,len(joint_positions)):
                    total_travel += abs(joint_positions[x-1]-joint_positions[x])


                if joint == 0:
                    joint_velocity = np.gradient(joint_positions, np.arange(len(joint_positions)*0.1,step = 0.1))
                    joint_acceleration = np.gradient(joint_velocity, np.arange(len(joint_positions) * 0.1, step=0.1))

                    accel_sore += np.sum(np.square(joint_acceleration))


            DC_tracker.append(DC_total)
            Travel_tracker.append(total_travel)
            Acc_tracker.append(accel_sore)

        plt.figure(figsize=(10, 4), dpi=200)

        scaled_DC_tracker = min_max_scaler.fit_transform(-np.array(DC_tracker).reshape(-1, 1))*100*0.2
        scaled_Travel_tracker = min_max_scaler.fit_transform(-np.array(Travel_tracker).reshape(-1, 1))*100*0.4
        scaled_Acc_tracker = min_max_scaler.fit_transform(-np.array(Acc_tracker).reshape(-1, 1))*100*0.4

        X_ax, scaled_DC_tracker, scaled_Travel_tracker, scaled_Acc_tracker = zip(*sorted(zip(X_ax, scaled_DC_tracker, scaled_Travel_tracker, scaled_Acc_tracker)))

        plt.plot(X_ax,scaled_DC_tracker, lw = 0.5, label="Direction changes in joints 1-6",linestyle='dashed', marker='o')
        plt.plot(X_ax,scaled_Travel_tracker, lw = 0.5, label="Total travel in joint 1-6",linestyle='dashed', marker='o')
        plt.plot(X_ax,scaled_Acc_tracker, lw = 0.5,  label="Acceleration in joint 1",linestyle='dashed', marker='o')

        plt.xlabel('Rotation around Z in degrees [°]')
        plt.ylabel('Local Score')
        plt.ylim((-10,110))
        plt.xlim((-145, 145))

        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3)
        plt.savefig(f"../Latex/figures/LocalScores_{path}.png", bbox_inches='tight',dpi=1200)
        plt.close()
        plt.figure(figsize=(10, 4))

        SCORE = np.array(scaled_DC_tracker)+ np.array(scaled_Travel_tracker) + np.array(scaled_Acc_tracker)

        max_value = np.max(SCORE)
        max_index = X_ax[int(np.where(SCORE == max_value)[0])]

        print(max_value, max_index)


        plt.plot(X_ax,SCORE, lw = 0.5, c="red", label = "Global score",linestyle='dashed', marker='o')
        plt.scatter(max_index, max_value, s = 250, c="green", label="Optimal boundary condition", marker = "2")

        plt.xlabel('Rotation around Z in degrees [°]')
        plt.ylabel('Score')

        plt.ylim((0, 105))
        plt.xlim((-145, 145))
        plt.vlines(max_index, -0, max_value, linestyle="dashed")
        plt.hlines(max_value, -145, max_index, linestyle="dashed")
        plt.text(max_index, max_value+5, f"Global score = {np.round(max_value,1)}, Optimal rotation = {int(max_index)}°", color='black')

        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=2)
        plt.savefig(f"../Latex/figures/best_c_{path}.png",bbox_inches='tight',dpi=1200)
        #plt.savefig(f"../Latex/figures/best_c_{path}_combi.png", bbox_inches='tight',dpi=1000)
        #plt.show()
        plt.close()


def RealG():
    files = glob.glob(f'RealG_angles/*')

    DC_tracker1 = []
    DC_tracker2 = []
    DC_tracker3 = []
    V_tracker4 = []
    T_tracker6 = []

    X_ax = []

    for idx, file in enumerate(files):
        C_val = np.float64(file.split("_")[-1].split(".npy")[0])

        X_ax.append(C_val)

        DC_1 = 0
        DC_2 = 0
        DC_3 = 0
        V_4 = 0
        T_6 = 0
        for joint in range(6):

            joint_positions = np.degrees(np.load(file)[:, joint])
            joint_positions = simplify_angle(joint_positions)

            if joint == 0:
                DC_1 = count_direction_changes(joint_positions)

            if joint == 1:
                DC_2 = count_direction_changes(joint_positions)

            if joint == 2:
                DC_3 = count_direction_changes(joint_positions)


            if joint == 3:
                joint_velocity = np.gradient(joint_positions, np.arange(len(joint_positions)*0.1,step = 0.1))
                V_4 = np.sum(np.square(joint_velocity))

            if joint == 5:
                for x in range(1, len(joint_positions)):
                    T_6 += abs(joint_positions[x - 1] - joint_positions[x])

        DC_tracker1.append(DC_1)
        DC_tracker2.append(DC_2)
        DC_tracker3.append(DC_3)
        V_tracker4.append(V_4)
        T_tracker6.append(T_6)

    plt.figure(figsize=(10, 4), dpi=200)

    scaled_DC_tracker1 = min_max_scaler.fit_transform(-np.array(DC_tracker1).reshape(-1, 1))*100*0.2
    scaled_DC_tracker2 = min_max_scaler.fit_transform(-np.array(DC_tracker2).reshape(-1, 1))*100*0.2
    scaled_DC_tracker3 = min_max_scaler.fit_transform(-np.array(DC_tracker3).reshape(-1, 1))*100*0.2
    scaled_V_tracker4 = min_max_scaler.fit_transform(-np.array(V_tracker4).reshape(-1, 1)) * 100 * 0.2
    scaled_T_tracker6 = min_max_scaler.fit_transform(-np.array(T_tracker6).reshape(-1, 1)) * 100 * 0.2


    X_ax, scaled_DC_tracker1, scaled_DC_tracker2, scaled_DC_tracker3,scaled_V_tracker4,scaled_T_tracker6 = zip(*sorted(zip(X_ax, scaled_DC_tracker1, scaled_DC_tracker2, scaled_DC_tracker3,scaled_V_tracker4,scaled_T_tracker6)))

    plt.plot(X_ax,scaled_DC_tracker1, lw = 0.5, label="Direction changes in joints 1",linestyle='dashed', marker='o')
    plt.plot(X_ax,scaled_DC_tracker2, lw = 0.5, label="Direction changes in joints 2",linestyle='dashed', marker='o')
    plt.plot(X_ax,scaled_DC_tracker3, lw = 0.5,  label="Direction changes in joints 3",linestyle='dashed', marker='o')
    plt.plot(X_ax, scaled_V_tracker4, lw=0.5, label="Velocity in joint 5", linestyle='dashed', marker='o')
    plt.plot(X_ax, scaled_T_tracker6, lw=0.5, label="Total travel in joint 6", linestyle='dashed', marker='o')

    plt.xlabel('Rotation around Z in degrees [°]')
    plt.ylabel('Local Score')
    plt.ylim((-3,25))

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3)
    plt.savefig(f"../Latex/figures/LocalScores_{4}.png", bbox_inches='tight',dpi=1200)
    plt.close()
    plt.figure(figsize=(10, 4))

    SCORE = np.array(scaled_DC_tracker1)+ np.array(scaled_DC_tracker2) + np.array(scaled_DC_tracker3) + np.array(scaled_V_tracker4) + np.array(scaled_T_tracker6)

    max_value = np.max(SCORE)
    max_index = X_ax[int(np.where(SCORE == max_value)[0])]

    print(max_value, max_index)


    plt.plot(X_ax,SCORE, lw = 0.5, c="red", label = "Global score",linestyle='dashed', marker='o')
    plt.scatter(max_index, max_value, s = 200, c="green", label="Optimal boundary condition", marker = "2")
    plt.ylim((0, 105))
    plt.xlim((-145, 145))
    plt.vlines(max_index, -0, max_value, linestyle="dashed")
    plt.hlines(max_value, -145, max_index, linestyle="dashed")
    plt.text(max_index, max_value + 5, f"Global score = {np.round(max_value, 1)}, Optimal rotation = {int(max_index)}°",
             color='black')
    plt.xlabel('Rotation around Z in degrees [°]')
    plt.ylabel('Score')

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=2)
    plt.savefig(f"../Latex/figures/best_c_{4}.png",bbox_inches='tight',dpi=1200)
    #plt.savefig(f"../Latex/figures/best_c_{4}_combi.png", bbox_inches='tight',dpi=1000)
    #plt.show()
    plt.close()


def TWODplot():


    for toolpath in [1,2,3]:

        a = range(-45, 46, 2)
        b = range(-135, 140, 5)
        DC_tracker234 = []
        DC_tracker1 = []
        V_tracker = []
        Acc_tracker = []


        for kipp_winkel in range(-45, 46, 2):  # -25, 26 ,10
            for c_axis in range(-135, 140, 5):
                #print(kipp_winkel,c_axis)
                try:
                    #if kipp_winkel==-15 and c_axis==2:
                    #    print("panic")
                    DC234 = 0
                    DC1 = 0
                    accel_sore = 0
                    v_score = 0

                    joints = np.load(f"Joint_angles_lowres/path_{toolpath}_rot_0_tilt_{kipp_winkel}_C_{c_axis}.npy")

                    for j in range(6):
                        joint = joints[:,j]
                        joint = np.degrees(joint)

                        joint = simplify_angle(joint)


                        if j==0:
                            DC1 = count_direction_changes(joint)
                            #joint_velocity = np.gradient(joint, np.arange(len(joint) * 0.1, step=0.1))
                            #joint_acceleration = np.gradient(joint_velocity,np.arange(len(joint) * 0.1, step=0.1))

                            #accel_sore = np.sum(np.square(joint_acceleration))

                        if j == 1 or j==2 or j==4:
                            DC234+=count_direction_changes(joint)

                        if j==5:

                            joint_velocity = np.gradient(joint, np.arange(len(joint) * 0.1, step=0.1))
                            v_score = np.sum(np.square(joint_velocity))

                        if j == 3:
                            joint_velocity = np.gradient(joint, np.arange(len(joint) * 0.1, step=0.1))
                            joint_acceleration = np.gradient(joint_velocity, np.arange(len(joint) * 0.1, step=0.1))
                            accel_sore = np.sum(np.square(joint_acceleration))

                    DC_tracker234.append(DC234)
                    DC_tracker1.append(DC1)
                    Acc_tracker.append(accel_sore)
                    V_tracker.append(v_score)

                except:
                    print("asdasdasasdasdasdasdasdasdafdafggfdhjjjsdfjghsfd")
                    print(kipp_winkel, c_axis)

                    DC_tracker234.append(np.average(DC_tracker234))
                    DC_tracker1.append(np.average(DC_tracker1))
                    Acc_tracker.append(np.average(Acc_tracker))
                    V_tracker.append(np.average(V_tracker))

        DC_tracker234 = min_max_scaler.fit_transform(-np.array(DC_tracker234).reshape(-1, 1)) * 100 * 0.3
        DC_tracker1 = min_max_scaler.fit_transform(-np.array(DC_tracker1).reshape(-1, 1)) * 100 * 0.25
        Acc_tracker = min_max_scaler.fit_transform(-np.array(Acc_tracker).reshape(-1, 1)) * 100 * 0.25
        V_tracker = min_max_scaler.fit_transform(-np.array(V_tracker).reshape(-1, 1)) * 100 * 0.2


        score = np.array(DC_tracker234) + np.array(DC_tracker1)+ np.array(V_tracker)+ np.array(Acc_tracker)
        print("Length")
        print(len(score))
        matrix = np.reshape(score, (-1, 55))


        #plt.rcParams["figure.figsize"] = [7.00, 3.50]
        #plt.rcParams["figure.autolayout"] = True
        plt.figure(figsize=(9, 9))
        plt.imshow(matrix)
        # Set xticks and yticks

        x_org = list(range(0,55,4))
        x_new = list(range(-135, 140 ,20))
        plt.xticks(x_org, x_new)
        plt.xlabel("C in degrees [°]")
        plt.ylabel("Tilting in degrees [°]")
        plt.title("Score of the individual boundary conditions as a hyperplane")

        y_org = list(range(0,46,3)) #1
        y_new = list(range(-45, 46 ,6)) #2
        plt.yticks(y_org, y_new)

        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=12)
        cb.locator = tick_locator
        cb.update_ticks()

        plt.grid(color='black', linewidth=0.1)

        pos = np.unravel_index(matrix.argmax(), matrix.shape)
        plt.scatter(pos[1],pos[0],marker = "2",c="red",s = 300)

        print(pos)
        print(matrix[pos[0], pos[1]])

        plt.hlines(pos[0], 0, pos[1], linestyle="dashed", color="black")
        plt.vlines(pos[1], 45, pos[0], linestyle="dashed",color="black")
        plt.text(pos[1]+2, pos[0],
                 f"Global score = {np.round(matrix[pos[0], pos[1]],1)}, \nOptimal rotation C = {-135+5*int(pos[1])}° \nOptimal tilt = {-45+2*int(pos[0])}°",
                 color='black')

        plt.savefig(f"../Latex/figures/best_2D_{toolpath}.png", bbox_inches='tight',dpi=1000)
        #plt.show()
        plt.close()

        np.save(f"matrix_{toolpath}.npy", matrix)
        plt.close()


#basicplot()
#basicscore()
#RealG()
TWODplot()