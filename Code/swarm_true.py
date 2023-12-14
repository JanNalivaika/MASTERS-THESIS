import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from matplotlib import ticker
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

def count_direction_changes(time_series):
    direction_changes = 0
    remeber = 1
    time_series = np.round(time_series,2)
    for i in range(1, len(time_series)):
        if time_series[i-1] < time_series[i] and remeber == -1:
            direction_changes += 1
            remeber = 1
        if time_series[i-1] > time_series[i] and remeber == 1:
            direction_changes += 1
            remeber = -1

    return direction_changes



def get_score(list):
    DC_tracker234 = []
    DC_tracker1 = []
    V_tracker = []
    Acc_tracker = []


    for name in list:
        joints = np.load(name)
        DC234 = 0
        DC1 = 0
        accel_sore = 0
        v_score = 0


        for j in range(6):
            joint = joints[:, j]
            joint = np.degrees(joint)

            joint = simplify_angle(joint)

            if j == 0:
                DC1 = count_direction_changes(joint)
                # joint_velocity = np.gradient(joint, np.arange(len(joint) * 0.1, step=0.1))
                # joint_acceleration = np.gradient(joint_velocity,np.arange(len(joint) * 0.1, step=0.1))

                # accel_sore = np.sum(np.square(joint_acceleration))

            if j == 1 or j == 2 or j == 4:
                DC234 += count_direction_changes(joint)

            if j == 5:
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

    DC_tracker234 = min_max_scaler.fit_transform(-np.array(DC_tracker234).reshape(-1, 1)) * 100 * 0.3
    DC_tracker1 = min_max_scaler.fit_transform(-np.array(DC_tracker1).reshape(-1, 1)) * 100 * 0.25
    Acc_tracker = min_max_scaler.fit_transform(-np.array(Acc_tracker).reshape(-1, 1)) * 100 * 0.25
    V_tracker = min_max_scaler.fit_transform(-np.array(V_tracker).reshape(-1, 1)) * 100 * 0.2

    score = np.array(DC_tracker234) + np.array(DC_tracker1) + np.array(V_tracker) + np.array(Acc_tracker)

    return(-score.flatten())



class Particle:
    def __init__(self, x0):
        self.position_i = []  # particle position
        self.velocity_i = []  # particle velocity
        self.pos_best_i = []  # best position individual
        self.err_best_i = -1  # best error individual
        self.err_i = -1  # error individual
        self.visited = []

        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(-1, 1))
            self.position_i.append(x0[i])

            # evaluate current fitness

    def evaluate(self, score, score_list_ind, Position_list_ind):
        # self.err_i = costFunc(self.position_i)
        #self.position_i[0] = int(self.position_i[0])
        #self.position_i[1] = int(self.position_i[1])
        #name = f"Joint_angles_lowres/path_{path}_rot_0_tilt_{(self.position_i[0]*2)-45}_C_{(self.position_i[1]*5)-135}.npy"
        #print(name)
        #file = np.load(name)
        #matrix = np.load(f"matrix_{path}.npy")
        # matrix = normalize(matrix, axis=0, norm='l1')
        self.err_i = score
        # self.err_i = -self.position_i[0]-self.position_i[1]

        # check if current position is an individual best
        min_pos_idx = np.array(np.where(score_list_ind == score_list_ind.min())).flatten()
        min_pos = Position_list_ind[min_pos_idx[0]]

        self.pos_best_i = min_pos
        min_val = np.min(score_list_ind)
        self.err_best_i = min_val

            # update new particle velocity

    def update_velocity(self, pos_best_g):
        w = 0.4  # inertia weight
        c1 = 1  # cognitive constant
        c2 = 1  # social constant

        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()
            #r1 = 1
            #r2 = 1

            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            #vel_cognitive = 0
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            #print(w * self.velocity_i[i] + vel_cognitive + vel_social)

            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social
            #self.velocity_i[i] = 2
            #print("fin")

            # update the particle position based off new velocity updates

    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]

                # adjust minimum position if necessary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]


class PSO:

    def get_best_position(self):
        best_particle = min(self.swarm, key=lambda particle: particle.err_best_i)
        return best_particle.pos_best_i, best_particle.err_best_i

    def __init__(self, bounds, num_particles, max_iter, path):
        global num_dimensions

        num_dimensions = 2
        err_best_g = 1  # best error for group
        pos_best_g = []  # best position for group

        # establish the swarm
        self.swarm = []
        for i in range(0, num_particles):
            self.swarm.append(Particle([np.random.randint(0, high=54), np.random.randint(0, high=45)]))

            # for visualization

        i = 0
        if i == 0:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            for j in range(0, num_particles):
                ax.scatter(self.swarm[j].position_i[1], self.swarm[j].position_i[0], color='r', marker='o',
                           edgecolors='black')
            ax.set_xlim((0, 54))
            ax.set_ylim((0, 45))

            matrix = np.load(f"matrix_{path}.npy")
            # matrix = normalize(matrix, axis=0, norm='l1')
            im = ax.imshow(matrix)

            # ax.cla()

            x_org = list(range(0, 55, 4))
            x_new = list(range(-135, 140, 20))
            plt.xticks(x_org, x_new)
            y_org = list(range(0, 46, 3))  # 1
            y_new = list(range(-45, 46, 6))  # 2
            plt.yticks(y_org, y_new)

            plt.xlabel("C in Degrees [°]")
            plt.ylabel("Tilting in Degrees [°]")
            plt.savefig(f"../Latex/figures/swarm_true/{path}_{i}.png", bbox_inches='tight', dpi=1000)
            plt.close()
            plt.close()

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        while i < max_iter:
            ax.cla()
            # evaluate fitness of each particle
            for j in range(0, num_particles):

                self.swarm[j].position_i[0] = int(self.swarm[j].position_i[0])
                self.swarm[j].position_i[1] = int(self.swarm[j].position_i[1])
                Position_list.append(self.swarm[j].position_i[:])

                p0 = int(self.swarm[j].position_i[0])*5-135
                p1 = int(self.swarm[j].position_i[1])*2-45

                name = f"Joint_angles_lowres/path_{path}_rot_0_tilt_{p1}_C_{p0}.npy"
                File_list.append(name)

            score = get_score(File_list)

            for j in range(0, num_particles):
                score_list_ind = score[j::num_particles]
                Position_list_ind = np.array(Position_list)[j::num_particles][:]
                score_i = score_list_ind[-1]
                self.swarm[j].evaluate(score_i,score_list_ind,Position_list_ind)

                # determine if current particle is the best (globally)

            min_pos_idx = np.array(np.where(score == score.min())).flatten()
            pos_best_g = Position_list[int(min_pos_idx[0])]


            #pos_best_g = [0,0]
            print(pos_best_g)
            ax.scatter(pos_best_g[0], pos_best_g[1], color='green', marker='o', s = 300,
                       edgecolors='black')

                    # update the velocity and position of each particle
            for j in range(0, num_particles):
                self.swarm[j].update_velocity(pos_best_g)
                self.swarm[j].update_position(bounds)

                # plot particles
                ax.scatter(self.swarm[j].position_i[0], self.swarm[j].position_i[1], color='r', marker='o',
                           edgecolors='black')
                ax.set_xlim((0, 54))
                ax.set_ylim((0, 45))

            matrix = np.load(f"matrix_{path}.npy")

            ax.imshow(matrix)

            #plt.pause(0.5)

            i += 1

            x_org = list(range(0, 55, 4))
            x_new = list(range(-135, 140, 20))
            plt.xticks(x_org, x_new)
            y_org = list(range(0, 46, 3))  # 1
            y_new = list(range(-45, 46, 6))  # 2
            plt.yticks(y_org, y_new)

            plt.title(f"Toolpath {path}. PSO-algorithm iteration: {i}")

            plt.xlabel("C in Degrees [°]")
            plt.ylabel("Tilting in Degrees [°]")
            plt.savefig(f"../Latex/figures/swarm_true/{path}_{i}.png", bbox_inches='tight', dpi=1000)
            ax.cla()
        plt.close()


if __name__ == "__main__":




    bounds = [(0, 54), (0, 45)]  # input bounds
    num_particles = 20
    max_iter = 5

    for path in [2]:

        Position_list = []
        File_list = []
        Process_parameter_list = []

        pso = PSO(bounds, num_particles, max_iter, path)

        best_position, best_value = pso.get_best_position()
        print(f"Best Position: {best_position}")
        print(f"Best Value: {-best_value}")
