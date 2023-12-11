import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from matplotlib import ticker


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

    def evaluate(self, path):
        # self.err_i = costFunc(self.position_i)
        self.position_i[0] = int(self.position_i[0])
        self.position_i[1] = int(self.position_i[1])
        matrix = np.load(f"matrix_{path}.npy")
        # matrix = normalize(matrix, axis=0, norm='l1')
        self.err_i = -matrix[self.position_i[0], self.position_i[1]]
        # self.err_i = -self.position_i[0]-self.position_i[1]

        # check if current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_i

            # update new particle velocity

    def update_velocity(self, pos_best_g):
        w = 0.6  # inertia weight
        c1 = 1.5  # cognitive constant
        c2 = 0.3  # social constant

        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social

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
            self.swarm.append(Particle([np.random.randint(0, high=40), np.random.randint(0, high=40)]))

            # for visualization
        plt.figure(figsize=(10, 10))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.set_xlim(bounds[0])
        # ax.set_ylim(bounds[1])
        # ax.set_xlim((0,50))
        # ax.set_ylim((0,49))
        # plt.ion()
        # plt.show()

        # begin optimization loop
        i = 0
        while i < max_iter:
            ax.cla()
            # evaluate fitness of each particle
            for j in range(0, num_particles):
                self.swarm[j].evaluate(path)

                # determine if current particle is the best (globally)
                if self.swarm[j].err_i < err_best_g:
                    pos_best_g = list(self.swarm[j].position_i)
                    err_best_g = float(self.swarm[j].err_i)

                    # update the velocity and position of each particle
            for j in range(0, num_particles):
                self.swarm[j].update_velocity(pos_best_g)
                self.swarm[j].update_position(bounds)

                # plot particles
                ax.scatter(self.swarm[j].position_i[1], self.swarm[j].position_i[0], color='r', marker='o',
                           edgecolors='black')
                ax.set_xlim((0, 54))
                ax.set_ylim((0, 45))

            matrix = np.load(f"matrix_{path}.npy")
            # matrix = normalize(matrix, axis=0, norm='l1')
            im = ax.imshow(matrix)
            if i == 0:
                fig.colorbar(im)

            # plt.pause(0.5)
            # ax.cla()
            i += 1

            x_org = list(range(0, 55, 4))
            x_new = list(range(-135, 140, 20))
            plt.xticks(x_org, x_new)
            y_org = list(range(0, 46, 3))  # 1
            y_new = list(range(-45, 46, 6))  # 2
            plt.yticks(y_org, y_new)

            # cb = plt.colorbar()
            # tick_locator = ticker.MaxNLocator(nbins=12)
            # cb.locator = tick_locator
            # cb.update_ticks()

            # plt.title(f"Traversing the hyperplane of toolpath {path} with a PSO-algorithm. Iteration: {i}")
            plt.title(f"Toolpath {path}. PSO-algorithm iteration: {i}")

            plt.xlabel("C in Degrees [°]")
            plt.ylabel("Tilting in Degrees [°]")
            plt.savefig(f"../Latex/figures/swarm/{path}_{i}.png", bbox_inches='tight', dpi=1000)
            print(i, path)
        plt.close()


if __name__ == "__main__":

    bounds = [(0, 45), (0, 54)]  # input bounds
    num_particles = 5
    max_iter = 10

    for path in [1, 2, 3]:
        pso = PSO(bounds, num_particles, max_iter, path)

        best_position, best_value = pso.get_best_position()
        print(f"Best Position: {best_position}")
        print(f"Best Value: {-best_value}")

# initial = [5, 5]  # initial starting location [x1, x2]
# bounds = [(-10, 10), (-10, 10)]  # input bounds


# error with
# path_2_rot_0_tilt_3_C_-23
# path_2_rot_0_tilt_3_C_-22
# path_1_rot_0_tilt_-1_C_15
