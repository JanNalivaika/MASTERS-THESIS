import random
import matplotlib.pyplot as plt
import numpy as np

class Particle:
    def __init__(self, x0):
        self.position_i = []  # particle position
        self.velocity_i = []  # particle velocity
        self.pos_best_i = []  # best position individual
        self.err_best_i = -1  # best error individual
        self.err_i = -1  # error individual

        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(-1, 1))
            self.position_i.append(x0[i])

            # evaluate current fitness

    def evaluate(self, costFunc):
        self.err_i = costFunc(self.position_i)

        # check if current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_i

            # update new particle velocity

    def update_velocity(self, pos_best_g):
        w = 0.6  # inertia weight
        c1 = 20  # cognitive constant
        c2 = 3  # social constant

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


    def __init__(self, costFunc, x0, bounds, num_particles, max_iter):
        global num_dimensions

        num_dimensions = len(x0)
        err_best_g = -1  # best error for group
        pos_best_g = []  # best position for group

        # establish the swarm
        self.swarm = []
        for i in range(0, num_particles):
            self.swarm.append(Particle(x0))

            # for visualization
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.set_xlim(bounds[0])
        #ax.set_ylim(bounds[1])
        ax.set_xlim((-40,40))
        ax.set_ylim((-40,40))
        plt.ion()
        plt.show()

        # begin optimization loop
        i = 0
        while i < max_iter:
            # evaluate fitness of each particle
            for j in range(0, num_particles):
                self.swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if self.swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g = list(self.swarm[j].position_i)
                    err_best_g = float(self.swarm[j].err_i)

                    # update the velocity and position of each particle
            for j in range(0, num_particles):
                self.swarm[j].update_velocity(pos_best_g)
                self.swarm[j].update_position(bounds)

                # plot particles
                ax.scatter(self.swarm[j].position_i[0], self.swarm[j].position_i[1], color='b', marker='o')
                ax.set_xlim((-40, 40))
                ax.set_ylim((-40, 40))
            i+=1
            plt.pause(0.01)
            ax.cla()




if __name__ == "__main__":
    # example usage
    def sphere(x):
        fitness = 0.4 / (1 + 0.02 * ((x[0] - (-20)) ** 2 + (x[1] - (-20)) ** 2)) \
                     + 0.2 / (1 + 0.5 * ((x[0] - (-5)) ** 2 + (x[1] - (-25)) ** 2)) \
                     + 0.7 / (1 + 0.01 * ((x[0] - (0)) ** 2 + (x[1] - (30)) ** 2)) \
                     + 1 / (1 + 2 * ((x[0] - (30)) ** 2 + (x[1] - (0)) ** 2)) \
                     + 0.05 / (1 + 0.1 * ((x[0] - (30)) ** 2 + (x[1] - (-30)) ** 2))
        return -fitness
        #return sum([xi ** 2 for xi in x])


    initial = [0, 0]  # initial starting location [x1, x2]
    bounds = [(-40, 40), (-40, 40)]  # input bounds
    num_particles = 3
    max_iter = 100

    pso = PSO(sphere, initial, bounds, num_particles, max_iter)

    best_position, best_value = pso.get_best_position()
    print("Best Position:", best_position)
    print("Best Value:", best_value)


#initial = [5, 5]  # initial starting location [x1, x2]
#bounds = [(-10, 10), (-10, 10)]  # input bounds