import numpy as np
import random


def objective_function(pop):
    fitness = np.zeros(pop.shape[0])
    for i in range(pop.shape[0]):
        x = pop[i]
        fitness[i] = 0.4 / (1 + 0.02 * ((x[0] - (-20)) ** 2 + (x[1] - (-20)) ** 2)) \
                     + 0.2 / (1 + 0.5 * ((x[0] - (-5)) ** 2 + (x[1] - (-25)) ** 2)) \
                     + 0.7 / (1 + 0.01 * ((x[0] - (0)) ** 2 + (x[1] - (30)) ** 2)) \
                     + 1 / (1 + 2 * ((x[0] - (30)) ** 2 + (x[1] - (0)) ** 2)) \
                     + 0.05 / (1 + 0.1 * ((x[0] - (30)) ** 2 + (x[1] - (-30)) ** 2))
    return fitness


def selection(pop, fitness, pop_size):
    elite = np.argmax(fitness)
    index = np.delete(np.arange(pop.shape[0]), elite)
    P = fitness[index] / np.sum(fitness[index])
    index_selected = np.random.choice(index, size=pop_size - 1, replace=True, p=P)
    next_generation = np.concatenate((pop[elite][np.newaxis], pop[index_selected]), axis=0)
    return next_generation


def crossover(pop, crossover_rate):
    offspring = np.zeros((crossover_rate, pop.shape[1]))
    for i in range(int(crossover_rate / 2)):
        r1, r2 = random.sample(range(pop.shape[0]), 2)
        cutting_point = random.randint(1, pop.shape[1] - 1)
        offspring[2 * i, :cutting_point] = pop[r1, :cutting_point]
        offspring[2 * i, cutting_point:] = pop[r2, cutting_point:]
        offspring[2 * i + 1, :cutting_point] = pop[r2, :cutting_point]
        offspring[2 * i + 1, cutting_point:] = pop[r1, cutting_point:]
    return offspring


def mutation(pop, mutation_rate):
    offspring = np.zeros((mutation_rate, pop.shape[1]))
    for i in range(int(mutation_rate / 2)):
        r1, r2 = random.sample(range(pop.shape[0]), 2)
        cutting_point = random.randint(0, pop.shape[1] - 1)
        offspring[2 * i] = pop[r1]
        offspring[2 * i, cutting_point] = pop[r2, cutting_point]
        offspring[2 * i + 1] = pop[r2]
        offspring[2 * i + 1, cutting_point] = pop[r1, cutting_point]
    return offspring


def local_search(pop, fitness, lower_bounds, upper_bounds, step_size, rate):
    index = np.argmax(fitness)  # Get the index of the best individual
    best_individual = pop[index]  # Get the best individual from the population
    offspring = np.repeat(best_individual[np.newaxis], rate * 2 * pop.shape[1], axis=0)

    for i in range(rate * 2 * pop.shape[1]):
        j = i % pop.shape[1]
        mutation = random.uniform(-step_size, step_size)
        candidate_solution = np.copy(offspring[i])
        candidate_solution[j] += mutation
        candidate_solution = np.clip(candidate_solution, lower_bounds[j], upper_bounds[j])
        candidate_fitness = objective_function(candidate_solution[np.newaxis])
        if candidate_fitness > fitness[index]:
            offspring[i] = candidate_solution

    return offspring


def main():
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    # Axes3D import has side effects, it enables using projection='3d' in add_subplot
    import matplotlib.pyplot as plt
    import random
    from matplotlib import cm

    def fun(x):
        return 0.4 / (1 + 0.02 * ((x[0] - (-20)) ** 2 + (x[1] - (-20)) ** 2)) \
                     + 0.2 / (1 + 0.5 * ((x[0] - (-5)) ** 2 + (x[1] - (-25)) ** 2)) \
                     + 0.7 / (1 + 0.01 * ((x[0] - (0)) ** 2 + (x[1] - (30)) ** 2)) \
                     + 1 / (1 + 2 * ((x[0] - (30)) ** 2 + (x[1] - (0)) ** 2)) \
                     + 0.05 / (1 + 0.1 * ((x[0] - (30)) ** 2 + (x[1] - (-30)) ** 2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-50.0, 50.0, 1)
    X, Y = np.meshgrid(x, y)
    zs = np.array(fun([np.ravel(X), np.ravel(Y)]))
    Z = zs.reshape(X.shape)
    my_col = cm.jet(Z / np.amax(Z))
    ax.plot_surface(X, Y, Z,facecolors = my_col,linewidth=0, antialiased=True)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()







    # Set the parameters for the genetic algorithm
    # print(objective_function(np.asarray([[30,0]])))
    initial_population_size = 100
    num_variables = 2
    num_generations = 100
    crossover_rate = int(initial_population_size*5)
    mutation_rate = int(initial_population_size*3)
    step_size = 10
    local_search_rate =  1

    # Define the bounds for each variable
    lim = 40
    lower_bounds = [-lim, -lim]
    upper_bounds = [lim, lim]

    # Initialize the population randomly within the bounds
    population = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(initial_population_size, num_variables))

    # Run the genetic algorithm for a specified number of generations
    for generation in range(num_generations):
        # Evaluate the fitness of the population
        fitness = objective_function(population)

        # Perform selection to create the next generation
        next_generation = selection(population, fitness, initial_population_size)

        # Perform crossover
        offspring = crossover(next_generation, crossover_rate)

        # Perform mutation
        offspring = mutation(offspring, mutation_rate)

        # Combine the current population with the offspring
        population = np.vstack((population, offspring))

        # Perform local search
        population = local_search(population, fitness, lower_bounds, upper_bounds, step_size, local_search_rate)

        # Select the best individuals for the next generation
        fitness = objective_function(population)
        sorted_indices = np.argsort(fitness)[::-1]
        population = population[sorted_indices[:initial_population_size]]

        # Update the fitness array after selecting the best individuals
        fitness = fitness[sorted_indices[:initial_population_size]]

        # Print the best fitness value for each generation
        best_fitness = np.max(fitness)
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

        # Print the final best solution
    best_solution = population[np.argmax(fitness)]
    print("Final Best Solution:")
    print(best_solution)
    print("Final Best Fitness:")
    print(np.max(fitness))

    return crossover_rate, mutation_rate, step_size, local_search_rate, np.max(fitness)
if __name__ == "__main__":

    main()



