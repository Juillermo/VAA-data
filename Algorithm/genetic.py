from random import randint, random
from copy import deepcopy
from datetime import datetime

from Algorithm.model import IntegerProgramming
from Algorithm.data import RANDOM_STATE, QUESTIONS, DataHolder, L_SET
from Algorithm.utils import unfold_matrix, PLOTS_PATH, plot_confusion

GENETIC_PATH = "genetic_output/"


class Gene:
    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate
        if randint(0, 1):
            self.allele = True
        else:
            self.allele = False

    @staticmethod
    def set_mutation_rate(length, alpha=1):
        Gene.mutation_rate = int(length / alpha)

    def mutate(self):
        if not randint(0, self.mutation_rate - 1):
            self.allele = not self.allele

    def get_allele(self):
        return self.allele

    def __repr__(self):
        return str(int(self.allele))


class Individual:
    number_of_evaluations = 0
    bit_string = 0

    def __init__(self, genome=[]):
        self.fitness = 0
        if not genome:
            self.genome = [Gene(Individual.bit_string)
                           for _ in range(Individual.bit_string)]
        else:
            self.genome = genome

    def mutate(self):
        [gene.mutate() for gene in self.genome]

    @staticmethod
    def crossover(father1, father2):
        cross_point1 = randint(1, Individual.bit_string - 1)
        cross_point2 = randint(1, Individual.bit_string - 1)
        while cross_point1 == cross_point2:
            cross_point1 = randint(1, Individual.bit_string - 1)
            cross_point2 = randint(1, Individual.bit_string - 1)
        if cross_point1 > cross_point2:
            temp = cross_point1
            cross_point1 = cross_point2
            cross_point2 = temp
        mixed_genome = [deepcopy(gene) if i < cross_point1 or i >= cross_point2
                        else deepcopy(father2.get_gene(i))
                        for (i, gene) in enumerate(father1.genome)]
        ind_type = father1.__class__
        return ind_type(mixed_genome)

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_n_evaluations(self):
        return self.number_of_evaluations

    def reset_evaluations(self):
        Individual.number_of_evaluations = 0

    def get_gene(self, key):
        return self.genome[key]

    def get_genome(self):
        return self.genome

    @staticmethod
    def set_bit_string(bit_string):
        Individual.bit_string = bit_string

    def __repr__(self):
        return "".join([str(gene) for gene in self.genome]) \
               + ", Fitness: " + str(self.fitness)


class Population:
    def __init__(self, pop_size=100):
        self.pop_size = pop_size
        self.individuals = [Individual() for _ in range(self.pop_size)]
        self.best_individual = None
        self.worst_fitness = 0
        self.worst_fitnesses = []

    def pick_individual(self):
        rankings = [self.worst_fitness - individual.get_fitness()
                    for individual in self.individuals]
        total_fitness = sum(rankings)
        rankings = [fitness / total_fitness for fitness in rankings]

        roulette_sum = 0
        wheel = []
        for fitness in rankings:
            roulette_sum += fitness
            wheel.append(roulette_sum)
        pick = random()
        i = 0
        while wheel[i] < pick:
            i += 1

        return self.individuals[i]

    def get_individuals(self):
        return self.individuals

    def set_individuals(self, individuals):
        self.individuals = individuals
        self.refresh_best_individual()
        self.refresh_worst_fitness()

    def get_best_individual(self):
        return self.best_individual

    def refresh_best_individual(self):
        best_fitness = 1000000
        best_individual = None
        for individual in self.individuals:
            if individual.get_fitness() < best_fitness:
                best_individual = individual
                best_fitness = individual.get_fitness()
        if best_individual is None:
            raise Exception("Failure at detecting best individual")
        self.best_individual = best_individual

    def refresh_worst_fitness(self):
        worst_fitness = 0
        for individual in self.individuals:
            if individual.get_fitness() > worst_fitness:
                worst_fitness = individual.get_fitness()
        self.worst_fitnesses.insert(0, worst_fitness)
        if len(self.worst_fitnesses) > 5:
            self.worst_fitnesses.pop()
        self.worst_fitness = max(self.worst_fitnesses)

    def get_size(self):
        return self.pop_size


class Evolver:
    def __init__(self, training_steps=5):
        data_obj = DataHolder()
        self.U_train, self.P_train, self.V_train = data_obj.get_training_data()
        self.model = IntegerProgramming(training_steps)

        self.n_of_evals = 0
        self.generation = 0

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_name = GENETIC_PATH + timestamp + '.txt'

    def evolve(self, goal_evals=10, population_size=10):
        with open(self.file_name, 'w+') as a_file:
            bit_string = len(QUESTIONS)
            Individual.set_bit_string(bit_string)
            population = Population(population_size)

            [self.evaluate(individual)
             for individual in population.get_individuals()]

            population.refresh_best_individual()
            population.refresh_worst_fitness()
            self.print_individual(population.get_best_individual())
            a_file.write(str(self.n_of_evals) + ", " +
                         str(population.get_best_individual().get_fitness()) + "\n")

            while self.n_of_evals < goal_evals:
                self.generation += 1
                new_pop = [population.get_best_individual()]
                for _ in range(population.get_size() - 1):
                    ind1 = population.pick_individual()
                    if random() <= 0.6:
                        ind2 = population.pick_individual()
                        child = Individual.crossover(ind1, ind2)
                    else:
                        child = deepcopy(ind1)
                    child.mutate()
                    new_pop.append(child)

                [self.evaluate(individual) for individual in new_pop]
                population.set_individuals(new_pop)

                self.print_individual(population.get_best_individual())
                a_file.write(str(self.n_of_evals) + ", " +
                             str(population.get_best_individual().get_fitness()) + "\n")

            print("Evolution process finished, num of generations: " + str(self.generation))

        with open(GENETIC_PATH + 'spread.txt', 'w+') as a_file:
            [a_file.write(str((ind.get_genome(), ind.get_fitness()) + "\n")
                          for ind in population.get_individuals()]

    def evaluate(self, individual):
        """Evaluates fitness using the other populations' best individuals"""
        matrix_combination = individual.get_genome()
        individual.set_fitness(self.model.train(self.U_train, self.P_train, self.V_train, matrix_combination))
        self.n_of_evals += 1

    def print_individual(self, individual):
        print("Evals" + str(self.n_of_evals) + ", " + str(individual) +
              ", " + str(individual.get_genome()))


if __name__ == "__main__":
    gradient_steps_per_evaluation = 5
    goal_evals = 100
    population_size = 10

    evolver = Evolver(gradient_steps_per_evaluation)
    evolver.evolve(goal_evals, population_size)
