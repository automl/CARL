import copy
import numpy as np


class TournamentSelection():

    def __init__(self, config):
        self.cfg = config

    def _tournament(self, fitness_values):
        selection = np.random.randint(0, len(fitness_values), size=self.cfg.nevo.tournament_size)
        selection_values = [fitness_values[i] for i in selection]
        winner = selection[np.argmax(selection_values)]
        return winner

    def select(self, population):
        last_fitness = [indi.fitness[-1] for indi in population]
        rank = np.argsort(last_fitness).argsort()

        max_id = max([ind.index for ind in population])

        elite = copy.deepcopy([population[np.argsort(rank)[-1]]][0])

        new_population = []
        if self.cfg.nevo.elitism:
            new_population.append(elite.clone())
            selection_size = self.cfg.nevo.population_size - 1
        else:
            selection_size = self.cfg.nevo.population_size

        for idx in range(selection_size):
            max_id += 1
            actor_parent = population[self._tournament(rank)]
            new_individual = actor_parent.clone(max_id)
            new_individual.train_log["parent_index"] = actor_parent.index
            new_population.append(new_individual)

        return elite, new_population
