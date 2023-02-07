import tensorflow as tf
import numpy as np
import random

def generate_population(population_size, num_hyperparams, max_value):
  population = []
  for i in range(population_size):
    hyperparams = []
    for j in range(num_hyperparams):
      hyperparams.append(random.uniform(0, max_value))
    population.append(hyperparams)
  return population

def evaluate_fitness(model, population, data, labels):
  fitness = []
  for hyperparams in population:
    model.set_weights(hyperparams)
    loss, accuracy = model.evaluate(data, labels)
    fitness.append(accuracy)
  return fitness

def select_parents(population, fitness, num_parents):
  parents = []
  for i in range(num_parents):
    max_index = np.argmax(fitness)
    parents.append(population[max_index])
    fitness[max_index] = -1
  return parents

def breed_offspring(parents, offspring_size):
  offspring = []
  for i in range(offspring_size):
    parent1 = random.choice(parents)
    parent2 = random.choice(parents)
    offspring.append(parent1 + parent2)
  return offspring

def mutate_offspring(offspring, mutation_prob, max_value):
  for i in range(len(offspring)):
    for j in range(len(offspring[i])):
      if random.uniform(0, 1) < mutation_prob:
        offspring[i][j] = random.uniform(0, max_value)
  return offspring

def train_model(model, data, labels, population_size, num_generations, num_parents, offspring_size, mutation_prob, max_value):
  population = generate_population(population_size, len(model.get_weights()), max_value)
  for i in range(num_generations):
    fitness = evaluate_fitness(model, population, data, labels)
    parents = select_parents(population, fitness, num_parents)
    offspring = breed_offspring(parents, offspring_size)
    offspring = mutate_offspring(offspring, mutation_prob, max_value)
    population = offspring
  best_index = np.argmax(fitness)
  best_weights = population[best_index]
  model.set_weights(best_weights)
  return model