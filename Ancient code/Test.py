import random
import csv 
import numpy as np
import pandas as pd

def simulated_annealing(target, coefficients, max_iterations=15000, initial_temperature=80, cooling_rate=0.001):
  # Set up the initial solution with random values
  solution = [random.randint(-1, 1) for _ in coefficients]

  # Calculate the initial "energy" of the solution
  energy = abs(sum([solution[i] * coefficients[i] for i in range(len(coefficients))]) - target)

  # Set the initial temperature
  temperature = initial_temperature

  for _ in range(max_iterations):
    # Choose a random index to modify
    index = random.randint(0, len(solution) - 1)

    # Create a new solution by modifying the random index
    new_solution = solution[:]
    new_solution[index] = random.randint(-1, 1)

    # Calculate the "energy" of the new solution
    new_energy = abs(sum([new_solution[i] * coefficients[i] for i in range(len(coefficients))]) - target)

    # Calculate the change in energy
    delta_energy = new_energy - energy

    # If the new solution is better, always accept it
    if delta_energy < 0:
      solution = new_solution
      energy = new_energy
    else:
      # Otherwise, calculate the probability of acceptance
      probability = 2.71828 ** (-delta_energy / temperature)

      # Use the probability to determine whether to accept the new solution
      if random.uniform(0, 1) < probability:
        solution = new_solution
        energy = new_energy

    # Cool the temperature
    temperature *= 1 - cooling_rate

  return solution

for i in range(10000):

  Result = []
  Target_result = []

  # Define the target value and the coefficients
  target = -291
  coefficients = [694, 174, 180, 59, 158, 60, 37, 25, 156, 49, 56, 20, 57, 21, 16, 13, 168, 39, 43, 17, 43, 14, 24, 7, 27, 20, 20, 3, 15, 6, 4, 2, 165, 47, 56, 17, 29, 16, 9, 9, 42, 17, 19, 6, 13, 6, 13, 0, 48, 21, 11, 6, 15, 5, 11, 5, 15, 9, 6, 1, 6, 6, 4, 0, 167, 47, 31, 18, 54, 18, 16, 3, 50, 9, 17, 8, 16, 2, 11, 2, 61, 17, 9, 5, 19, 15, 6, 4, 8, 8, 10, 6, 3, 3, 2, 1, 40, 12, 15, 5, 13, 7, 2, 4, 16, 6, 4, 2, 9, 2, 2, 2, 29, 4, 5, 3, 6, 0, 2, 2, 11, 2, 2, 2, 2, 1, 1, 0, 178, 31, 35, 15, 43, 9, 11, 6, 53, 10, 18, 5, 15, 9, 8, 4, 36, 15, 17, 8, 19, 7, 5, 2, 22, 7, 8, 1, 5, 4, 2, 0, 52, 16, 16, 4, 24, 7, 9, 1, 18, 5, 10, 9, 6, 3, 2, 2, 23, 9, 6, 3, 4, 3, 1, 2, 6, 2, 1, 1, 4, 1, 1, 0, 58, 11, 18, 2, 23, 9, 5, 1, 16, 6, 5, 0, 7, 2, 2, 1, 24, 5, 5, 2, 9, 5, 2, 1, 5, 4, 3, 0, 4, 2, 1, 0, 24, 2, 12, 3, 6, 2, 1, 0, 4, 2, 5, 2, 3, 3, 1, 0, 8, 2, 5, 2, 4, 0, 0, 0, 1, 2, 0, 1, 1, 1, 0, 0]
  #coefficients = [19, 25, 21, 30, 19, 19, 11, 28, 20, 14, 29, 23, 27, 22, 25, 64, 25, 21, 14, 25, 15, 27, 4, 3, 1, 1, 4, 1, 1, 4, 7, 12, 19, 12, 30, 21, 1, 2, 2, 1, 8, 1, 26, 5, 4, 3, 5, 19, 24, 30, 20, 60, 0, 3, 6, 13, 0, 3, 3, 15, 4, 24, 21, 50, 21, 11, 0, 0, 23, 29, 2, 8, 12, 0, 3, 14, 25, 10, 3, 9, 25, 31, 1, 7, 24, 57, 2, 10, 1, 5, 4, 12, 4, 15, 12, 32, 6, 4, 2, 8, 3, 4, 3, 14, 3, 6, 4, 12, 3, 12, 10, 38, 27, 0, 6, 18, 10, 9, 15, 36, 5, 19, 13, 31, 14, 45, 40, 93, 25, 0, 8, 2, 15, 3, 3, 6, 21, 2, 25, 4, 26, 5, 3, 19, 8, 1, 1, 3, 2, 2, 4, 8, 5, 5, 6, 12, 2, 14, 16, 36, 31, 3, 27, 8, 3, 3, 3, 13, 30, 7, 52, 9, 5, 9, 10, 38, 29, 5, 0, 20, 10, 11, 12, 45, 4, 11, 10, 42, 10, 44, 40, 89, 15, 1, 0, 7, 27, 3, 3, 18, 30, 2, 5, 14, 71, 8, 13, 46, 29, 6, 8, 23, 3, 20, 10, 39, 8, 13, 14, 46, 14, 27, 37, 96, 29, 3, 7, 13, 5, 16, 15, 36, 7, 13, 16, 35, 14, 45, 33, 96, 64, 12, 14, 31, 10, 34, 37, 103, 12, 23, 37, 107, 39, 106, 108, 291]

  # Solve the linear combination problem
  solution = simulated_annealing(target, coefficients)

  # Print the solution
  print(solution)

  for i1, i2 in zip(solution, coefficients):
    Target_result.append(i1*i2)

  Target_result = sum(Target_result)

  Result.extend(solution)
  Result.append(Target_result)

  with open('Combination2.csv', 'a') as f:
        
      # using csv.writer method from CSV package
      write = csv.writer(f)
      write.writerow(Result)
  