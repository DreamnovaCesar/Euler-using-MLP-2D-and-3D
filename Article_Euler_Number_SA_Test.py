from Article_Euler_Number_Simulated_Annealing import MetaHeuristic
from Article_Euler_Number_Libraries import csv

for i in range(10000):

  Result = []
  Target_result = []

  # * Define the target value and the coefficients
  Final_target = -291
  Final_coefficients = [694, 174, 180, 59, 158, 60, 37, 25, 156, 49, 56, 20, 57, 21, 16, 13, 168, 39, 43, 17, 43, 14, 24, 7, 27, 20, 20, 3, 15, 6, 4, 2, 165, 47, 56, 17, 29, 16, 9, 9, 42, 17, 19, 6, 13, 6, 13, 0, 48, 21, 11, 6, 15, 5, 11, 5, 15, 9, 6, 1, 6, 6, 4, 0, 167, 47, 31, 18, 54, 18, 16, 3, 50, 9, 17, 8, 16, 2, 11, 2, 61, 17, 9, 5, 19, 15, 6, 4, 8, 8, 10, 6, 3, 3, 2, 1, 40, 12, 15, 5, 13, 7, 2, 4, 16, 6, 4, 2, 9, 2, 2, 2, 29, 4, 5, 3, 6, 0, 2, 2, 11, 2, 2, 2, 2, 1, 1, 0, 178, 31, 35, 15, 43, 9, 11, 6, 53, 10, 18, 5, 15, 9, 8, 4, 36, 15, 17, 8, 19, 7, 5, 2, 22, 7, 8, 1, 5, 4, 2, 0, 52, 16, 16, 4, 24, 7, 9, 1, 18, 5, 10, 9, 6, 3, 2, 2, 23, 9, 6, 3, 4, 3, 1, 2, 6, 2, 1, 1, 4, 1, 1, 0, 58, 11, 18, 2, 23, 9, 5, 1, 16, 6, 5, 0, 7, 2, 2, 1, 24, 5, 5, 2, 9, 5, 2, 1, 5, 4, 3, 0, 4, 2, 1, 0, 24, 2, 12, 3, 6, 2, 1, 0, 4, 2, 5, 2, 3, 3, 1, 0, 8, 2, 5, 2, 4, 0, 0, 0, 1, 2, 0, 1, 1, 1, 0, 0]
  

  # * Solve the linear combination problem
  SA = MetaHeuristic(target = Final_target, coefficients = Final_coefficients, max_iterations = 15000, initial_temperature = 80, cooling_rate = 0.001)
  Solution = SA.simulated_annealing()

  # * Print the solution
  print(Solution)

  for i1, i2 in zip(Solution, Final_coefficients):
    Target_result.append(i1*i2)

  Target_result = sum(Target_result)

  Result.extend(Solution)
  Result.append(Target_result)

  with open('Combination2.csv', 'a') as f:
        
      # * using csv.writer method from CSV package
      write = csv.writer(f)
      write.writerow(Result)