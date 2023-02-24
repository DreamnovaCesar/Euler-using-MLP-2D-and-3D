from Article_Euler_Number_Libraries import random

from Article_Euler_Number_Utilities import Utilities

class MetaHeuristic(Utilities):
  """
    Utilities inheritance

    Methods:
        data_dic(): description

        remove_all_files(): description

        remove_random_files(): description

        remove_all(): description

    """
  
  # * Initializing (Constructor)
  def __init__(self, **kwargs) -> None:
      """
      Keyword Args:
          folder (str): description 
          NFR (int): description
      """

      # * Instance attributes (Protected)
      self.__Target = kwargs.get('target', None);
      self.__Coefficients = kwargs.get('coefficients', None);
      self.__Max_iterations = kwargs.get('max_iterations', 15000);
      self.__Initial_temperature = kwargs.get('initial_temperature', 80);
      self.__Cooling_rate = kwargs.get('cooling_rate', 0.001);

  # * Class variables
  def __repr__(self):
          return f"""[{self.__Target}, 
                      {self.__Coefficients},
                      {self.__Max_iterations},
                      {self.__Initial_temperature},
                      {self.__Cooling_rate}]""";

  # * Class description
  def __str__(self):
      return  f'';
  
  # * Deleting (Calling destructor)
  def __del__(self):
      print('');

  # * Get data from a dic
  def data_dic(self):

      return {'Target': str(self.__Target),
              'Coefficients': str(self.__Coefficients),
              'Max_iterations': str(self.__Max_iterations),
              'Initial_temperature': str(self.__Initial_temperature),
              'Cooling_rate': str(self.__Cooling_rate)
              };

  # ? 
  @Utilities.time_func
  def simulated_annealing(self):

    # * Set up the initial solution with random values
    solution = [random.randint(-1, 1) for _ in self.__Coefficients]

    # * Calculate the initial "energy" of the solution
    energy = abs(sum([solution[i] * self.__Coefficients[i] for i in range(len(self.__Coefficients))]) - self.__Target)

    # * Set the initial temperature
    Temperature = self.__Initial_temperature

    for _ in range(self.__Max_iterations):
      # * Choose a random index to modify
      index = random.randint(0, len(solution) - 1)

      # * Create a new solution by modifying the random index
      new_solution = solution[:]
      new_solution[index] = random.randint(-1, 1)

      # * Calculate the "energy" of the new solution
      new_energy = abs(sum([new_solution[i] * self.__Coefficients[i] for i in range(len(self.__Coefficients))]) - self.__Target)

      # * Calculate the change in energy
      delta_energy = new_energy - energy

      # * If the new solution is better, always accept it
      if delta_energy < 0:
        solution = new_solution
        energy = new_energy
      else:
        # * Otherwise, calculate the probability of acceptance
        probability = 2.71828 ** (-delta_energy / Temperature)

        # * Use the probability to determine whether to accept the new solution
        if random.uniform(0, 1) < probability:
          solution = new_solution
          energy = new_energy

      # * Cool the temperature
      Temperature *= 1 - self.__Cooling_rate

    return solution

  