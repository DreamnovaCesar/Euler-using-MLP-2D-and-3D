import numpy as np

class ResultPrinter:
    def print_results(self, arrays: np.ndarray, results: np.ndarray) -> None:
        for array, result in zip(arrays, results):
            print('{} -------------- {}'.format(array, result))
            print('The result is: {}'.format(result))
            print('The true value is: {}'.format(result))
            print('\n')