from functools import wraps

import tensorflow as tf

class GPUDetector(object):
    """A decorator to detect whether a GPU is available."""

    @staticmethod  
    def detect_GPU(func):  
        """
        Wrapper function that measures the execution time of the input function.

        Parameters
        ----------
        func: The function to be timed..

        Returns
        -------
        The wrapped function that measures the execution time of the input function.
        """
        @wraps(func)  
        def wrapper(*args, **kwargs):  
            gpu_name = tf.test.gpu_device_name();
            gpu_available = tf.config.list_physical_devices('GPU');

            print("\n");
            print(gpu_available);
            print("\n");

            if not gpu_available:
                print("GPU device not found");
            elif "GPU" not in gpu_name:
                print("GPU device not found");
            else:
                print('Found GPU at: {}'.format(gpu_name));
            
            print("\n");
            result = func(*args, **kwargs);
            
            return result
        return wrapper