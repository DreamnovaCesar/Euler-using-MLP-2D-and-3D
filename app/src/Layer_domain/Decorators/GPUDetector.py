from functools import wraps

import tensorflow as tf

class GPUDetector(object):
    """
    A class for detecting the availability of a GPU for a function using the GPUDetector.detect_GPU method.

    Examples:
    ---------
    >>> @GPUDetector.detect_GPU
    ... def my_func():
    ...     return tf.reduce_sum(tf.random.normal([1000, 1000]))
    ...
    >>> my_func()
    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
    Found GPU at: /device:GPU:0
    <tf.Tensor: shape=(), dtype=float32, numpy=-508.8744

    Notes
    -----
    This class decorator uses the `functools.wraps` decorator to preserve the
    metadata of the original function, such as the function name, docstring, and
    parameter information.
    

    """

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