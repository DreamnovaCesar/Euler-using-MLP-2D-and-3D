
from functools import wraps

class DisplayTraining(object):
    """Class decorator to display print statements before and after a function.

    This class decorator can be used to wrap a function and display print statements
    before and after the function is executed. This can be useful for tracking the
    progress of long-running functions.

    Examples
    --------
    To use this decorator, simply add the `@DisplayTraining.display`
    decorator to the function you want to wrap:

    >>> @DisplayTraining.display
    ... def my_function(x):
    ...     print("Executing my_function with x={}".format(x))
    ...     # Do some long-running calculation here
    ...     print("my_function complete")

    Now, when `my_function` is called, the decorator will print a message before and
    after the function is executed:

    Output
    -------

    >>> my_function(42)
    Training...

    Executing my_function with x=42

    my_function complete

    Model trained

    Notes
    -----
    This class decorator uses the `functools.wraps` decorator to preserve the
    metadata of the original function, such as the function name, docstring, and
    parameter information.

    """

    @staticmethod  
    def display(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            # * Prints that training has begun
            print('\n')
            print("Training...")
            print('\n')

            result = func(*args, **kwargs)

            # * Prints that training has completed
            print('\n')
            print("Model trained")
            print('\n')
            return result
        return wrapper

    
