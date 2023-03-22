
from functools import wraps

class DisplayModelSave(object):
    """
    The DisplayModelSave class has a static method named display. This method takes a function as its argument, and it returns a wrapped version of the input function that prints "Saving model..." after the function is called.

    Example
    ---------

    >>> @DisplayModelSave.display
    ... def my_function():
    ...     # Do some long-running calculation here

    Output
    -------
    >>> my_function()

    # Do some long-running calculation here
    Saving model...

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

            result = func(*args, **kwargs)

            # * Prints that the model has been saved
            print("Saving model...")
            print('\n')

            return result
        return wrapper

    
