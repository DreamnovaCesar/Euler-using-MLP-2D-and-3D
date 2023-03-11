import numpy as np 

from Class_ShowDataFromTxt import ShowDataFromTxt

class ShowData2D(ShowDataFromTxt):
    """
    A class that loads data from files.
    """
    
    def show_data_from_file(Array : np.ndarray):
        """
        Load a txt file and convert it into a NumPy array.

        Parameters
        ----------
        Array_file : str
            The path to the txt file to be loaded.

        Returns
        -------
        numpy.ndarray
            The loaded NumPy array.

        Raises
        ------
        ValueError
            If the number of dimensions specified in the object constructor
            (i.e., the `_ndim` attribute) is not either "2D" or "3D".

        Notes
        -----
        The txt file should contain comma-separated values representing pixel
        intensities of an image. The number of rows and columns in the txt file
        should be consistent with the image dimensions.

        If the `_ndim` attribute is set to "2D", the loaded array will be of shape
        `(n_rows, n_columns)`. If the `_ndim` attribute is set to "3D", the loaded
        array will be of shape `(n_channels, n_rows, n_columns)`. In the latter
        case, the `n_channels` value is inferred from the number of rows and
        columns in the txt file.

        """

        try:

            # * Convert the array to integers (if it's not already) for consistency.
            Array = Array.astype(int)

            # * Print the array and its shape. (2D)
            print('\n')
            print('Array obtained')
            print('\n')
            print(Array)
            print('\n')
            print('Number of Rows: {}'.format(Array.shape[0]))
            print('\n')
            print('Number of Columns: {}'.format(Array.shape[1]))
            print('\n')

            
            # * Return the NumPy array.
            return Array

        except OSError as err:
                print('Error: {} ❌'.format(err)) #! Alert