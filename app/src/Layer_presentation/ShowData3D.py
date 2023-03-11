import numpy as np 

from Class_ShowDataFromTxt import ShowDataFromTxt

class ShowData3D(ShowDataFromTxt):
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
            (i.e., the `_ndim` attribute) is not either "3D".

        Notes
        -----
        The txt file should contain comma-separated values representing pixel
        intensities of an image. The number of rows and columns in the txt file
        should be consistent with the image dimensions.

        """
        
        try:

            # * If `_ndim` is set to "3D", the array is reshaped to 3D and printed.
            Height = Array.shape[0]/Array.shape[1]
            Array = Array.reshape(int(Height), int(Array.shape[1]), int(Array.shape[1]))

            # * Print the array and its shape. (3D)
            print('\n')
            print('Array obtained')
            print('\n')
            print(Array)
            print('\n')
            print('Number of Channels: {}'.format(Array.shape[0]))
            print('\n')
            print('Number of Rows: {}'.format(Array.shape[1]))
            print('\n')
            print('Number of Columns: {}'.format(Array.shape[2]))
            print('\n')

            
            # * Return the NumPy array.
            return Array

        except OSError as err:
                print('Error: {} ❌'.format(err)) #! Alert