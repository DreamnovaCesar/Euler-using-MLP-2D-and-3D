import numpy as np

from .Generator import Generator

class GeneratorObject(Generator):
    """
    A generator that produces a 3D array with given probabilities of 0 and 1.

    Parameters
    ----------
    Prob_0 : float
        Probability of 0.
    Prob_1 : float
        Probability of 1.
    _Width : int
        Width of the 3D array.
    _Height : int
        Height of the 3D array.
    _Depth : int
        Depth of the 3D array.

    Returns
    -------
    Data_3D_edges_complete : ndarray
        A 3D array with given probabilities of 0 and 1, with edges added and concatenated with zeros.

    """
    def generator(
        Prob_0: float, 
        Prob_1: float,
        _Width : int,
        _Height : int,
        _Depth : int,
    ):
        """
        Generates a 3D array with given probabilities of 0 and 1, with edges added and concatenated with zeros.

        Parameters
        ----------
        Prob_0 : float
            Probability of 0.
        Prob_1 : float
            Probability of 1.
        _Width : int
            Width of the 3D array.
        _Height : int
            Height of the 3D array.
        _Depth : int
            Depth of the 3D array.

        Returns
        -------
        Data_3D_edges_complete : ndarray
            A 3D array with given probabilities of 0 and 1, with edges added and concatenated with zeros.

        """

        # * Generate random 3D binary data with given probabilities
        Data_3D = np.random.choice(2, _Height * _Depth * _Width, p = [Prob_0, Prob_1]);
        Data_3D = Data_3D.reshape((_Height * _Depth), _Width);
        Data_3D_plot = Data_3D.reshape((_Height, _Depth, _Width));
        
        # * Initialize arrays with zeros to add edges to the data
        Data_3D_edges_complete = np.zeros((Data_3D_plot.shape[1] + 2, 
                                           Data_3D_plot.shape[2] + 2))
        Data_3D_edges_concatenate = np.zeros((Data_3D_plot.shape[1] + 2, 
                                              Data_3D_plot.shape[2] + 2))
        
        # * Add edges to the 2D data
        Data_3D_read = np.zeros((Data_3D.shape[0] + 2, 
                                 Data_3D.shape[1] + 2))
        # * Add edges to the 3D data
        Data_3D_edges = np.zeros((Data_3D_plot.shape[0] + 2,
                                  Data_3D_plot.shape[1] + 2, 
                                  Data_3D_plot.shape[2] + 2))
        
        # * Get 3D image and interpretation of 3D from 2D .txt
        Data_3D_read[1:Data_3D_read.shape[0] - 1, 
                     1:Data_3D_read.shape[1] - 1] = Data_3D
        
        Data_3D_edges[1:Data_3D_edges.shape[0] - 1, 
                      1:Data_3D_edges.shape[1] - 1, 
                      1:Data_3D_edges.shape[2] - 1] = Data_3D_plot

        # * Concatenate zeros to the 2D data
        Data_3D_read = np.concatenate((Data_3D_edges_concatenate, Data_3D_read), axis = 0)
        Data_3D_read = np.concatenate((Data_3D_read, Data_3D_edges_concatenate), axis = 0)

        # * Concatenate zeros to the 3D data
        for k in range(len(Data_3D_edges) - 2):
            Data_3D_edges_complete = np.concatenate((Data_3D_edges_complete, Data_3D_edges[k + 1]), axis = 0)

        Data_3D_edges_complete = np.concatenate((Data_3D_edges_complete, Data_3D_edges_concatenate), axis = 0)

        return Data_3D_edges_complete
    