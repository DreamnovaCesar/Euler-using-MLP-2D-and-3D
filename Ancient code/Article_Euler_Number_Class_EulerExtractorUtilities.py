from Article_Euler_Number_Libraries import *
from Article_Euler_Number_Utilities import Utilities
from Article_Euler_Number_Info_3D_General import *
from Article_Euler_Number_Class_BinaryConversion import BinaryConversion

from abc import ABC
from abc import abstractmethod

import json

# ? Utilities class for Euler number extraction for 2D and 3D
class EulerExtractorUtilities(Utilities):
    """
    Utilities class for Euler number extraction for 2D and 3D
    
    Attributes
    ----------
        _Input : numpy.ndarray
            Input data.
        _Output : numpy.ndarray
            Output data.
        _ndim : str

        _Folder_data : str 
            Folder containing the data.
        _Model_name : str
            Name of the model used.
        _Columns : list
            List of column names for the data.

    Methods
    ----------
        __repr__()
        Return the string representation of this EulerExtractorUtilities instance.
        __str__()
            Return a string representation of this EulerExtractorUtilities instance.
        __del__()
            Destructor that is called when this EulerExtractorUtilities instance is destroyed.
        data_dic():
            Returns a dictionary containing the object's attributes.
        create_json_file()
            Creates a JSON file with the object's attributes.
        create_dataframe_history()
            Method to create dataframe from the trained model.
        plot_data_loss()
            Method to plot the loss. plot_data_loss takes the history 
            of the training process and plots the loss of the model as 
            a function of the number of epochs. 
        plot_data_accuracy()
            Method to plot the accuracy. plot_data_accuracy also takes the history 
            of the training process as an argument and plots the accuracy of the model as 
            a function of the number of epochs.
        read_image_with_metadata() : abstract method
            Load a txt file and convert it into a NumPy array.
        model_prediction() : abstract method
            Uses machine learning models to make predictions on an array.
        MLP_training_euler_number() : abstract method
            Train a multi-layer perceptron (MLP).
    """

    # * Initializing (Constructor)
    def __init__(self, **kwargs) -> None:
        """
        Constructor method for EulerExtractorUtilities.

        Keyword Arguments
        -----------------
        Input : numpy.ndarray
            Input data.
        Output : numpy.ndarray
            Output data.
        ndim : int 
            Number of dimensions.
        Folder : str
            Folder containing the data.
        ModelName : str
            Name of the model used.
        """

        # * General parameters
        self._Input = kwargs.get('Input', None)
        self._Output = kwargs.get('Output', None)
        self._ndim = kwargs.get('ndim', None)
        self._Folder_data = kwargs.get('Folder', None)
        self._Model_name = kwargs.get('ModelName', None)
        self._Columns = ["Loss", "Accuracy"]

    # * Class variables
    def __repr__(self):
        """
        Returns the string representation of this EulerExtractorUtilities instance.

        Returns
        ----------
        str 
            The string representation of this EulerExtractorUtilities instance.
        """
        return f'''[{self._Input}, 
                    {self._Output}, 
                    {self._Folder_data}, 
                    {self._Model_name}, 
                    {self._Columns}]''';

    # * Class description
    def __str__(self):
        """
        Returns a string representation of this EulerExtractorUtilities instance.

        Returns
        ----------
        str 
            A string representation of this EulerExtractorUtilities instance.
        """
        return  f'Utilities class for Euler number extraction.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        """
        Destructor that is called when this EulerExtractorUtilities instance is destroyed.
        """
        print('Destructor called, Euler number class destroyed.');

    # * Get data from a dic
    def data_dic(self):
        """
        Returns a dictionary containing the instance's Inputs, Outputs, ndim and, Folder_data, Model_name values.

        Returns
        ----------
        dict 
            A dictionary containing the instance's Inputs, Outputs, ndim, Folder_data, Model_name values.
        """
        return {'Inputs': str(self._Input),
                'Outputs': str(self._Output),
                'Folder_data': str(self._Folder_data),
                'Model_name': str(self._Model_name),
                };

    # * Creates a JSON file with the given data and saves it to the specified file path.
    def create_json_file(self):
        """
        Creates a JSON file with the given data and saves it to the specified file path.
        A JSON file containing the instance's Inputs, Outputs, ndim, Folder_data, Model_name values.

        Returns
        ----------
        None
        """
        Data = {'Inputs': str(self._Input),
                'Outputs': str(self._Output),
                'Folder_data': str(self._Folder_data),
                'Model_name': str(self._Model_name),
                };

        with open('JSON documents', 'w') as file:
            json.dump(Data, file)

    # * _Input attribute
    @property
    def _Input_property(self):
        """Getter method for the `Input` property."""
        return self._Input

    @_Input_property.setter
    def _Input_property(self, New_value):
        """Setter method for the `Input` property.

        Parameters
        ----------
        New_value : str
            The new value to be assigned to the `Input` attribute.
        """
        self._Input = New_value
    
    @_Input_property.deleter
    def _Input_property(self):
        """Deleter method for the `Input` property."""
        print("Deleting Input...")
        del self._Input

    # * _Output attribute
    @property
    def _Output_property(self):
        """Getter method for the `Output` property."""
        return self._Output

    @_Output_property.setter
    def _Output_property(self, New_value):
        """Setter method for the `Output` property.

        Parameters
        ----------
        New_value : str
            The new value to be assigned to the `Output` attribute.
        """
        self._Output = New_value
    
    @_Output_property.deleter
    def _Output_property(self):
        """Deleter method for the `Output` property."""
        print("Deleting Output...")
        del self._Output

    # * _Folder_data attribute
    @property
    def _Folder_data_property(self):
        """Getter method for the `Folder_data` property."""
        return self._Folder_data

    @_Folder_data_property.setter
    def _Folder_data_property(self, New_value):
        """Setter method for the `Folder_data` property.

        Parameters
        ----------
        New_value : str
            The new value to be assigned to the `Folder_data` attribute.
        """
        self._Folder_data = New_value
    
    @_Folder_data_property.deleter
    def _Folder_data_property(self):
        """Deleter method for the `Folder_data` property."""
        print("Deleting Folder_data...")
        del self._Folder_data

     # * _Model_name attribute
    @property
    def _Model_name_property(self):
        """Getter method for the `Model_name` property."""
        return self._Model_name

    @_Model_name_property.setter
    def _Model_name_property(self, New_value):
        """Setter method for the `Model_name` property.

        Parameters
        ----------
        New_value : str
            The new value to be assigned to the `Model_name` attribute.
        """
        self._Model_name = New_value
    
    @_Model_name_property.deleter
    def _Model_name_property(self):
        """Deleter method for the `Model_name` property."""
        print("Deleting Model_name...")
        del self._Model_name

    # ? Static method to create dataframe from history
    @staticmethod
    @Utilities.time_func
    def create_dataframe_history(Column_names: Any, Folder_save: str, CSV_name: str, Hist_data: Any) -> None: 
        """
        Method to create dataframe from the trained model

        Parameters
        ----------
        Column_names : str
            Column names for the dataframe
        Folder_save : str 
            Folder to save the dataframe
        CSV_name : str
            Name of the CSV file
        Hist_data : object
            History data from the trained model to be added to the dataframe
        """
        
        # * Create an empty dataframe with column names for the dataframe
        Dataframe_created = pd.DataFrame(columns = Column_names)

        # * Extract loss and accuracy data from the history
        Accuracy = Hist_data.history["accuracy"]
        Loss = Hist_data.history["loss"]

        # * Combine loss and accuracy data
        History_data = zip(Loss, Accuracy)

        # * Add the combined data to the dataframe
        for _, (Loss_, Accuracy_) in enumerate(History_data):
            Dataframe_created.loc[len(Dataframe_created.index)] = [Loss_, Accuracy_]

        # * Save the dataframe to a CSV file
        Dataframe_name = "Dataframe_{}_Loss_And_Accuracy.csv".format(CSV_name)
        Dataframe_folder = os.path.join(Folder_save, Dataframe_name)
        Dataframe_created.to_csv(Dataframe_folder)

    # ? Method to plot loss
    @Utilities.time_func
    def plot_data_loss(self, Hist_data: object) -> None:
        """
        Method to plot the loss. plot_data_loss takes the history 
        of the training process and plots the loss of the model as 
        a function of the number of epochs. 

        Parameters
        ----------
        Hist_data : object 
            History data to be plotted (Loss extraction).

        """

        # * Create a figure with size of 8x8
        plt.figure(figsize = (8, 8))
        plt.title('Training loss')
        plt.xlabel ("# Epoch")
        plt.ylabel ("# Loss")
        plt.ylim([0, 2.0])
        plt.plot(Hist_data.history["loss"])
        #plt.show()
        plt.close()

        # * Set the name of the figure to be saved and set the folder to save the figure
        Figure_name = "Figure_Loss_{}.png".format(self._Model_name)
        Figure_name_folder = os.path.join(self._Folder_data, Figure_name)

        # * Save the figure in the specified folder
        plt.savefig(Figure_name_folder)
        plt.close()

    # ? Method to plot accuracy
    @Utilities.time_func
    def plot_data_accuracy(self, Hist_data: object) -> None:
        """
        Method to plot the accuracy. plot_data_accuracy also takes the history 
        of the training process as an argument and plots the accuracy of the model as 
        a function of the number of epochs.

        Parameters
        ----------
        Hist_data : object 
            History data to be plotted (Accuracy extraction).
        """

        # * Create a figure with size of 8x8
        plt.figure(figsize = (8, 8))
        plt.title('Training accuracy')
        plt.xlabel ("# Epoch")
        plt.ylabel ("# Acuracy")
        plt.ylim([0, 1])
        plt.plot(Hist_data.history["accuracy"])
        #plt.show()

        # * Set the name of the figure to be saved and set the folder to save the figure
        Figure_name = "Figure_Accuracy_{}.png".format(self._Model_name)
        Figure_name_folder = os.path.join(self._Folder_data, Figure_name)

        # * Save the figure in the specified folder
        plt.savefig(Figure_name_folder)
        plt.close()

    # ? Load a txt file and convert it into a NumPy array..
    @abstractmethod
    def read_image_with_metadata(self) -> None:
        """
        Abstract method. Load a txt file and convert it into a NumPy array.

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

        pass
        
    # ? Abstract method. Train a multi-layer perceptron (MLP)    
    @profile
    @Utilities.time_func
    @Utilities.detect_GPU
    @abstractmethod
    def MLP_training_euler_number(self) -> None:
        """
        Abstract method. Train a multi-layer perceptron (MLP)

        Notes
        -----

        The function takes three optional parameters: Opt, lr, and Epochs. Opt is the name of the optimizer to use, 
        lr is the learning rate, and Epochs is the number of training epochs. The function first checks the value of 
        the _ndim attribute to determine the input shape of the MLP. If _ndim is "2D", the MLP architecture has one hidden layer. 
        If _ndim is "3D", the MLP architecture has two hidden layers.

        The function then compiles the model with the specified optimizer and loss function. It trains the model and stores 
        the training history in Hist_data. Finally, it saves the trained model as an h5 file and the history of the trained model inside a csv file. 
        The function also plots the data loss and accuracy.

        If _ndim is not set to "2D" or "3D", the function raises a ValueError.
        """

        pass

    # ? Method to utilize prediction model such as MLP and RF
    @profile
    @Utilities.time_func
    @Utilities.detect_GPU
    @abstractmethod
    def model_prediction(self) -> None:
        """
        Abstract method. Uses machine learning models to make predictions on an array.

        Notes
        -----
        This method first loads the machine learning model from the file specified by the `Model`
        argument, then uses it to make predictions on the given `Array`. The method prints the
        predicted results and the true values for each sample, as well as the final result for the
        entire array (which is always 0 for this implementation).

        The method also applies the appropriate post-processing for each machine learning model:

        - For a Keras model, the predicted result is converted to a single integer using a threshold
        of 0.5. If the `Array` argument is 3D, the predicted integer is mapped to a different range
        of values.
        - For a scikit-learn model, the predicted result is already a single integer. If the `Array`
        argument is 3D, the predicted integer is mapped to a different range of values.
        """

        pass

# ? Extract Euler numbers from 3D arrays.
class EulerExtractor3D(EulerExtractorUtilities):
    """Extract Euler numbers from 3D arrays.
    
    This class inherits from EulerExtractorUtilities.
    
    Parameters
    ----------
    **kwargs : dict
        Keyword arguments passed to the parent class EulerExtractorUtilities.
    
    Attributes
    ----------
    _Input : str
        The path of the input image file.
    _Output : str
        The path to save the output data.
    _ndim : str
        The number of dimensions of the data (2D, 3D).
    _Folder_data : str
        The path of the folder where the data is saved.
    _Model_name : str
        The name of the model used to extract the Euler number.
    _Columns : int
        The number of columns of the input data.
    
    Methods
    -------
    __repr__()
        Return the string representation of this EulerExtractor3D instance.
    __str__()
        Return a string representation of this EulerExtractor3D instance.
    __del__()
        Destructor that is called when this EulerExtractor3D instance is destroyed.
    data_dic()
        Return a dictionary containing the instance's Inputs, Outputs, ndim, Folder_data, and Model_name values.
    create_json_file()
        Create a JSON file with the given data and save it to the specified file path.
    print_octovoxel_order()
        Static method to print the octovoxel order.
    read_image_with_metadata()
        Load a txt file and convert it into a NumPy array.
    MLP_training_euler_number()
        Train a multi-layer perceptron (MLP).
    model_prediction()
        Uses machine learning models to make predictions on an array.
    get_number_octovoxels(Object)
        Method to obtain the combination of octovoxels in a 3D object.
    MLP_training_octovoxels()
        Train a multi-layer perceptron (MLP) for a 2D image.
    MLP_predict_octovoxels()
        Method to utilize a prediction model such as MLP.
    RF_training_euler_number()
        Train a Random Forest model for a 3D image.
    """

    # * Initializing (Constructor, super)
    def __init__(self, **kwargs) -> None:
        """Constructor method for EulerExtractor3D.
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to the parent class EulerExtractor3D.

        Keyword Arguments
        -----------------
        ndim : str
            n-dimensions.

        """

        super().__init__(**kwargs)
        
        self.ndim = '3D';
        
    # * Class variables
    def __repr__(self):
        """Return the string representation of this EulerExtractor3D instance.
        
        Returns
        -------
        str 
            The string representation of this EulerExtractor3D instance.
        """
        return f'''[{self._Input}, 
                    {self._Output}, 
                    {self._Folder_data}, 
                    {self._Model_name}, 
                    {self._Columns},
                    {self.ndim}]''';

    # * Class description
    def __str__(self):
        """Return a string representation of this EulerExtractor3D instance.
        
        Returns
        -------
        str 
            A string representation of this EulerExtractor3D instance.
        """
        return  f'Utilities class for Euler number 3D extraction.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        """
        Destructor that is called when this EulerExtractor3D instance is destroyed.
        """
        print('Destructor called, Euler number 3D class destroyed.');

    # * Get data from a dic
    def data_dic(self):
        """
        Returns a dictionary containing the instance's Inputs, Outputs, ndim and, Folder_data, Model_name values.

        Returns
        ----------
        dict 
            A dictionary containing the instance's Inputs, Outputs, ndim, Folder_data, Model_name values.
        """
        return {'Inputs': str(self._Input),
                'Outputs': str(self._Output),
                'Folder_data': str(self._Folder_data),
                'Model_name': str(self._Model_name),
                'ndim': str(self.ndim)
                };

    def create_json_file(self):
        """
        Creates a JSON file with the given data and saves it to the specified file path.
        A JSON file containing the instance's Inputs, Outputs, ndim, Folder_data, Model_name values.
        
        Returns
        ----------
        None
        """
        Data = {'Inputs': str(self._Input),
                'Outputs': str(self._Output),
                'Folder_data': str(self._Folder_data),
                'Model_name': str(self._Model_name),
                'ndim': str(self.ndim)
                };

        with open('JSON documents', 'w') as file:
            json.dump(Data, file)

    # ? Static method to print the order of octovoxel letters.
    @staticmethod
    def print_octovoxel_order() -> None:
        """
        Print the order of octovoxel letters.

        Returns
        ----------
        None
        
        """

        # * Define the order of the octovoxel letters.
        Letters = ('a', 'c', 'b', 'd', 'e', 'h', 'f', 'g')
        
        # * Create a 2x2x2 array of zeros to hold the mapping of octovoxel letters to values.
        Array_octovoxel = np.zeros((2, 2, 2))

        # * Define the mapping of octovoxel letters to values using the order specified in `letters`.
        for i, letter in enumerate(Letters):
            Array_octovoxel.flat[i] = i + 1

        # * Print the array mapping the letters to values.
        print('\n')
        print(Array_octovoxel)
        print('\n')

    # ? Method to obtain 1D arrays from a 3D array (np.ndarray)
    @profile
    @Utilities.time_func
    def get_arrays_3D(self, Object: str) -> list[np.ndarray]:
        """
        Obtain 1D arrays from a 3D array (np.ndarray).

        Parameters
        ----------
        Object : str
            The name of the input file.

        Returns
        ----------
        list[np.ndarray]
            A list of 1D arrays.

        Notes
        ----------
        This method takes in a 3D array (np.ndarray) and returns a list of 1D arrays. The input 3D array is read from an
        input file with the name specified by `Object`. The method extracts the values of the surrounding voxels to create
        an array of predictions for each pixel in the 3D array. The resulting list of 1D arrays is returned.

        The method also prints the kernel and prediction arrays for each pixel in the 3D array.

        """

        # Create an empty list to store the resulting 1D arrays
        Arrays = []

        # Define an integer variable to set the number of asterisks to use for display purposes
        Asterisks = 30

       # Call the read_image_with_metadata method to obtain a 3D array
        Array = self.read_image_with_metadata(Object)

        # Create two zero-filled arrays - one with shape (l, l, l) and the other with shape (8)
        #Array_prediction_octov = np.zeros((l, l, l))
        Array_prediction = np.zeros((8))

        # * Initial size of the octovovel.
        l = 2

        # Iterate over each pixel in the 3D array
        for i in range(Array.shape[0] - 1):
            for j in range(Array.shape[1] - 1):
                for k in range(Array.shape[2] - 1):

                    #Array[i:l + i, j:l + j, k:l + k]

                    # Extract the values of the surrounding voxels to create an array of predictions
                    Array_prediction[0] = Array[i + 1][j][k]
                    Array_prediction[1] = Array[i + 1][j][k + 1]

                    Array_prediction[2] = Array[i][j][k]
                    Array_prediction[3] = Array[i][j][k + 1]

                    Array_prediction[4] = Array[i + 1][j + 1][k]
                    Array_prediction[5] = Array[i + 1][j + 1][k + 1]

                    Array_prediction[6] = Array[i][j + 1][k]
                    Array_prediction[7] = Array[i][j + 1][k + 1]
                    print('\n')

                    # Display the kernel and prediction arrays
                    print("*" * Asterisks)
                    Array_prediction_list = Array_prediction.tolist()
                    Array_prediction_list_int = [int(i) for i in Array_prediction_list]
                    print("Kernel array")
                    print(Array[i:l + i, j:l + j, k:l + k])
                    print('\n')
                    print("Prediction array")
                    print(Array_prediction)
                    print('\n')
                    Arrays.append(Array_prediction_list_int)
                    print("*" * Asterisks)
                    print('\n')

        # Display the resulting list of 1D arrays
        for i in range(len(Arrays)):
            print('{} ---- {}'.format(i, Arrays[i]))
        print('\n')
        
        return Arrays

    # ? Load a txt file and convert it into a NumPy array..
    def read_image_with_metadata(Array_file: str) -> np.ndarray:
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

        # * Load the txt file into a NumPy array.
        Array = np.loadtxt(Array_file, delimiter = ',')
        
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

    # ? Train a multi-layer perceptron (MLP)
    def MLP_training_euler_number(self, Opt: str = "ADAM", lr: float = 0.001, Epochs: int = 20) -> Any:
        """
        Train a multi-layer perceptron (MLP)

        Parameters:
        -----------
        Opt: str, optional (default="ADAM")
            Name of the optimizer to use. The available options are:
            "ADAM", "NADAM", "ADAMAX", "ADAGRAD", "ADADELTA", "SGD", "RMSPROP", and "FTRL".

        lr: float, optional (default=0.001)
            Learning rate for the optimizer.

        Epochs: int, optional (default=20)
            Number of training epochs.

        Returns:
        --------
        Hist_data: object
            Object containing the training history.

        Raises:
        -------
        ValueError: if the `_ndim` attribute is not set to "2D" or "3D".

        Examples:
        ---------
        >>> model = EulerExtractorUtilities()
        >>> model.MLP_training_euler_number(Opt="ADAM", lr=0.001, Epochs=20)

        Notes
        -----

        The function takes three optional parameters: Opt, lr, and Epochs. Opt is the name of the optimizer to use, 
        lr is the learning rate, and Epochs is the number of training epochs. 

        The function then compiles the model with the specified optimizer and loss function. It trains the model and stores 
        the training history in Hist_data. Finally, it saves the trained model as an h5 file and the history of the trained model inside a csv file. 
        The function also plots the data loss and accuracy.

        """

        # * Prints the shape of the input and output tensors
        print(self._Input.shape)
        print(self._Output.shape)
        print(self.ndim)

        # * List of optimizer options
        Optimizers = ("ADAM", "NADAM", "ADAMAX", "ADAGRAD", "ADADELTA", "SGD", "RMSPROP", "FTRL");

        if(Opt == Optimizers[0]):
            Opt = Adam(learning_rate = lr);
        elif(Opt == Optimizers[1]):
            Opt = Nadam(learning_rate = lr);
        elif(Opt == Optimizers[2]):
            Opt = Adamax(learning_rate = lr);
        elif(Opt == Optimizers[3]):
            Opt = Adagrad(learning_rate = lr);
        elif(Opt == Optimizers[4]):
            Opt = Adadelta(learning_rate = lr);
        elif(Opt == Optimizers[5]):
            Opt = SGD(learning_rate = lr);
        elif(Opt == Optimizers[6]):
            Opt = RMSprop(learning_rate = lr);
        elif(Opt == Optimizers[7]):
            Opt = Ftrl(learning_rate = lr);
        
        # * If input is 3D, define model architecture with two hidden layers
                
        # *
        Model = Sequential()
        Model.add(Input(shape = self._Input.shape[1],));
        Model.add(Dense(64, activation = "sigmoid"));
        Model.add(Dense(4, activation = 'softmax'));

        # *
        Model.compile(
            optimizer = Opt, 
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
            #sparse_categorical_crossentropy
        )   

        # * Prints that training has begun
        print('\n')
        print("Training...")
        print('\n')

        # * Trains the model and stores the training history
        Hist_data = Model.fit(self._Input, self._Output, epochs = Epochs, verbose = False)

        # * Prints that training has completed
        print('\n')
        print("Model trained")
        print('\n')

        # * Save the trained model as an h5 file
        Model_name_save = '{}.h5'.format(self._Model_name)
        Model_folder_save = os.path.join(self._Folder_data, Model_name_save)

        Model.save(Model_folder_save)

        # * Prints that the model has been saved
        print("Saving model...")
        print('\n')

        # * Save the history of the trained model inside a csv file
        self.create_dataframe_history(self._Columns, self._Folder_data, self._Model_name, Hist_data)

        # * Plot the data loss
        self.plot_data_loss(Hist_data)

        # * Plot the data accuracy
        self.plot_data_accuracy(Hist_data)
        print('\n')

        #return Hist_data

    # ? Method to utilize prediction model such as MLP and RF
    def model_prediction(self, Model: str, Arrays: np.ndarray) -> None:
        """
        Uses machine learning models to make predictions on an array.

        Parameters
        ----------
        Model : str
            The path to the saved model file. Must have extension '.h5' for a Keras model or '.joblib'
            for a scikit-learn model.
        Arrays : np.ndarray
            For a 3D array, the first dimension represents the time steps, the second dimension represents the samples,
            and the third dimension represents the features.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the `Model` argument does not have the required file extension.

        Notes
        -----
        This method first loads the machine learning model from the file specified by the `Model`
        argument, then uses it to make predictions on the given `Array`. The method prints the
        predicted results and the true values for each sample, as well as the final result for the
        entire array (which is always 0 for this implementation).

        """

        # * Initialize the prediction result to zero
        Prediction_result = 0

        # * Add a dimension to the input array to make it suitable for prediction
        #Array = np.expand_dims(Array, axis = 1)

        # * Read multilayer perceptron model
        if Model.endswith('.h5'):
            # * Load the model using the keras load_model() function
            Model_prediction = load_model(Model)

            # * Loop through the input array
            for _, Array in enumerate(Arrays):
                
                print("Prediction!")

                # * Make the prediction and extract the class with highest probability
                Result = np.argmax(Model_prediction.predict([Array]), axis = 1)
                print(Result)
                
                if(Result == 0):
                    Result = 0
                elif(Result == 1):
                    Result = 1
                elif(Result == 2):
                    Result = -1
                elif(Result == 3):
                    Result = -2

                # * Print the prediction and true values
                print('{} -------------- {}'.format(Array, Result))
                print('The result is: {}'.format(Result))
                print('The true value is: {}'.format(Result))
                print('\n')

                # * Print the final prediction result
                Prediction_result += Result
        
        # * Read machine learning model
        elif Model.endswith('.joblib'):
            # * Load the model using the joblib load() function
            Model_prediction = joblib.load(Model)

            # * Loop through the input array
            for _, Array in enumerate(Arrays):
                
                print("Prediction!")

                # * Make the prediction
                Result = Model_prediction.predict([Array])
                print(Result)

                if(Result == 0):
                    Result = 0
                elif(Result == 1):
                    Result = 1
                elif(Result == 2):
                    Result = -1
                elif(Result == 3):
                    Result = -2

                # * Print the prediction and true values
                print('{} -------------- {}'.format(Array, Result))
                print('The result is: {}'.format(Result))
                print('The true value is: {}'.format(Result))
                print('\n')

                # * Print the final prediction result
                Prediction_result += Result

            else:
                pass

        # * Print the final prediction result
        print('\n')
        print('Euler: {}'.format(Prediction_result))
        print('\n')

        return Prediction_result
    
    # ? Method to obtain the combination of octovoxel in a 3D object
    @Utilities.time_func
    def get_number_octovoxels(self, Object: str = None) -> np.ndarray:
        """
        Method to obtain the combination of octovoxel in a 3D object

        Parameters
        ----------
        Object : str, optional
            Description of the object. (Default is None)

        Returns
        -------
        np.ndarray
            1D array containing the count of each octovoxel combination.

        Notes
        -----
        This method reads a 3D array using the `read_image_with_metadata` method,
        generates the octovoxel truth table using the `BinaryConversion` object,
        and returns a 1D array containing the count of each octovoxel combination.

        Examples
        --------
        >>> object = '*.txt'
        >>> octovoxels = get_number_octovoxels(*.txt)
        >>> print(octovoxels)
        [0, 1, 2, ..., 0, 0, 0]
        """
        # Number of iterations to perform
        Combinations_3D = 256

        # * Saving the function into a varible array.
        Array = self.read_image_with_metadata(Object);

        # * Create a BinaryConversion object to generate the octovoxel truth table
        BinaryObject = BinaryConversion(Combinations_3D, self.ndim)
        Qs = BinaryObject.decimal_to_binary_list();

        # * Initialize an array to store the count of each octovoxel combination
        Qs_value = np.zeros((Combinations_3D), dtype = 'int');
        
        # * Initial size of the octovovel.
        l = 2

        # * From each combination of the truth table, subtract the number of times each octovovel combination is presented.
        for i in range(Array.shape[0] - 1):
            for j in range(Array.shape[1] - 1):
                for k in range(Array.shape[2] - 1):

                    for Index in range(len(Qs)):
                        
                        # * Compare arrays with different dimensions.
                        if(np.array_equal(np.array(Array[i:l + i, j:l + j, k:l + k]), np.array(Qs[Index]))):
                            Qs_value[Index] += 1;
                            #print('Q{}_value: {}'.format(Index, Qs_value[Index]));

                    # * print the difference between arrays
                    #print(Qs_value)
                    #print('\n')

        print(Qs_value);

        #List_string = ''

        """
        for i in range(256):

            List_string = List_string + str(Qs_value[i]) + ', ';
            if(i == 255):
                List_string = List_string + str(Qs_value[i]);

        print('[{}]'.format(List_string))
        """
        
        return Qs_value
    
    # ? Method to to train a MLP for a 2D image
    @Utilities.time_func
    @Utilities.detect_GPU
    def MLP_training_octovoxels(self, Dataframe_:pd.DataFrame, Opt: str = "ADAM", lr: float = 0.000001, Epochs: int = 20) -> Any:
        """
        Train a multi-layer perceptron (MLP) for a 2D image.

        Parameters
        ----------
        Dataframe_: str or file-like object
            File containing a pandas dataframe with the input data and labels.
        Opt: str, optional
            Name of the optimizer to be used (default is "ADAM"). 
            Possible values: "ADAM", "NADAM", "ADAMAX", "ADAGRAD", "ADADELTA", "SGD", "RMSPROP", "FTRL".
        lr: float, optional
            Learning rate for the optimizer (default is 0.000001).
        Epochs: int, optional
            Number of training epochs (default is 20).

        Returns
        -------
        Hist_data: tf.keras.callbacks.History
            Training history.

        Raises
        ------
        ValueError
            If the given optimizer name is not valid.

        Examples
        --------
        >>> MLP_training_getoctovoxels('data.csv', Opt='ADAM', lr=0.0001, Epochs=50)

        Notes
        -----
        This method trains an MLP using the input data and labels provided in a pandas dataframe. 
        The model is trained using the specified optimizer and learning rate for the given number of epochs.
        The training history is returned and saved in a file, and plots of the loss and accuracy are generated.
        The trained model is also saved in a file with the name specified by the user.

        """
        
        # * Check if the optimizer name is valid and convert it to uppercase.
        Opt_uppercase = Opt.upper()

        # * Possible optimizers to use.
        Optimizers = ("ADAM", "NADAM", "ADAMAX", "ADAGRAD", "ADADELTA", "SGD", "RMSPROP", "FTRL");

        # * Read the data from the given file into a pandas dataframe.
        Dataframe = pd.read_csv(Dataframe_);

        # * Extract the input data and labels from the dataframe.
        X = Dataframe.iloc[:, 1:257].values;
        Y = Dataframe.iloc[:, -1].values;

        # * Reshape the labels array to have the same number of dimensions as the input data.
        Y = np.expand_dims(Y, axis = 1);

        # * Define the architecture of the MLP.
        Model = tf.keras.Sequential();
        Model.add(tf.keras.layers.Input(shape = X.shape[1],));
        Model.add(tf.keras.layers.Dense(units = 1200, activation = 'relu'));
        Model.add(tf.keras.layers.Dense(units = 1));
        
        # * Select the optimizer to be used.
        if(Opt_uppercase == Optimizers[0]):
            Opt_uppercase = Adam(learning_rate = lr);
        elif(Opt_uppercase == Optimizers[1]):
            Opt_uppercase = Nadam(learning_rate = lr);
        elif(Opt_uppercase == Optimizers[2]):
            Opt_uppercase = Adamax(learning_rate = lr);
        elif(Opt_uppercase == Optimizers[3]):
            Opt_uppercase = Adagrad(learning_rate = lr);
        elif(Opt_uppercase == Optimizers[4]):
            Opt_uppercase = Adadelta(learning_rate = lr);
        elif(Opt_uppercase == Optimizers[5]):
            Opt_uppercase = SGD(learning_rate = lr);
        elif(Opt_uppercase == Optimizers[6]):
            Opt_uppercase = RMSprop(learning_rate = lr);
        elif(Opt_uppercase == Optimizers[7]):
            Opt_uppercase = Ftrl(learning_rate = lr);

        Model.compile(
            optimizer = Opt_uppercase, 
            loss = 'mean_squared_error',
            metrics = ['accuracy']
        )

        # * Prints that training has begun
        print('\n')
        print("Training...")
        print('\n')
        
        # * Trains the model and stores the training history
        Hist_data = Model.fit(X, Y, batch_size = 8, epochs = Epochs)

        # * Prints that training has completed
        print('\n')
        print("Model trained")
        print('\n')

        # * Save the trained model as an h5 file
        Model_name_save = '{}.h5'.format(self._Model_name)
        Model_folder_save = os.path.join(self._Folder_data, Model_name_save)
        Model.save(Model_folder_save)

        # * Prints that the model has been saved
        print("Saving model...")
        print('\n')

        # * Save the history of the trained model inside a csv file
        self.create_dataframe_history(self._Columns, self._Folder_data, self._Model_name, Hist_data)

        # * Plot the data loss
        self.plot_data_loss(Hist_data)
        
        # * Plot the data accuracy
        self.plot_data_accuracy(Hist_data)
        print('\n')

        return Hist_data
    
    # ? Method to utilize prediction model such as MLP
    @Utilities.time_func
    @Utilities.detect_GPU
    def MLP_predict_octovoxels(self, Model: str, Object: str):
        """
        Method to utilize a prediction model such as MLP.

        Parameters
        ----------
        Model : str
            The path to the saved model in h5 format.
        Object : str
            The path to the object to predict.

        Returns
        -------
        numpy.ndarray
            An array with the predicted values.

        """

        try:
            
            # * Check if the model is a h5 file
            if Model.endswith('.h5'):
            
                # * Convert the string representation of the octovoxel array to a numpy array
                Array = self.get_number_octovoxels(Object)

                # * Expand the dimensions of the array to match the expected input shape of the model
                Array = np.expand_dims(Array, axis = 0)

                # * Load the model from the specified file
                Model_prediction = load_model(Model)

                # * Make a prediction on the input array
                Result = Model_prediction.predict(Array)
                
                print(Result)

                return Result

            else:
                print('The file is not h5 ❌'.format()) #! Alert

        except OSError as err:
            print('Error: {} ❌'.format(err)) #! Alert
        
    # ? Method to to train a RF for a 3D image
    @profile
    @Utilities.time_func
    @Utilities.detect_GPU
    def RF_training_euler_number(self) -> None:
        """
        Train a Random Forest model for a 3D image.

        This method uses the input and output data stored in the object to train a Random Forest model
        with the following parameters:
        - criterion = 'gini'
        - n_estimators = 10
        - random_state = 2
        - n_jobs = 10

        After training the model, it is saved as a .joblib file in the data folder. The method then
        predicts the output for the input data and calculates the accuracy of the model. Finally, it
        prints the prediction output, the original output, and the accuracy of the model.

        Returns:
        None
        """

        # * Print the shape of the input and output data
        print(self._Input.shape)
        print(self._Output.shape)

        # Create a Random Forest model
        Model_RF = RandomForestClassifier(  criterion = 'gini',
                                            n_estimators = 10,
                                            random_state = 2,
                                            n_jobs = 10)

        # * Print status message
        print('\n')
        print("Training...")
        print('\n')

        # * Train the Random Forest model
        Model_RF.fit(self._Input, self._Output)

        # * Print status message
        print('\n')
        print("Model trained")
        print('\n')

        # * Save the trained model using joblib
        Model_name_save = '{}.joblib'.format(self._Model_name)
        Model_folder_save = os.path.join(self._Folder_data, Model_name_save)
        joblib.dump(Model_RF, Model_folder_save)

        # * Make predictions on the input data
        pred_Input = Model_RF.predict(self._Input)

        # * Print prediction and output labels
        print('Prediction output')
        print(pred_Input)
        print('\n')
        print('Original output')
        print(self._Output)
        print('\n')

        # * Calculate accuracy of the model
        Accuracy = accuracy_score(self._Output, pred_Input)

        # * Print accuracy of the model
        print('Result: {}'.format(Accuracy))
        print('\n')

        # * Print status message
        print("Saving model...")
        print('\n')

        print('\n')

# ?
class EulerExtractor2D(EulerExtractorUtilities):
    """Extract Euler numbers from 2D arrays.

    This class inherits from EulerExtractorUtilities.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments passed to the parent class EulerExtractorUtilities.

    Attributes
    ----------
    _Input : str
        The path of the input image file.
    _Output : str
        The path to save the output data.
    _ndim : str
        The number of dimensions of the data (2D, 3D).
    _Folder_data : str
        The path of the folder where the data is saved.
    _Model_name : str
        The name of the model used to extract the Euler number.
    _Columns : int
        The number of columns of the input data.

    Methods
    -------
    __repr__()
        Return the string representation of this EulerExtractor2D instance.
    __str__()
        Return a string representation of this EulerExtractor2D instance.
    __del__()
        Destructor that is called when this EulerExtractor2D instance is destroyed.
    data_dic() -> dict
        Returns a dictionary containing the instance's Inputs, Outputs, ndim and, Folder_data, Model_name values.
    create_json_file() -> None
        Creates a JSON file with the given data and saves it to the specified file path.
    read_image_with_metadata()
        Load a txt file and convert it into a NumPy array.
    MLP_training_euler_number()
        Train a multi-layer perceptron (MLP).
    model_prediction()
        Uses machine learning models to make predictions on an array.
    get_number_pixeles_2D(Object: str) -> list[np.ndarray]
        Method to obtain the combination of bit-quads in a image
    connectivity_4_prediction_2D(Arrays) -> None
        Method to utilize connectivity 4 arrays to search for euler's number.
    connectivity_8_prediction_2D(Arrays) -> None
        Method to utilize connectivity 8 arrays to search for euler's number.

    """
    
    # * Initializing (Constructor, super)
    def __init__(self, **kwargs) -> None:
        """Constructor method for EulerExtractor2D.
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to the parent class EulerExtractor2D.

        Keyword Arguments
        -----------------
        ndim : str
            n-dimensions.

        """

        super().__init__(**kwargs)
        
        self.ndim = '2D';
    
    # * Class variables
    def __repr__(self):
        """
        Returns the string representation of this EulerExtractor2D instance.

        Returns
        ----------
        str 
            The string representation of this EulerExtractor2D instance.
        """
        return f'''[{self._Input}, 
                    {self._Output}, 
                    {self._Folder_data}, 
                    {self._Model_name},
                    {self._Columns},
                    {self.ndim}]''';

    # * Class description
    def __str__(self):
        """
        Returns a string representation of this EulerExtractor2D instance.

        Returns
        ----------
        str 
            A string representation of this EulerExtractor2D instance.
        """
        return  f'Class for Euler number 2D extraction.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        """
        Destructor that is called when this EulerExtractor2D instance is destroyed.
        """
        print('Destructor called, Euler number 2D class destroyed.');

    # * Get data from a dic
    def data_dic(self):
        """
        Returns a dictionary containing the instance's Inputs, Outputs, Folder_data, Model_name, and ndim values.

        Returns
        ----------
        dict 
            A dictionary containing the instance's Inputs, Outputs, ndim, Folder_data, Model_name, and ndim values.
        """
        return {'Inputs': str(self._Input),
                'Outputs': str(self._Output),
                'Folder_data': str(self._Folder_data),
                'Model_name': str(self._Model_name),
                'ndim': str(self.ndim)
                };

    def create_json_file(self):
        """
        Creates a JSON file with the given data and saves it to the specified file path.
        A JSON file containing the instance's Inputs, Outputs, ndim, Folder_data, Model_name values.
        
        Returns
        ----------
        None
        """
        Data = {'Inputs': str(self._Input),
                'Outputs': str(self._Output),
                'Folder_data': str(self._Folder_data),
                'Model_name': str(self._Model_name),
                'ndim': str(self.ndim)
                };

        with open('JSON documents', 'w') as file:
            json.dump(Data, file)

    # ? Load a txt file and convert it into a NumPy array..
    def read_image_with_metadata(self, Array_file: str) -> np.ndarray:
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

        # * Load the txt file into a NumPy array.
        Array = np.loadtxt(Array_file, delimiter = ',')
        
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
    
    # ? Train a multi-layer perceptron (MLP)
    def MLP_training_euler_number(self, Opt: str = "ADAM", lr: float = 0.001, Epochs: int = 20) -> Any:
        """
        Train a multi-layer perceptron (MLP)

        Parameters:
        -----------
        Opt: str, optional (default="ADAM")
            Name of the optimizer to use. The available options are:
            "ADAM", "NADAM", "ADAMAX", "ADAGRAD", "ADADELTA", "SGD", "RMSPROP", and "FTRL".

        lr: float, optional (default=0.001)
            Learning rate for the optimizer.

        Epochs: int, optional (default=20)
            Number of training epochs.

        Returns:
        --------
        Hist_data: object
            Object containing the training history.

        Raises:
        -------
        ValueError: if the `_ndim` attribute is not set to "2D" or "3D".

        Examples:
        ---------
        >>> model = EulerExtractorUtilities()
        >>> model.MLP_training_euler_number(Opt="ADAM", lr=0.001, Epochs=20)
        """

        # * Prints the shape of the input and output tensors
        print(self._Input.shape)
        print(self._Output.shape)
        print(self._ndim)

        # * List of optimizer options
        Optimizers = ("ADAM", "NADAM", "ADAMAX", "ADAGRAD", "ADADELTA", "SGD", "RMSPROP", "FTRL");

        if(Opt == Optimizers[0]):
            Opt = Adam(learning_rate = lr);
        elif(Opt == Optimizers[1]):
            Opt = Nadam(learning_rate = lr);
        elif(Opt == Optimizers[2]):
            Opt = Adamax(learning_rate = lr);
        elif(Opt == Optimizers[3]):
            Opt = Adagrad(learning_rate = lr);
        elif(Opt == Optimizers[4]):
            Opt = Adadelta(learning_rate = lr);
        elif(Opt == Optimizers[5]):
            Opt = SGD(learning_rate = lr);
        elif(Opt == Optimizers[6]):
            Opt = RMSprop(learning_rate = lr);
        elif(Opt == Optimizers[7]):
            Opt = Ftrl(learning_rate = lr);
        
        # * Model layer for 2D
        Model = Sequential()
        Model.add(Input(shape = self._Input.shape[1],));
        Model.add(Dense(9, activation = "sigmoid"))
        Model.add(Dense(1, activation = 'tanh'))

        Model.compile(
            optimizer = Opt, 
            loss = 'mean_squared_error',
            metrics = ['accuracy']
        )

        # * Prints that training has begun
        print('\n')
        print("Training...")
        print('\n')

        # * Trains the model and stores the training history
        Hist_data = Model.fit(self._Input, self._Output, epochs = Epochs, verbose = False)

        # * Prints that training has completed
        print('\n')
        print("Model trained")
        print('\n')

        # * Save the trained model as an h5 file
        Model_name_save = '{}.h5'.format(self._Model_name)
        Model_folder_save = os.path.join(self._Folder_data, Model_name_save)

        Model.save(Model_folder_save)

        # * Prints that the model has been saved
        print("Saving model...")
        print('\n')

        # * Save the history of the trained model inside a csv file
        self.create_dataframe_history(self._Columns, self._Folder_data, self._Model_name, Hist_data)

        # * Plot the data loss
        self.plot_data_loss(Hist_data)

        # * Plot the data accuracy
        self.plot_data_accuracy(Hist_data)
        print('\n')

        #return Hist_data

    # ? Method to utilize prediction model such as MLP and RF
    def model_prediction(self, Model: str, Arrays: np.ndarray) -> None:
        """
        Uses machine learning models to make predictions on an array.

        Parameters
        ----------
        Model : str
            The path to the saved model file. Must have extension '.h5' for a Keras model or '.joblib'
            for a scikit-learn model.
        Array : np.ndarray
            A 2D or 3D array of features to make predictions on. For a 2D array, the first dimension
            represents the samples and the second dimension represents the features. For a 3D array,
            the first dimension represents the time steps, the second dimension represents the samples,
            and the third dimension represents the features.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the `Model` argument does not have the required file extension.

        Notes
        -----
        This method first loads the machine learning model from the file specified by the `Model`
        argument, then uses it to make predictions on the given `Array`. The method prints the
        predicted results and the true values for each sample, as well as the final result for the
        entire array (which is always 0 for this implementation).
        """

        # * Initialize the prediction result to zero
        Prediction_result = 0

        # * Add a dimension to the input array to make it suitable for prediction
        #Array = np.expand_dims(Array, axis = 1)

        # * Read multilayer perceptron model
        if Model.endswith('.h5'):
            # * Load the model using the keras load_model() function
            Model_prediction = load_model(Model)

            # * Loop through the input array
            for _, Array in enumerate(Arrays):
                
                print("Prediction!")

                # * Make the prediction and extract the class with highest probability
                Result = np.argmax(Model_prediction.predict([Array]), axis = 1)
                print(Result)
                
                if(Result > 0.5):
                    Result = 1
                elif(Result < 0.5 and Result > -0.5):
                    Result = 0
                elif(Result < -0.5):
                    Result = -1

                # * Print the prediction and true values
                print('{} -------------- {}'.format(Array, Result))
                print('The result is: {}'.format(Result))
                print('The true value is: {}'.format(Result))
                print('\n')

                # * Print the final prediction result
                Prediction_result += Result
        
        # * Read machine learning model
        elif Model.endswith('.joblib'):
            # * Load the model using the joblib load() function
            Model_prediction = joblib.load(Model)

            # * Loop through the input array
            for _, Array in enumerate(Arrays):
                
                print("Prediction!")

                # * Make the prediction
                Result = Model_prediction.predict([Array])
                print(Result)
    
                if(Result > 0.5):
                    Result = 1
                elif(Result < 0.5 and Result > -0.5):
                    Result = 0
                elif(Result < -0.5):
                    Result = -1

                # * Print the prediction and true values
                print('{} -------------- {}'.format(Array, Result))
                print('The result is: {}'.format(Result))
                print('The true value is: {}'.format(Result))
                print('\n')

                # * Print the final prediction result
                Prediction_result += Result

            else:
                pass

        # * Print the final prediction result
        print('\n')
        print('Euler: {}'.format(Prediction_result))
        print('\n')

        return Prediction_result

    # ? Method to obtain the combination of octovoxel in a 3D object
    @Utilities.time_func
    def get_number_pixeles_2D(self, Object: str) -> list[np.ndarray]:
        """
        Method to obtain the combination of bit-quads in a image

        Parameters
        ----------
        Object : str
            The name of the input file.

        Returns
        -------
        numpy.ndarray
            Method to obtain the combination of bit-quads in a image

        """

        # * Number of iterations to perform
        Combinations_2D = 16

        # * Saving the function into a varible array.
        Array = self.read_image_with_metadata(Object, ndim = self.ndim);

        # * Extract the truth table.
        BinaryObject = BinaryConversion(Combinations_2D, self.ndim)
        Qs = BinaryObject.decimal_to_binary_list();

        # * Create a empty numpy array.
        Qs_value = np.zeros((Combinations_2D), dtype = 'int');
        
        # * Initial size of the octovovel.
        l = 2

        # * From each combination of the truth table, subtract the number of times each octovovel combination is presented.
        for i in range(Array.shape[0] - 1):
            for j in range(Array.shape[1] - 1):

                for Index in range(len(Qs)):
                    
                    # * Compare arrays with different dimensions.
                    if(np.array_equal(np.array(Array[i:l + i, j:l + j]), np.array(Qs[Index]))):
                        Qs_value[Index] += 1;
                        #print('Q{}_value: {}'.format(Index, Qs_value[Index]));

                # * print the difference between arrays
                #print(Qs_value)
                #print('\n')

        print(Qs_value);

        #List_string = ''

        """
        for i in range(256):

            List_string = List_string + str(Qs_value[i]) + ', ';
            if(i == 255):
                List_string = List_string + str(Qs_value[i]) + ', ';

        print('[{}]'.format(List_string))
        """
        
        return Qs_value

    # ? Method to utilize connectivity 4 arrays to search for euler's number
    @profile
    @Utilities.time_func
    @Utilities.detect_GPU
    def connectivity_4_prediction_2D(self, Arrays)-> None:
        """
        Utilize connectivity 4 arrays to search for Euler's number.

        Parameters
        ----------
        Arrays : List[numpy.ndarray]
            A list of 2D binary arrays.

        Returns
        -------
        None

        Notes
        -----

        This algorithm works by checking if a cell in a 2D array has any neighboring cells 
        (in any of the cardinal directions: up, down, left, or right). In the case of a 2D array, 
        the algorithm looks for patterns of "1"s and "0"s in a 2x2 neighborhood around each cell. 
        The connectivity_4_prediction_2D method looks for a specific set of patterns 
        (defined by the Connectivity_4_first_array, Connectivity_4_second_array, and Connectivity_4_third_array arrays) 
        to calculate the result.

        More information can be found in the following paper: 
        - Finding the Optimal Bit-Quad Patterns for Computing the Euler Number of 2D Binary Images Using Simulated Annealing.
            Authors : Wilfrido Gomez-Flores1, Humberto Sossa, and Fernando Arce.


        """

        Connectivity_4_first_array = np.array([1, 0, 0, 0], dtype = 'int')
        Connectivity_4_second_array = np.array([1, 1, 1, 0], dtype = 'int')
        Connectivity_4_third_array = np.array([1, 0, 0, 1], dtype = 'int')

        # * Initialize result
        Result_connected_4 = 0

        # * Loop over input arrays
        for _, Array in enumerate(Arrays):

            print(np.array(Array))

            if(np.array_equal(np.array(Array), Connectivity_4_first_array)):
                Result_connected_4 += 1
                print('Connectivity 4: {}'.format(Result_connected_4))

            if(np.array_equal(np.array(Array), Connectivity_4_second_array)):
                Result_connected_4 -= 1
                print('Connectivity 4: {}'.format(Result_connected_4))

            if(np.array_equal(np.array(Array), Connectivity_4_third_array)):
                Result_connected_4 += 1
                print('Connectivity 4: {}'.format(Result_connected_4))
            print('\n')

        print('\n')
        print('Connectivity 4: {}'.format(Result_connected_4))
        print('\n')

    # ? Method to utilize connectivity 8 arrays to search for euler's number
    @profile
    @Utilities.time_func
    @Utilities.detect_GPU
    def connectivity_8_prediction_2D(self, Arrays) -> None:
        """
        Utilize connectivity 4 arrays to search for Euler's number.

        Parameters
        ----------
        Arrays : List[numpy.ndarray]
            A list of 2D binary arrays.

        Returns
        -------
        None

        Notes
        -----

        This algorithm works by checking if a cell in a 2D array has any neighboring cells 
        (in any of the cardinal directions: up, down, left, or right). In the case of a 2D array, 
        the algorithm looks for patterns of "1"s and "0"s in a 2x2 neighborhood around each cell. 
        The connectivity_8_prediction_2D method looks for a specific set of patterns 
        (defined by the Connectivity_8_first_array, Connectivity_8_second_array, and Connectivity_4_third_array arrays) 
        to calculate the result.

        More information can be found in the following paper: 
        - Finding the Optimal Bit-Quad Patterns for Computing the Euler Number of 2D Binary Images Using Simulated Annealing.
            Authors : Wilfrido Gomez-Flores1, Humberto Sossa, and Fernando Arce.

        """

        Connectivity_8_first_array = np.array([1, 0, 0, 0], dtype = 'int')
        Connectivity_8_second_array = np.array([1, 1, 1, 0], dtype = 'int')
        Connectivity_8_third_array = np.array([0, 1, 1, 0], dtype = 'int')

        # * Initialize result
        Result_connected_8 = 0

        # * Loop over input arrays
        for _, Array in enumerate(Arrays):

            print(np.array(Array))

            if(np.array_equal(np.array(Array), Connectivity_8_first_array)):
                Result_connected_8 += 1
                print('Connectivity 8: {}'.format(Result_connected_8))

            if(np.array_equal(np.array(Array), Connectivity_8_second_array)):
                Result_connected_8 -= 1
                print('Connectivity 8: {}'.format(Result_connected_8))

            if(np.array_equal(np.array(Array), Connectivity_8_third_array)):
                Result_connected_8 += 1
                print('Connectivity 8: {}'.format(Result_connected_8))

        print('\n')
        print('Connectivity 8: {}'.format(Result_connected_8))
        print('\n')