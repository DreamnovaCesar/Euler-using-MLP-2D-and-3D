
from Article_Euler_Number_Libraries import *
from Article_Euler_Number_Utilities import Utilities

from Article_Euler_Number_3D_General import *

# ?
class OctovoxelEulerANN(Utilities):
    """
    Utilities inheritance

    '''''

    Methods:
        data_dic(): description

        create_dataframe_history(): description
        
        plot_data_loss(): description

        plot_data_accuracy(): description

    """

    # * Initializing (Constructor)
    def __init__(self, **kwargs) -> None:
        """
        Keyword Args:
            --- input (np.ndarray): description 
            --- output (np.ndarray): description
            folder (str): description
            FD (bool):description
            MN (str):description
            epochs (int):description
        """

        # * General parameters
        #self._Input = kwargs.get('input', None)
        #self._Output = kwargs.get('output', None)
        #self.Object = kwargs.get('object', None)

        # *
        self._Folder = kwargs.get('folder', None)
        self._Model_name = kwargs.get('MN', None)
        self._Epochs = kwargs.get('epochs', None)

        self._Columns = ["Loss", "Accuracy"]

        if(isinstance(self._Epochs, str)):
            self._Epochs = int(self._Epochs)

    # * Class variables
    def __repr__(self):
            return f'''[{self._Folder},  
                        {self._Epochs}, 
                        {self._Model_name},
                        {self._Columns}]''';

    # * Class description
    def __str__(self):
        return  f'.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, ... class destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'folder': str(self._Folder),
                'folder': str(self._Model_name),
                'epochs': str(self._Epochs),
                };

    # * _Folder attribute
    @property
    def _Folder_property(self):
        return self._Folder

    @_Folder_property.setter
    def _Folder_property(self, New_value):
        print("Changing folder...")
        self._Folder = New_value
    
    @_Folder_property.deleter
    def _Folder_property(self):
        print("Deleting folder...")
        del self._Folder

    # * _Model_name attribute
    @property
    def _Model_name_property(self):
        return self._Model_name

    @_Model_name_property.setter
    def _Model_name_property(self, New_value):
        print("Changing model name...")
        self._Model_name = New_value
    
    @_Model_name_property.deleter
    def _Model_name_property(self):
        print("Deleting model name...")
        del self._Model_name

    # * _Epochs attribute
    @property
    def _Epochs_property(self):
        return self._Epochs

    @_Epochs_property.setter
    def _Epochs_property(self, New_value):
        print("Changing epochs...")
        self._Epochs = New_value
    
    @_Epochs_property.deleter
    def _Epochs_property(self):
        print("Deleting epochs...")
        del self._Epochs

    # * _Columns kwargs
    @property
    def _Columns_property(self):
        return self._Columns

    @_Columns_property.setter
    def _Columns_property(self, New_value):
        print("Changing columns names...")
        self._Columns = New_value
    
    @_Columns_property.deleter
    def _Columns_property(self):
        print("Deleting columns names...")
        del self._Columns

    # ? Static method to create dataframe from history
    @staticmethod
    @Utilities.time_func
    def create_dataframe_history(Column_names: Any, Folder_save: str, CSV_name: str, Hist_data: Any) -> None: 
        """
        Method to plot loss

        Args:
            Column_names (Any): description
            Folder_save (str): description
            CSV_name (str): description
            Hist_data (Any): description
        """

        # * Lists
        #Column_names = ['Folder', 'New Folder', 'Animal', 'Label']
        
        # *
        Dataframe_created = pd.DataFrame(columns = Column_names)

        # *
        Accuracy = Hist_data.history["accuracy"]
        Loss = Hist_data.history["loss"]
        
        History_data = zip(Loss, Accuracy)

        # *
        for i, (l, a) in enumerate(History_data):
            Dataframe_created.loc[len(Dataframe_created.index)] = [l, a]

        # *
        Dataframe_name = "Dataframe_{}_Loss_And_Accuracy.csv".format(CSV_name)
        Dataframe_folder = os.path.join(Folder_save, Dataframe_name)

        # *
        Dataframe_created.to_csv(Dataframe_folder)

    # ? Method to plot loss
    @Utilities.time_func
    def plot_data_loss(self, Hist_data: Any) -> None:
        """
        Method to plot loss

        Args:
            Hist_data (Any): description
        """

        Maxloss = max(Hist_data.history["loss"])

        plt.figure(figsize = (8, 8))
        plt.title('Training loss')
        plt.xlabel ("# Epoch")
        plt.ylabel ("# Loss")
        plt.ylim([0, Maxloss])
        plt.plot(Hist_data.history["loss"])
        #plt.show()

        Figure_name = "Figure_Loss_{}.png".format(self._Model_name)
        Figure_name_folder = os.path.join(self._Folder, Figure_name)

        plt.savefig(Figure_name_folder)
        plt.close()

    # ? Method to plot accuracy
    @Utilities.time_func
    def plot_data_accuracy(self, Hist_data: Any) -> None:
        """
        Method to plot accuracy

        Args:
            Hist_data (Any): description
        """
        plt.figure(figsize = (8, 8))
        plt.title('Training accuracy')
        plt.xlabel ("# Epoch")
        plt.ylabel ("# Acuracy")
        plt.ylim([0, 1])
        plt.plot(Hist_data.history["accuracy"])
        #plt.show()

        Figure_name = "Figure_Accuracy_{}.png".format(self._Model_name)
        Figure_name_folder = os.path.join(self._Folder, Figure_name)

        plt.savefig(Figure_name_folder)
        plt.close()

    # ? Read medata from a 3D object
    @staticmethod
    @Utilities.time_func
    def read_image_with_metadata_3D(Array_file: str) -> np.ndarray:
        """
        Static method to load txt and convert it into tensors

        Args:
            Array_file (str): description
        """
        # *
        Array = np.loadtxt(Array_file, delimiter = ',')
        
        # *
        Height = Array.shape[0]/Array.shape[1]
        Array_new = Array.reshape(int(Height), int(Array.shape[1]), int(Array.shape[1]))

        # *
        Array_new = Array_new.astype(int)

        print('\n')
        print('Array obtained')
        print('\n')
        print(Array_new)
        print('\n')
        print('Number of channels: {}'.format(Array_new.shape[0]))
        print('\n')
        print('Number of rows: {}'.format(Array_new.shape[1]))
        print('\n')
        print('Number of columns: {}'.format(Array_new.shape[2]))
        print('\n')

        return Array_new

    # ? Method to obtain the combination of octovoxel in a 3D object
    @Utilities.time_func
    def get_octovoxel_3D(self, Object: str) -> list[np.ndarray]:
        """
        Method to obtain 1D arrays from a 3D array

        Args:
            Object (str): description

        """

        l = 2

        # * Saving the function into a varible array
        Array_new = self.read_image_with_metadata_3D(Object);

        # * Extract the truth table
        Qs = table_binary_multi_256(256);

        # * Create a empty numpy array
        Qs_value = np.zeros((256), dtype = 'int');

        # * From each combination of the truth table, subtract the number of times each octovovel combination is presented.
        for i in range(Array_new.shape[0] - 1):
            for j in range(Array_new.shape[1] - 1):
                for k in range(Array_new.shape[2] - 1):

                    for Index in range(len(Qs)):
                        
                        # * Compare arrays with different dimensions.
                        if(np.array_equal(np.array(Array_new[i:l + i, j:l + j, k:l + k]), np.array(Qs[Index]))):
                            Qs_value[Index] += 1;
                            print('Q{}_value: {}'.format(Index, Qs_value[Index]));

                    # * print the difference between arrays
                    print(Qs_value)
                    print('\n')
        
        List_string = ''

        for i in range(256):

            List_string = List_string + str(Qs_value[i]) + ', ';
            if(i == 255):
                List_string = List_string + str(Qs_value[i]) + ', ';

        print('[{}]'.format(List_string))

        return Qs_value

    # ? Method to to train a MLP for a 2D image
    @Utilities.time_func
    @Utilities.detect_GPU
    def MLP_octovoxel_training_3D(self, Dataframe_:pd.DataFrame) -> Any:
        """
        Method to to train a MLP for a 3D image

        """

        # *
        Dataframe = pd.read_csv(Dataframe_)

        # * Return a dataframe with only the data without the labels
        X = Dataframe.iloc[:, 1:257].values

        # * Return a dataframe with only the labels
        Y = Dataframe.iloc[:, -1].values

        #X = np.expand_dims(X, axis = 1)
        Y = np.expand_dims(Y, axis = 1)

        Model = tf.keras.Sequential()
        Model.add(tf.keras.layers.Input(shape = X.shape[1],))
        Model.add(tf.keras.layers.Dense(units = 1200, activation = 'relu', kernel_initializer = 'normal'))
        Model.add(tf.keras.layers.Dense(units = 1))

        Opt = Adam(learning_rate = 0.000001)

        Model.compile(
            optimizer = Opt, 
            loss = 'mean_squared_error',
            metrics = ['accuracy']
        )

        print('\n')
        print("Training...")
        print('\n')
        
        # *
        Hist_data = Model.fit(X, Y, batch_size = 8, epochs = self._Epochs)

        print('\n')
        print("Model trained")
        print('\n')

        # *
        Model_name_save = '{}.h5'.format(self._Model_name)
        Model_folder_save = os.path.join(self._Folder, Model_name_save)
        Model.save(Model_folder_save)

        print("Saving model...")
        print('\n')

        # *
        #Loss = Hist_data.history['loss']
        #Accuracy = Hist_data.history['accuracy']

        # *
        self.create_dataframe_history(self._Columns, self._Folder, self._Model_name, Hist_data)

        # *
        self.plot_data_loss(Hist_data)
        
        # *
        self.plot_data_accuracy(Hist_data)
        print('\n')

        return Hist_data

        # ? Method to utilize prediction model such as MLP
    @Utilities.time_func
    @Utilities.detect_GPU
    def MLP_octovoxel_prediction_3D(self, Model: str, Object: str):
        """
        Method to utilize prediction model such as MLP

        """

        # *
        Array = self.get_octovoxel_3D(Object)

        # *
        Array = np.expand_dims(Array, axis = 0)

        # * Read multilayer perceptron model
        if Model.endswith('.h5'):
            Model_prediction = load_model(Model)

        # *
        Model_prediction = load_model(Model)
        Result = Model_prediction.predict(Array)

        print(Result)

        return Result
