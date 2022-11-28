from Article_Euler_Number_Libraries import *
from Article_Euler_Number_Utilities import Utilities

# ?
class EulerNumberML(Utilities):
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
            input (np.ndarray): description 
            output (np.ndarray): description
            folder (str): description
            FD (bool):description
            MN (str):description
            epochs (int):description
        """

        # * General parameters
        self._Input = kwargs.get('input', None)
        self._Output = kwargs.get('output', None)
        #self.Object = kwargs.get('object', None)

        # *
        self._Folder = kwargs.get('folder', None)
        self._Folder_data = kwargs.get('FD', False)

        self._Model_name = kwargs.get('MN', None)
        self._Epochs = kwargs.get('epochs', None)

        self._Columns = ["Loss", "Accuracy"]

        if(isinstance(self._Epochs, str)):
            self._Epochs = int(self._Epochs)

    # * Class variables
    def __repr__(self):
            return f'[{self._Input}, {self._Output}, {self._Folder}, {self._Folder_data}, {self._Model_name}, {self._Epochs}, {self._Columns}]';

    # * Class description
    def __str__(self):
        return  f'.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, Euler number class destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'Inputs': str(self._Input),
                'Outputs': str(self._Output),
                'Folder path': str(self._Folder),
                'Folder path save': str(self._Folder_data),
                'Model name': str(self._Model_name),
                'Epochs': str(self._Epochs),
                };

    # * _Input attribute
    @property
    def _Input_property(self):
        return self._Input

    @_Input_property.setter
    def _Input_property(self, New_value):
        print("Changing input...")
        self._Input = New_value
    
    @_Input_property.deleter
    def _Input_property(self):
        print("Deleting input...")
        del self._Input

    # * _Output attribute
    @property
    def _Output_property(self):
        return self._Output

    @_Output_property.setter
    def _Output_property(self, New_value):
        print("Changing output...")
        self._Output = New_value
    
    @_Output_property.deleter
    def _Output_property(self):
        print("Deleting output...")
        del self._Output

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

    # * _Folder_data attribute
    @property
    def _Folder_data_property(self):
        return self._Folder_data

    @_Folder_data_property.setter
    def _Folder_data_property(self, New_value):
        print("Changing folders state...")
        self._Folder_data = New_value
    
    @_Folder_data_property.deleter
    def _Folder_data_property(self):
        print("Deleting folders state...")
        del self._Folder_data

    # * _Folder attribute
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

    # * _Folder attribute
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
        plt.figure(figsize = (8, 8))
        plt.title('Training loss')
        plt.xlabel ("# Epoch")
        plt.ylabel ("# Loss")
        plt.ylim([0, 2.0])
        plt.plot(Hist_data.history["loss"])
        #plt.show()
        plt.close()

        Figure_name = "Figure_Loss_{}.png".format(self._Model_name)
        Figure_name_folder = os.path.join(self._Folder, Figure_name)

        plt.savefig(Figure_name_folder)

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

# ?
class EulerNumberML3D(EulerNumberML):
    """
    EulerNumberML inheritance

    '''''

    Methods:
        data_dic(): description

        print_octovoxel_order_3D(): description

        read_image_with_metadata_3D(): description

        Show_array_3D(): description

        true_data_3D(): description

        Predictions_3D(): description

        obtain_arrays_from_object_3D(): description

        obtain_arrays_3D(): description

        model_euler_MLP_3D(): description
        
        model_euler_RF_3D(): description

        model_prediction_3D(): description

    """

    # * Initializing (Constructor, super)
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        """
        Keyword Args:
            input (np.ndarray): description 
            output (np.ndarray): description
            folder (str): description
            FD (bool):description
            MN (str):description
            epochs (int):description
        """

    # * Class variables
    def __repr__(self):
            return f'[{self._Input}, {self._Output}, {self._Folder}, {self._Folder_data}, {self._Model_name}, {self._Epochs}, {self._Columns}]';

    # * Class description
    def __str__(self):
        return  f'.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, Euler number class destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'Inputs': str(self._Input),
                'Outputs': str(self._Output),
                'Folder path': str(self._Folder),
                'Folder path save': str(self._Folder_data),
                'Model name': str(self._Model_name),
                'Epochs': str(self._Epochs),
                };

    # ? Static method to print octovoxel.
    @staticmethod
    def print_octovoxel_order_3D() -> None:
        """
        Static method to print octovoxel.

        Args:
            Array_file (str): description
        """

        Letters = ('a', 'c', 'b', 'd', 'e', 'h', 'f', 'g')
        # *
        Array_prediction_octov = np.zeros((2, 2, 2))

        # *
        Array_prediction_octov[0][0][0] = 1 #'a'
        Array_prediction_octov[0][0][1] = 2 #'c'
        Array_prediction_octov[0][1][0] = 3 #'b'
        Array_prediction_octov[0][1][1] = 4 #'d'
        Array_prediction_octov[1][0][0] = 5 #'e'
        Array_prediction_octov[1][0][1] = 6 #'h'
        Array_prediction_octov[1][1][0] = 7 #'f'
        Array_prediction_octov[1][1][1] = 8 #'g'

        print('\n')
        print(Array_prediction_octov)
        print('\n')

        for i, letter in enumerate(Letters):
            print('{} ------> {}'.format(i + 1, letter))
        print('\n')

    # ? Static method to load txt and convert it into 3D tensor.
    @staticmethod
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

    # ? Static method to show array using plt 3D
    @staticmethod
    def Show_array_3D(Image: str) -> np.ndarray:
        """
        Static method to show array using plt 3D

        Args:
            Image (str): description
        """

        # *
        Data = np.genfromtxt(Image, delimiter = ",")
        
        Height = Data.shape[0]/Data.shape[1]
        Data = Data.reshape((int(Height), int(Data.shape[1]), int(Data.shape[1])));

        # *
        print(Data)

        # *
        ax = plt.figure().add_subplot(projection = '3d')
        ax.voxels(Data, edgecolors = 'gray')
        plt.show()

    # ? Static method to change the result for a multiclass task
    @staticmethod
    def true_data_3D(Result: int) -> int:
        """
        Static method to change the result for a multiclass trask

        Args:
            Result (int): description
        
        Returns:
            int: description
        """

        if Result == 0:
            New_Result = 0
        elif Result == 1:
            New_Result = 1
        elif Result == 2:
            New_Result = -1
        elif Result == 3:
            New_Result = -2

        return New_Result

    # ? Method to utilize prediction models such as MLP and RF
    def Predictions_3D(self, Model_name: str, Model_prediction: Any, Prediction_value: Any) -> int:
        """
        Method to utilize prediction model such as MLP and RF

        Args:
            Model_name (str): description
            Model_prediction (Any): description
            Prediction_value (Any): description

        Returns:
            int: description
        """

        #Asterisks = 30

        print("Prediction!")
        #Do not use Model.predict, use model instead
        #Result = Model([Prediction_value])

        if Model_name.endswith('.h5'):
            Result = np.argmax(Model_prediction.predict([Prediction_value]), axis = 1)
            print(Result)

        elif Model_name.endswith('.joblib'):
            Result = Model_prediction.predict([Prediction_value])
            print(Result)

        #Result = np.argmax(Model.predict([Prediction_value]), axis = 0)

        #print(Result)

        True_result = self.true_data_3D(Result)

        #print("*" * Asterisks)
        print('{} -------------- {}'.format(Prediction_value, True_result))
        print('The result is: {}'.format(Result))
        print('The true value is: {}'.format(True_result))
        #print("*" * Asterisks)
        print('\n')

        return True_result

    # ? Method to obtain 1D arrays from a 3D array
    @Utilities.time_func
    @profile
    def obtain_arrays_from_object_3D(self, Object: str) -> list[np.ndarray]:
        """
        Method to obtain 1D arrays from a 3D array

        Args:
            Object (str): description

        """

        #Array = np.loadtxt(self.Object, delimiter = ',')

        # *
        Arrays = []
        Asterisks = 30

        l = 2

        # *
        Array_new = self.read_image_with_metadata_3D(Object)

        # * Creation of empty numpy arrays 3D
        Array_prediction_octov = np.zeros((l, l, l))
        Array_prediction = np.zeros((8))

        # *
        for i in range(Array_new.shape[0] - 1):
            for j in range(Array_new.shape[1] - 1):
                for k in range(Array_new.shape[2] - 1):

                    # *
                    #Array_prediction_octov[0][0][0] = Array_new[i][j][k]
                    #Array_prediction_octov[0][0][1] = Array_new[i][j][k + 1]

                    #Array_prediction_octov[0][1][0] = Array_new[i][j + 1][k]
                    #Array_prediction_octov[0][1][1] = Array_new[i][j + 1][k + 1]

                    #Array_prediction_octov[1][0][0] = Array_new[i + 1][j][k]
                    #Array_prediction_octov[1][0][1] = Array_new[i + 1][j][k + 1]

                    #Array_prediction_octov[1][1][0] = Array_new[i + 1][j + 1][k]
                    #Array_prediction_octov[1][1][1] = Array_new[i + 1][j + 1][k + 1]

                    Array_new[i:l + i, j:l + j, k:l + k]

                    # *
                    Array_prediction[0] = Array_new[i + 1][j][k]
                    Array_prediction[1] = Array_new[i + 1][j][k + 1]

                    Array_prediction[2] = Array_new[i][j][k]
                    Array_prediction[3] = Array_new[i][j][k + 1]

                    Array_prediction[4] = Array_new[i + 1][j + 1][k]
                    Array_prediction[5] = Array_new[i + 1][j + 1][k + 1]

                    Array_prediction[6] = Array_new[i][j + 1][k]
                    Array_prediction[7] = Array_new[i][j + 1][k + 1]
                    print('\n')

                    # *
                    print("*" * Asterisks)
                    Array_prediction_list = Array_prediction.tolist()
                    Array_prediction_list_int = [int(i) for i in Array_prediction_list]

                    # *
                    print("Kernel array")
                    print(Array_new[i:l + i, j:l + j, k:l + k])
                    print('\n')
                    print("Prediction array")
                    print(Array_prediction)
                    print('\n')
                    Arrays.append(Array_prediction_list_int)
                    print("*" * Asterisks)
                    print('\n')

        # *
        for i in range(len(Arrays)):
            print('{} ---- {}'.format(i, Arrays[i]))
        print('\n')
        
        return Arrays

    # ? Method to obtain 1D arrays from a 3D array (np.ndarray)
    @Utilities.time_func
    @profile
    def obtain_arrays_3D(self, Array_new: str) -> list[np.ndarray]:
        """
        Method to obtain 1D arrays from a 3D array (np.ndarray)

        Args:
            Array_new (str): description

        """

        #Array = np.loadtxt(self.Object, delimiter = ',')

        # *
        Arrays = []
        Asterisks = 30

        l = 2

        # *
        #Array_new = self.read_image_with_metadata_3D(Object)

        # *
        Array_prediction_octov = np.zeros((l, l, l))
        Array_prediction = np.zeros((8))

        # *
        for i in range(Array_new.shape[0] - 1):
            for j in range(Array_new.shape[1] - 1):
                for k in range(Array_new.shape[2] - 1):

                    # *
                    #Array_prediction_octov[0][0][0] = Array_new[i][j][k]
                    #Array_prediction_octov[0][0][1] = Array_new[i][j][k + 1]

                    #Array_prediction_octov[0][1][0] = Array_new[i][j + 1][k]
                    #Array_prediction_octov[0][1][1] = Array_new[i][j + 1][k + 1]

                    #Array_prediction_octov[1][0][0] = Array_new[i + 1][j][k]
                    #Array_prediction_octov[1][0][1] = Array_new[i + 1][j][k + 1]

                    #Array_prediction_octov[1][1][0] = Array_new[i + 1][j + 1][k]
                    #Array_prediction_octov[1][1][1] = Array_new[i + 1][j + 1][k + 1]

                    Array_new[i:l + i, j:l + j, k:l + k]

                    # *
                    Array_prediction[0] = Array_new[i + 1][j][k]
                    Array_prediction[1] = Array_new[i + 1][j][k + 1]

                    Array_prediction[2] = Array_new[i][j][k]
                    Array_prediction[3] = Array_new[i][j][k + 1]

                    Array_prediction[4] = Array_new[i + 1][j + 1][k]
                    Array_prediction[5] = Array_new[i + 1][j + 1][k + 1]

                    Array_prediction[6] = Array_new[i][j + 1][k]
                    Array_prediction[7] = Array_new[i][j + 1][k + 1]
                    print('\n')

                    # *
                    print("*" * Asterisks)
                    Array_prediction_list = Array_prediction.tolist()
                    Array_prediction_list_int = [int(i) for i in Array_prediction_list]

                    # *
                    print("Kernel array")
                    print(Array_new[i:l + i, j:l + j, k:l + k])
                    print('\n')
                    print("Prediction array")
                    print(Array_prediction)
                    print('\n')
                    Arrays.append(Array_prediction_list_int)
                    print("*" * Asterisks)
                    print('\n')

        # *
        for i in range(len(Arrays)):
            print('{} ---- {}'.format(i, Arrays[i]))
        print('\n')
        
        return Arrays

    # ? Method to to train a MLP for a 3D image
    @Utilities.time_func
    @Utilities.detect_GPU
    @profile
    def model_euler_MLP_3D(self) -> Any:
        """
        Method to to train a MLP for a 3D image

        """

        # *
        print(self._Input.shape)
        print(self._Output.shape)

        # *
        Model = Sequential()
        Model.add(Dense(units = 1, input_shape = [8]))
        Model.add(Dense(64, activation = "sigmoid"))
        Model.add(Dense(4, activation = 'softmax'))

        # *
        Opt = Adam(learning_rate = 0.001)

        # *
        Model.compile(
            optimizer = Opt, 
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
            #sparse_categorical_crossentropy
        )   

        # *
        print('\n')
        print("Training...")
        print('\n')

        # *
        Hist_data = Model.fit(self._Input, self._Output, epochs = self._Epochs, verbose = False)

        print('\n')
        print("Model trained")
        print('\n')

        # * Saving model using .h5
        Model_name_save = '{}.h5'.format(self._Model_name)
        Model_folder_save = os.path.join(self._Folder, Model_name_save)

        Model.save(Model_folder_save)

        # *
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

    # ? Method to to train a RF for a 3D image
    @Utilities.time_func
    @Utilities.detect_GPU
    @profile
    def model_euler_RF_3D(self) -> None:
        """
        Method to to train a RF for a 3D image

        """

        # *
        print(self._Input.shape)
        print(self._Output.shape)

        # *
        Model_RF = RandomForestClassifier(  criterion = 'gini',
                                            n_estimators = 10,
                                            random_state = 2,
                                            n_jobs = 10)

        # *
        print('\n')
        print("Training...")
        print('\n')

        # *
        Model_RF.fit(self._Input, self._Output)

        print('\n')
        print("Model trained")
        print('\n')

        # * Saving model using .h5
        Model_name_save = '{}.joblib'.format(self._Model_name)
        Model_folder_save = os.path.join(self._Folder, Model_name_save)

        joblib.dump(Model_RF, Model_folder_save)

        pred_Input = Model_RF.predict(self._Input)

        print('Prediction output')
        print(pred_Input)
        print('\n')
        print('Original output')
        print(self._Output)

        print('\n')
        
        AC = accuracy_score(self._Output, pred_Input)

        print('Result: {}'.format(AC))
        print('\n')

        # *
        print("Saving model...")
        print('\n')

        # *
        #Loss = Hist_data.history['loss']
        #Accuracy = Hist_data.history['accuracy']

        # *
        #self.create_dataframe_history(self.Columns, self.Folder, self.Model_name, Hist_data)

        # *
        #self.plot_data_loss(Hist_data)

        # *
        #self.plot_data_accuracy(Hist_data)
        print('\n')

    # ? Method to utilize prediction model such as MLP and RF
    @Utilities.time_func
    @Utilities.detect_GPU
    @profile
    def model_prediction_3D(self, Model, Arrays) -> None:
        """
        Method to utilize prediction model such as MLP and RF

        """

        #Array = np.loadtxt(r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Example_3D_1.txt", delimiter = ',')

        # *
        Prediction_result_3D = 0
        Asterisks = 30

        # * Read multilayer perceptron model
        if Model.endswith('.h5'):
            Model_prediction = load_model(Model)

        # * Read machine learning model
        elif Model.endswith('.joblib'):
            Model_prediction = joblib.load(Model)

        # *
        for i, Array in enumerate(Arrays):

            True_result_3D = self.Predictions_3D(Model, Model_prediction, Array) ####
            Prediction_result_3D += True_result_3D

        print('\n')

        print('Euler: {}'.format(Prediction_result_3D))
        print('\n')

        return Prediction_result_3D

# ?
class EulerNumberML2D(EulerNumberML):
    """
    EulerNumberML inheritance

    '''''

    Methods:
        data_dic(): description

        read_image_with_metadata_2D(): description

        true_data_2D(): description

        Predictions_2D(): description

        obtain_arrays_from_object_2D(): description

        obtain_arrays_2D(): description

        model_euler_MLP_2D(): description
        
        model_euler_RF_2D(): description

        model_prediction_2D(): description

    """
    
    # * Initializing (Constructor, super)
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        """
        Keyword Args:
            input (np.ndarray): description 
            output (np.ndarray): description
            folder (str): description
            FD (bool):description
            MN (str):description
            epochs (int):description
        """

    # ? Static method to load txt and convert it into 2D tensor.
    @staticmethod
    def read_image_with_metadata_2D(Array_file: str) -> np.ndarray:
        """
        Static method to load txt and convert it into tensors

        Args:
            Array_file (str): description
        """

        Array = np.loadtxt(Array_file, delimiter = ',')
        
        Array = Array.astype(int)

        print('\n')
        print('Array obtained')
        print('\n')
        print(Array)
        print('\n')
        print('Number of rows: {}'.format(Array.shape[0]))
        print('\n')
        print('Number of columns: {}'.format(Array.shape[1]))
        print('\n')
        
        return Array

    # ? Static method to change the result for a binary task
    @staticmethod
    def true_data_2D(Result: int) -> int:
        """
        Static method to change the result for a binary task

        Args:
            Result (int): description
        
        Returns:
            int: description
        """

        if Result > 0.5:
            New_Result = 1
        elif Result < 0.5 and Result > -0.5:
            New_Result = 0
        elif Result < -0.5:
            New_Result = -1

        return New_Result

    # ? Method to utilize prediction model such as MLP
    def Predictions_2D(self, Model: Any, Prediction_value: Any) -> int:
        """
        Method to utilize prediction model such as MLP and RF

        Args:
            Model_prediction (Any): description
            Prediction_value (Any): description

        Returns:
            int: description
        """
        #Asterisks = 30

        print("Prediction!")
        #Do not use Model.predict, use model instead
        Result = Model.predict([Prediction_value])

        True_result = self.true_data_2D(Result)

        #print("*" * Asterisks)
        print('The result is: {}'.format(Result))
        print('The true value is: {}'.format(True_result))
        #print("*" * Asterisks)
        print('\n')

        return True_result

    # ? Method to obtain 1D arrays from a 2D array
    @Utilities.time_func
    def obtain_arrays_from_object_2D(self, Object) -> list[np.ndarray]:
        """
        Method to obtain 1D arrays from a 2D array

        Args:
            Object (str): description

        """

        # *
        Arrays = []
        Asterisks = 30

        k = 2

        # *
        Array = self.read_image_with_metadata_2D(Object)

        # *
        Array_comparison = np.zeros((k, k), dtype = 'int')
        Array_prediction = np.zeros((4), dtype = 'int')

        for i in range(Array.shape[0] - 1):
            for j in range(Array.shape[1] - 1):

                #Array_comparison[0][0] = Array[i][j]
                #Array_comparison[1][0] = Array[i + 1][j]
                #Array_comparison[0][1] = Array[i][j + 1]
                #Array_comparison[1][1] = Array[i + 1][j + 1]

                Array[i:k + i, j:k + j]

                Array_prediction[0] = Array[i][j]
                Array_prediction[1] = Array[i][j + 1]
                Array_prediction[2] = Array[i + 1][j]
                Array_prediction[3] = Array[i + 1][j + 1]

                print('\n')
                #print("*" * Asterisks)

                # *
                print("*" * Asterisks)
                Array_prediction_list = Array_prediction.tolist()
                Array_prediction_list_int = [int(i) for i in Array_prediction_list]

                # *
                print("Kernel array")
                print(Array[i:k + i, j:k + j])
                print('\n')
                print("Prediction array")
                print(Array_prediction)
                print('\n')
                Arrays.append(Array_prediction_list_int)
                print("*" * Asterisks)
                print('\n')

        # *
        for i in range(len(Arrays)):
            print('{} ---- {}'.format(i, Arrays[i]))
        print('\n')
        
        return Arrays

    # ? Method to obtain 1D arrays from a 2D array
    @Utilities.time_func
    def obtain_arrays_2D(self, Array) -> list[np.ndarray]:
        """
        Method to obtain 1D arrays from a 2D array (np.ndarray)

        Args:
            Array_new (str): description

        """

        # *
        Arrays = []
        Asterisks = 30

        k = 2

        # * Creation of empty numpy arrays 2D
        Array_comparison = np.zeros((k, k), dtype = 'int')
        Array_prediction = np.zeros((4), dtype = 'int')

        for i in range(Array.shape[0] - 1):
            for j in range(Array.shape[1] - 1):

                Array_comparison[0][0] = Array[i][j]
                Array_comparison[1][0] = Array[i + 1][j]
                Array_comparison[0][1] = Array[i][j + 1]
                Array_comparison[1][1] = Array[i + 1][j + 1]
                
                Array[i:k + i, j:k + j]

                Array_prediction[0] = Array[i][j]
                Array_prediction[1] = Array[i][j + 1]
                Array_prediction[2] = Array[i + 1][j]
                Array_prediction[3] = Array[i + 1][j + 1]

                print('\n')
                #print("*" * Asterisks)

                # *
                print("*" * Asterisks)
                Array_prediction_list = Array_prediction.tolist()
                Array_prediction_list_int = [int(i) for i in Array_prediction_list]

                # *
                print("Kernel array")
                print(Array[i:k + i, j:k + j])
                print('\n')
                print("Prediction array")
                print(Array_prediction)
                print('\n')
                Arrays.append(Array_prediction_list_int)
                print("*" * Asterisks)
                print('\n')

        # *
        for i in range(len(Arrays)):
            print('{} ---- {}'.format(i, Arrays[i]))
        print('\n')
        
        return Arrays

    # ? Method to to train a MLP for a 2D image
    @Utilities.time_func
    @Utilities.detect_GPU
    @profile
    def model_euler_MLP_2D(self) -> Any:
        """
        Method to to train a MLP for a 3D image

        """

        print(self._Input.shape)
        print(self._Output.shape)

        Model = Sequential()
        Model.add(Dense(units = 1, input_shape = [4]))
        Model.add(Dense(9, activation = "sigmoid"))
        Model.add(Dense(1, activation = 'tanh'))

        Opt = Adam(learning_rate = 0.1)

        Model.compile(
            optimizer = Opt, 
            loss = 'mean_squared_error',
            metrics = ['accuracy']
        )

        print('\n')
        print("Training...")
        print('\n')
        
        # *
        Hist_data = Model.fit(self._Input, self._Output, epochs = self._Epochs, verbose = False)

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
        #self.plot_data_accuracy(Hist_data)
        print('\n')

        return Hist_data

    # ? Method to utilize prediction model such as MLP
    @Utilities.time_func
    @Utilities.detect_GPU
    @profile
    def model_prediction_2D(self, Model, Arrays):
        """
        Method to utilize prediction model such as MLP

        """
        
        #Array = np.loadtxt(r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Example_3D_1.txt", delimiter = ',')

        # *
        Prediction_result_2D = 0
        Asterisks = 30

        # * Read multilayer perceptron model
        if Model.endswith('.h5'):
            Model_prediction = load_model(Model)

        # * Read machine learning model
        elif Model.endswith('.joblib'):
            Model_prediction = joblib.load(Model)

        # *
        for i, Array in enumerate(Arrays):

            True_result_2D = self.Predictions_2D(Model_prediction, Array) ####
            Prediction_result_2D += True_result_2D
        print('\n')

        print('Euler: {}'.format(Prediction_result_2D))
        print('\n')

        return Prediction_result_2D

    # ? Method to utilize connectivity 4 arrays to search for euler's number
    @Utilities.time_func
    @profile
    def connectivity_4_prediction_2D(self, Arrays)-> None:
        """
        Method to utilize connectivity 4 arrays to search for euler's number

        """

        Connectivity_4_first_array = np.array([1, 0, 0, 0], dtype = 'int')

        Connectivity_4_second_array = np.array([1, 1, 1, 0], dtype = 'int')

        Connectivity_4_third_array = np.array([1, 0, 0, 1], dtype = 'int')

        # *
        Result_connected_4 = 0

        # *
        for i, Array in enumerate(Arrays):

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
    @Utilities.time_func
    @profile
    def connectivity_8_prediction_2D(self, Arrays) -> None:
        """
        Method to utilize connectivity 8 arrays to search for euler's number

        """

        Connectivity_8_first_array = np.array([1, 0, 0, 0], dtype = 'int')

        Connectivity_8_second_array = np.array([1, 1, 1, 0], dtype = 'int')

        Connectivity_8_third_array = np.array([0, 1, 1, 0], dtype = 'int')

        # *
        Result_connected_8 = 0

        # *
        for _, Array in enumerate(Arrays):

            print(np.array(Array))

            if(np.array_equal(np.array(Array), Connectivity_8_first_array)):
                Result_connected_8 += 1
                print('Connectivity 4: {}'.format(Result_connected_8))

            if(np.array_equal(np.array(Array), Connectivity_8_second_array)):
                Result_connected_8 -= 1
                print('Connectivity 4: {}'.format(Result_connected_8))

            if(np.array_equal(np.array(Array), Connectivity_8_third_array)):
                Result_connected_8 += 1
                print('Connectivity 4: {}'.format(Result_connected_8))

        print('\n')

        print('Connectivity 4: {}'.format(Result_connected_8))
        print('\n')