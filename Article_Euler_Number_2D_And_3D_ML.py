from Article_Euler_Number_Libraries import *
from Article_Euler_Number_Utilities import Utilities

# ?
class EulerNumberML(Utilities):

    def __init__(self, **kwargs) -> None:
        """
        _summary_

        _extended_summary_
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

    def __repr__(self):

        kwargs_info = "{}, {}, {}, {}, {}, {}".format(self._Input, self._Output, self._Folder, self._Model_name, self._Epochs, self._Columns)

        return kwargs_info

    def __str__(self):
        pass

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

    # ? Create dataframes
    @staticmethod
    @Utilities.time_func
    def create_dataframe_history(Column_names: Any, Folder_save: str, CSV_name: str, Hist_data: Any) -> None: 

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

    # ?
    @Utilities.time_func
    def plot_data_loss(self, Hist_data: Any) -> None:
        """
        _summary_

        _extended_summary_

        Args:
            Hist_data (Any): _description_
        """
        #plt.figure(figsize = (20, 20))
        plt.title('Training loss')
        plt.xlabel ("# Epoch")
        plt.ylabel ("# Loss")
        plt.plot(Hist_data.history["loss"])
        #plt.show()

        Figure_name = "Figure_Loss_{}.png".format(self._Model_name)
        Figure_name_folder = os.path.join(self._Folder, Figure_name)

        plt.savefig(Figure_name_folder)

    # ?
    @Utilities.time_func
    def plot_data_accuracy(self, Hist_data: Any) -> None:
        """
        _summary_

        _extended_summary_

        Args:
            Hist_data (Any): _description_
        """
        #plt.figure(figsize = (20, 20))
        plt.title('Training accuracy')
        plt.xlabel ("# Epoch")
        plt.ylabel ("# Acuracy")
        plt.plot(Hist_data.history["accuracy"])
        #plt.show()

        Figure_name = "Figure_Accuracy_{}.png".format(self.Model_name)
        Figure_name_folder = os.path.join(self.Folder, Figure_name)

        plt.savefig(Figure_name_folder)

# ?
class EulerNumberML3D(EulerNumberML):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    # ?
    @staticmethod
    def print_octovoxel_order_3D() -> None:
        
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

    # ?
    @staticmethod
    def read_image_with_metadata_3D(Array_file: str) -> np.ndarray:
        """
        _summary_

        _extended_summary_

        Args:
            Array_file (_type_): _description_
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

    # ?
    @staticmethod
    def Show_array_3D(Image: str) -> np.ndarray:
        
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

    # ?
    @staticmethod
    def true_data_3D(Result: int) -> int:
        """
        _summary_

        _extended_summary_

        Args:
            Result (int): _description_

        Returns:
            _type_: _description_
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

    # ? #Model_prediction, Array
    def Predictions_3D(self, Model_name: str, Model_prediction: Any, Prediction_value: Any) -> int:
        """
        _summary_

        _extended_summary_

        Args:
            Model_name (str): _description_
            Model_prediction (Any): _description_
            Prediction_value (Any): _description_

        Returns:
            int: _description_
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

    # ?
    @Utilities.time_func
    @profile
    def obtain_arrays_from_object_3D(self, Object: str) -> list[np.ndarray]:

        #Array = np.loadtxt(self.Object, delimiter = ',')

        # *
        Arrays = []
        Asterisks = 30

        # *
        Array_new = self.read_image_with_metadata_3D(Object)

        # *
        Array_prediction_octov = np.zeros((2, 2, 2))
        Array_prediction = np.zeros((8))

        # *
        for i in range(Array_new.shape[0] - 1):
            for j in range(Array_new.shape[1] - 1):
                for k in range(Array_new.shape[2] - 1):

                    # *
                    Array_prediction_octov[0][0][0] = Array_new[i][j][k]
                    Array_prediction_octov[0][0][1] = Array_new[i][j][k + 1]

                    Array_prediction_octov[0][1][0] = Array_new[i][j + 1][k]
                    Array_prediction_octov[0][1][1] = Array_new[i][j + 1][k + 1]

                    Array_prediction_octov[1][0][0] = Array_new[i + 1][j][k]
                    Array_prediction_octov[1][0][1] = Array_new[i + 1][j][k + 1]

                    Array_prediction_octov[1][1][0] = Array_new[i + 1][j + 1][k]
                    Array_prediction_octov[1][1][1] = Array_new[i + 1][j + 1][k + 1]

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
                    print(Array_prediction_octov)
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

    # ?
    @Utilities.time_func
    @Utilities.detect_GPU
    @profile
    def model_euler_MLP_3D(self) -> Any:
        """
        _summary_

        _extended_summary_

        Returns:
            Any: _description_
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
        Model.save(Model_name_save)

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

    # ?
    @Utilities.time_func
    @Utilities.detect_GPU
    @profile
    def model_euler_RF_3D(self) -> None:
        """
        _summary_

        _extended_summary_

        Returns:
            Any: _description_
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
        joblib.dump(Model_RF, Model_name_save)

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

    # ?
    @Utilities.time_func
    @Utilities.detect_GPU
    @profile
    def model_prediction_3D(self, Model, Arrays) -> None:
    
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

# ?
class EulerNumberML2D(EulerNumberML):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    # ?
    @staticmethod
    def read_image_with_metadata_2D(Array_file: str) -> np.ndarray:
        """
        _summary_

        _extended_summary_

        Args:
            Array_file (_type_): _description_

        Returns:
            _type_: _description_
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

    # ?
    @staticmethod
    def true_data_2D(Result: int) -> int:
        """
        _summary_

        _extended_summary_

        Args:
            Result (int): _description_

        Returns:
            _type_: _description_
        """
        if Result > 0.5:
            New_Result = 1
        elif Result < 0.5 and Result > -0.5:
            New_Result = 0
        elif Result < -0.5:
            New_Result = -1

        return New_Result

    # ?
    def Predictions_2D(self, Model: Any, Prediction_value: Any) -> int:
        """
        _summary_

        _extended_summary_

        Args:
            Model (Any): _description_
            Prediction_value (Any): _description_

        Returns:
            int: _description_
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

    # ?
    @Utilities.time_func
    def obtain_arrays_from_object_2D(self, Object) -> list[np.ndarray]:


        # *
        Arrays = []
        Asterisks = 30

        # *
        Array = self.read_image_with_metadata_2D(Object)

        # *
        Array_comparison = np.zeros((2, 2), dtype = 'int')
        Array_prediction = np.zeros((4), dtype = 'int')

        for i in range(Array.shape[0] - 1):
            for j in range(Array.shape[1] - 1):

                Array_comparison[0][0] = Array[i][j]
                Array_comparison[1][0] = Array[i + 1][j]
                Array_comparison[0][1] = Array[i][j + 1]
                Array_comparison[1][1] = Array[i + 1][j + 1]

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
                print(Array_comparison)
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

    # ?
    @Utilities.time_func
    @Utilities.detect_GPU
    @profile
    def model_euler_MLP_2D(self) -> Any:
        """
        _summary_

        _extended_summary_

        Returns:
            Any: _description_
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
        Model.save(Model_name_save)

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

    # ?
    @Utilities.time_func
    @Utilities.detect_GPU
    @profile
    def model_prediction_2D(self, Model, Arrays):

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

    # ?
    @Utilities.time_func
    @profile
    def connectivity_4_prediction_2D(self, Arrays)-> None:


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

    # ?
    @Utilities.time_func
    @profile
    def connectivity_8_prediction_2D(self, Arrays) -> None:


        Connectivity_8_first_array = np.array([1, 0, 0, 0], dtype = 'int')

        Connectivity_8_second_array = np.array([1, 1, 1, 0], dtype = 'int')

        Connectivity_8_third_array = np.array([0, 1, 1, 0], dtype = 'int')

        # *
        Result_connected_8 = 0

        # *
        for i, Array in enumerate(Arrays):

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