# ?
def data_tabla_input_ouput(Input_3D_array: np.ndarray, Output_3D_array: np.ndarray) -> None:
    """
    _summary_

    _extended_summary_

    Args:
        Input_3D_array (np.ndarray): _description_
        Output_3D_array (np.ndarray): _description_
    """
    print('\n')
    # *
    if(len(Input_3D_array) == len(Output_3D_array)):

        # *
        for i in range(len(Output_3D_array)):
            print('{} ------ {} ------ {}'.format(i, Input_3D_array[i], Output_3D_array[i]))

    else:
        raise AssertionError('Input and output dont have the same length. Input: {}, Output: {}'.format(len(Input_3D_array), len(Output_3D_array)))
    print('\n')

# ?
def bubbleSort(arr: list[int]) -> None:
    """
    _summary_

    _extended_summary_

    Args:
        arr (list[int]): _description_
    """
    # *
    len_array = len(arr)

    # *
    swapped = False

    # *
    for i in range(len_array - 1):
        for j in range(0, len_array - i - 1):
 
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
         
        if not swapped:
            return

    return arr

# ?
def print_array(Sorted_arr: list[int]) -> None:
    """
    _summary_

    _extended_summary_

    Args:
        Sorted_arr (list[int]): _description_
    """

    # *
    print('\n')

    print("Sorted array is: ")
    print('\n')

    # *
    for i in range(len(Sorted_arr)):
        print(" % d " % Sorted_arr[i], end = ",")
    print('\n')

# ?
def index_assign_value(Tuple_index: tuple[int], Tuple_value: tuple[int]) -> Any:
    """
    _summary_

    _extended_summary_

    Args:
        Tuple_index (tuple[int]): _description_
        Tuple_value (tuple[int]): _description_

    Raises:
        AssertionError: _description_

    Returns:
        Any: _description_
    """

    # *
    if(len(Tuple_index) == len(Tuple_value)):
        
        # *
        Emp_array = np.zeros((256), dtype = 'int')
        print('\n')

        # *
        print('{}'.format(Emp_array))
        print('\n')

        #OutList = []
        #print(Output.dtype)
        #print(len(Output))

        # *
        for i in range(len(Tuple_index)):
            print('Position: {} ------- Value: {}'.format(Tuple_index, Tuple_value))
        print('\n')

        # *
        print('{}'.format(len(Tuple_index)))
        print('{}'.format(len(Tuple_value)))
        print('\n')

        # *
        for i, element in enumerate(Emp_array):
            for j, value in enumerate(Tuple_index):
                if Tuple_index[j] == i:

                    Emp_array[i] = Tuple_value[j]
                else:
                    pass
        
        # *
        print('\n')
        print(Emp_array) 

        # *
        print('\n')
        print(Emp_array.dtype) 

    else:
        raise AssertionError('Tuples dont have the same length. Index: {}, Value: {}'.format(len(Tuple_index), len(Tuple_value)))
    print('\n')
    
    return Emp_array

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

def obtain_arrays_from_object_2D(Object) -> list[np.ndarray]:
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
        Array = read_image_with_metadata_2D(Object)
        Qs = []
        
        # *
        Q1 = np.array([ [ 0,  0],
                        [ 0,  0]    ], dtype = 'int')

        Q2 = np.array([ [ 0,  0],
                        [ 0,  1]    ], dtype = 'int')
        
        Q3 = np.array([ [ 0,  0],
                        [ 1,  0]    ], dtype = 'int')

        Q4 = np.array([ [ 0,  0],
                        [ 1,  1]    ], dtype = 'int')

        Q5 = np.array([ [ 0,  1],
                        [ 0,  0]    ], dtype = 'int')

        Q6 = np.array([ [ 0,  1],
                        [ 0,  1]    ], dtype = 'int')

        Q7 = np.array([ [ 0,  1],
                        [ 1,  0]    ], dtype = 'int')

        Q8 = np.array([ [ 0,  1],
                        [ 1,  1]    ], dtype = 'int')

        Q9 = np.array([ [ 1,  0],
                        [ 0,  0]    ], dtype = 'int')

        Q10 = np.array([ [ 1,  0],
                         [ 0,  1]    ], dtype = 'int')

        Q11 = np.array([    [ 1,  0],
                            [ 1,  0]    ], dtype = 'int')

        Q12 = np.array([    [ 1,  0],
                            [ 1,  1]    ], dtype = 'int')
        
        Q13 = np.array([    [ 1,  1],
                            [ 0,  0]    ], dtype = 'int')
        
        Q14 = np.array([    [ 1,  1],
                            [ 0,  1]    ], dtype = 'int')
        
        Q15 = np.array([    [ 1,  1],
                            [ 1,  0]    ], dtype = 'int')
        
        Q16 = np.array([    [ 1,  1],
                            [ 1,  1]    ], dtype = 'int')
        
        Qs.extend((Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15, Q16))
        Qs_value = np.zeros((16), dtype = 'int')

        Array_comparison = np.zeros((k, k), dtype = 'int')
        Array_prediction = np.zeros((4), dtype = 'int')

        for i in range(Array.shape[0] - 1):
            for j in range(Array.shape[1] - 1):
                
                #Array_comparison[0][0] = Array[i][j]
                #Array_comparison[1][0] = Array[i + 1][j]
                #Array_comparison[0][1] = Array[i][j + 1]
                #Array_comparison[1][1] = Array[i + 1][j + 1]

                #Array_prediction[0] = Array[i][j]
                #Array_prediction[1] = Array[i][j + 1]
                #Array_prediction[2] = Array[i + 1][j]
                #Array_prediction[3] = Array[i + 1][j + 1]

                for Index in range(len(Qs)):
                    
                    print('Kernel: {}'.format(Array[i:k + i, j:k + j]))
                    print('Qs: {}'.format(Qs[Index]))
                    print('\n')
                    print('\n')

                    if(np.array_equal(Array[i:k + i, j:k + j], Qs[Index])):
                        Qs_value[Index] += 1
                        print('Q{}_value: {}'.format(Index, Qs_value[Index]))
                
                print(Qs_value)
                print('\n')
                #print("*" * Asterisks)

                # *
                print("*" * Asterisks)
                #Array_prediction_list = Array_prediction.tolist()
                #Array_prediction_list_int = [int(i) for i in Array_prediction_list]

                # *
                #print("Kernel array")
                #print(Array[i:k + i, j:k + j])
                #print('\n')
                #print("Prediction array")
                #print(Array_prediction)
                #print('\n')
                #Arrays.append(Array_prediction_list_int)
                print("*" * Asterisks)
                print('\n')

        # *
        #for i in range(len(Arrays)):
            #print('{} ---- {}'.format(i, Arrays[i]))
        #print('\n')
        
        return Arrays

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

# ?
    @Utilities.time_func
    def generate_euler_samples_settings(self) -> None:
        """
        _summary_

        _extended_summary_
        """
        DataFrame = pd.DataFrame()
        
        # *
        Prediction = EulerNumberML3D(input = Input_3D_array, output = Output_3D_array, folder = self.__Folder);

        # *
        Remove_files = RemoveFiles(folder = self.__Folder)
        Remove_files.remove_all()

        # *
        for i in range(self.__Number_of_images):
            
            P_0 = random.uniform(0, 1)
            P_1 = 1 - P_0

            # *
            #Data_3D = np.random.randint(0, 2, (self._Height * self._Depth * self._Width));
            Data_3D = np.random.choice(2, self.__Height * self.__Depth * self.__Width, p = [P_0, P_1]);
            Data_3D = Data_3D.reshape((self.__Height * self.__Depth), self.__Width);
            Data_3D_plot = Data_3D.reshape((self.__Height, self.__Depth, self.__Width));

            # *
            Data_3D_edges_complete = np.zeros((Data_3D_plot.shape[1] + 2, Data_3D_plot.shape[2] + 2))
            Data_3D_edges_concatenate = np.zeros((Data_3D_plot.shape[1] + 2, Data_3D_plot.shape[2] + 2))
            Data_3D_read = np.zeros((Data_3D.shape[0] + 2, Data_3D.shape[1] + 2))
            
            # * 
            Data_3D_edges = np.zeros((Data_3D_plot.shape[0] + 2, Data_3D_plot.shape[1] + 2, Data_3D_plot.shape[2] + 2))
            
            # * Get 3D image and interpretation of 3D from 2D .txt
            Data_3D_read[1:Data_3D_read.shape[0] - 1, 1:Data_3D_read.shape[1] - 1] = Data_3D
            Data_3D_edges[1:Data_3D_edges.shape[0] - 1, 1:Data_3D_edges.shape[1] - 1, 1:Data_3D_edges.shape[2] - 1] = Data_3D_plot

            #print(Data_3D_read);
            #print(Data_3D_edges);
            #print('\n');

            # * Concatenate np.zeros
            Data_3D_read = np.concatenate((Data_3D_edges_concatenate, Data_3D_read), axis = 0)
            Data_3D_read = np.concatenate((Data_3D_read, Data_3D_edges_concatenate), axis = 0)

            for k in range(len(Data_3D_edges) - 2):
                Data_3D_edges_complete = np.concatenate((Data_3D_edges_complete, Data_3D_edges[k + 1]), axis = 0)

            Data_3D_edges_complete = np.concatenate((Data_3D_edges_complete, Data_3D_edges_concatenate), axis = 0)

            Array = Prediction.obtain_arrays_3D(Data_3D_edges);
            Euler_number = Prediction.model_prediction_3D(self.__Model_trained, Array);

            #print(Data_3D_read);

            # * 
            for j in range(self.__Depth + 2):
                
                # *
                Dir_name_images = "Images_random_{}_3D".format(i)

                # *
                Dir_data_images = self.__Folder + '/' + Dir_name_images

                # *
                Exist_dir_images = os.path.isdir(Dir_data_images)
                
                if Exist_dir_images == False:
                    Folder_path_images = os.path.join(self.__Folder, Dir_name_images)
                    os.mkdir(Folder_path_images)
                    print(Folder_path_images)
                else:
                    Folder_path_images = os.path.join(self.__Folder, Dir_name_images)
                    print(Folder_path_images)

                Image_name = "Image_slice_random_{}_{}_3D".format(i, j)
                Image_path = os.path.join(Folder_path_images, Image_name)
                #plt.title('P_0: {}, P_1: {}'.format(P_0, P_1))
                plt.title('Euler: {}'.format(Euler_number))
                plt.imshow(Data_3D_edges[j], cmap = 'gray', interpolation = 'nearest')
                plt.savefig(Image_path)
                plt.close()

            File_name = 'Image_random_{}_3D.txt'.format(i);
            Path = os.path.join(self.__Folder, File_name);
            np.savetxt(Path, Data_3D_edges_complete, fmt = '%0.0f', delimiter = ',');

            Array = get_octovoxel_3D(Path)

            print(Array)
            print('///// ' + str(Euler_number))

            Array = np.append(Array, Euler_number)

            print(Array)

            # * Return the new dataframe with the new data
            DataFrame = DataFrame.append(pd.Series(Array), ignore_index = True)
                
            Dataframe_name = 'Dataframe_test.csv'.format()
            Dataframe_folder = os.path.join(r'Objects\3D\Data', Dataframe_name)
            DataFrame.to_csv(Dataframe_folder)

    # ?
    @Utilities.time_func
    def create_data_euler_3D_settings(self) -> None:
        """
        _summary_

        _extended_summary_
        """
        
        # *
        global Input_3D_array
        global Output_3D_array

        # *
        Prediction = EulerNumberML3D(input = Input_3D_array, output = Output_3D_array, folder = self.__Folder);

        # *
        Remove_files = RemoveFiles(folder = self.__Folder)
        Remove_files.remove_all()

        # *
        for i in range(self.__Number_of_images):

            #Data_2D = np.random.randint(0, 2, (self._Height * self._Width))

            Euler_number = 0

            P_0 = 0.2
            P_1 = 0.8

            while(Euler_number != self.__Euler_number):
                
                # *
                Data_3D = np.random.choice(2, self.__Height * self.__Depth * self.__Width, p = [P_0, P_1]);
                Data_3D = Data_3D.reshape((self.__Height * self.__Depth), self.__Width);
                Data_3D_plot = Data_3D.reshape((self.__Height, self.__Depth, self.__Width));

                # *
                Data_3D_edges_concatenate = np.zeros((Data_3D_plot.shape[1] + 2, Data_3D_plot.shape[2] + 2))
                Data_3D_read = np.zeros((Data_3D.shape[0] + 2, Data_3D.shape[1] + 2))
                
                # *
                Data_3D_edges = np.zeros((Data_3D_plot.shape[0] + 2, Data_3D_plot.shape[1] + 2, Data_3D_plot.shape[2] + 2))
                
                # * Get 3D image and interpretation of 3D from 2D .txt
                Data_3D_read[1:Data_3D_read.shape[0] - 1, 1:Data_3D_read.shape[1] - 1] = Data_3D
                Data_3D_edges[1:Data_3D_edges.shape[0] - 1, 1:Data_3D_edges.shape[1] - 1, 1:Data_3D_edges.shape[2] - 1] = Data_3D_plot

                #print(Data_3D_read);
                #print(Data_3D_edges);
                #print('\n');

                # * Concatenate np.zeros
                Data_3D_read = np.concatenate((Data_3D_edges_concatenate, Data_3D_read), axis = 0)
                Data_3D_read = np.concatenate((Data_3D_read, Data_3D_edges_concatenate), axis = 0)

                Array = Prediction.obtain_arrays_3D(Data_3D_edges);
                Euler_number = Prediction.model_prediction_3D(self.__Model_trained, Array);

                if(Euler_number > self.__Euler_number):

                    if(P_0 != 0.98):

                        P_0 = P_0 - 0.02;
                        P_1 = P_1 + 0.02;

                else:
                    
                    if(P_1 != 0.98):
                        
                        P_0 = P_0 + 0.02;
                        P_1 = P_1 - 0.02;

            for j in range(self.__Depth):
                
                # *
                Dir_name_images = "Images_with_euler_{}_3D".format(j)

                # *
                Dir_data_images = self.__Folder + '/' + Dir_name_images

                # *
                Exist_dir_images = os.path.isdir(Dir_data_images)
                
                if Exist_dir_images == False:
                    Folder_path_images = os.path.join(self.__Folder, Dir_name_images)
                    os.mkdir(Folder_path_images)
                    print(Folder_path_images)
                else:
                    Folder_path_images = os.path.join(self.__Folder, Dir_name_images)
                    print(Folder_path_images)

                Image_name = "Image_slice_with_euler_{}_{}_3D".format(i, j)
                Image_path = os.path.join(Folder_path_images, Image_name)
                plt.title('P_0: {}, P_1: {}'.format(P_0, P_1))
                plt.imshow(Data_3D_edges[j], cmap = 'gray', interpolation = 'nearest')
                plt.savefig(Image_path)
                plt.close()

            File_name = 'Image_with_euler_{}_3D.txt'.format(i);
            Path = os.path.join(self.__Folder, File_name);
            np.savetxt(Path, Data_3D_read, fmt = '%0.0f', delimiter = ',');