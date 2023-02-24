import os
import json
import numpy as np
import pandas as pd

from Article_Euler_Number_Libraries import *
from Article_Euler_Number_Utilities import Utilities
from Article_Euler_Number_Class_RemoveFiles import RemoveFiles

from Article_Euler_Number_Info_2D_General import Input_2D
from Article_Euler_Number_Info_2D_General import Output_2D_4_Connectivity
from Article_Euler_Number_Info_2D_General import Output_2D_8_Connectivity

from Article_Euler_Number_Info_3D_General import Input_3D_array
from Article_Euler_Number_Info_3D_General import Output_3D_array

from Article_Euler_Number_Class_EulerExtractorUtilities import EulerExtractor2D
from Article_Euler_Number_Class_EulerExtractorUtilities import EulerExtractor3D

from abc import abstractmethod

# ?
class EulerGenerator(Utilities):

    def __init__(self, **kwargs) -> None:
        """
        _summary_

        _extended_summary_
        """

        # * General parameters
        self._Folder = kwargs.get('Folder', None);

        # * Shape 2D and 3D
        self._Height = kwargs.get('Height', 8);
        self._Width = kwargs.get('Width', 8);

        # * 3D
        self.__Depth = kwargs.get('Depth', 8);

        self._Number_of_images = kwargs.get('NI', None);
        self._Number_of_images = int(self.__Number_of_images);

        # *
        self._Save_image = kwargs.get('SI', True);

        # *
        self._Euler_number_desired = kwargs.get('EulerNumber', 1);

        # *
        self._Model_trained = kwargs.get('MT', None);

        if(isinstance(self.__Height, str)):
            self.__Height = int(self.__Height)
        
        if(isinstance(self.__Width, str)):
            self.__Width = int(self.__Width)
        
        if(isinstance(self.__Depth, str)):
            self.__Depth = int(self.__Depth)

        if(isinstance(self.__Euler_number_desired , str)):
            self.__Euler_number_desired  = int(self.__Euler_number_desired)

    def __repr__(self):
        """
        Returns the string representation of this BinaryConversion instance.

        Returns
        -------
        str 
            The string representation of this BinaryConversion instance.
        """
        return f'''[{self.Iter}, 
                    {self.ndim}]''';

    def __str__(self):
        pass
    
    # * Get data from a dic
    def data_dic(self):
        """
        Returns a dictionary containing the instance's Iter and ndim values.

        Returns
        -------
        dict 
            A dictionary containing the instance's Iter and ndim values.
        """
        return {};

    # * Creates a JSON file with the given data and saves it to the specified file path.
    def create_json_file(self):
        """
        Creates a JSON file with the given data and saves it to the specified file path.
        A JSON file containing the instance's Inputs, Outputs, ndim, Folder_data, Model_name values.

        Returns
        ----------
        None
        """
        Data = {};

        with open('JSON documents', 'w') as file:
            json.dump(Data, file)

    # * _Folder attribute
    @property
    def __Folder_property(self):
        return self._Folder

    @__Folder_property.setter
    def __Folder_property(self, New_value):
        print("Changing folder...");
        self.__Folder = New_value;
    
    @__Folder_property.deleter
    def __Folder_property(self):
        print("Deleting folder...");
        del self.__Folder

    # * _Number_of_images attribute
    @property
    def __Number_of_images_property(self):
        return self.__Number_of_images

    @__Number_of_images_property.setter
    def __Number_of_images_property(self, New_value):
        print("Changing number of images...");
        self.__Number_of_images = New_value;
    
    @__Number_of_images_property.deleter
    def __Number_of_images_property(self):
        print("Deleting number of images...");
        del self.__Number_of_images

    # * _Number_of_images attribute
    @property
    def __Number_of_images_property(self):
        return self.__Number_of_images

    @__Number_of_images_property.setter
    def __Number_of_images_property(self, New_value):
        print("Changing number of images...");
        self.__Number_of_images = New_value;
    
    @__Number_of_images_property.deleter
    def __Number_of_images_property(self):
        print("Deleting number of images...");
        del self.__Number_of_images

    # * __Save_image attribute
    @property
    def __Save_image_property(self):
        return self.__Save_image

    @__Save_image_property.setter
    def __Save_image_property(self, New_value):
        print("Changing number of images...");
        self.__Save_image = New_value;
    
    @__Save_image_property.deleter
    def __Save_image_property(self):
        print("Deleting number of images...");
        del self.__Save_image

    @Utilities.time_func
    @abstractmethod
    def generate_euler_samples_random():
        pass
    
    @Utilities.time_func
    @abstractmethod
    def generate_euler_samples_settings():
        pass

# ? Extract Euler numbers from 3D arrays.
class EulerImageGenerator(EulerGenerator):
    
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

    # ?
    @Utilities.time_func
    def generate_euler_samples_random(self) -> None:
        """
        _summary_

        _extended_summary_
        """

        P_0 = 0.4
        P_1 = 0.6

        # *
        Remove_files = RemoveFiles(folder = self.__Folder)
        Remove_files.remove_all()

        for i in range(self.__Number_of_images):

            # *
            #Data_2D = np.random.randint(0, 2, (self._Height * self._Width))
            Data_2D = np.random.choice(2, self.__Height * self.__Width, p = [P_0, P_1]);
            Data_2D = Data_2D.reshape(self.__Height, self.__Width);

            #print(Data_2D);
            #print('\n');
            
            if(self.__Save_image):

                # *
                Image_name = "Image_random_{}_2D.png".format(i)
                Image_path = os.path.join(self.__Folder, Image_name)
                
                # *
                Data_2D_edges = np.zeros((Data_2D.shape[0] + 2, Data_2D.shape[1] + 2))
                
                #print(Data_2D_edges);

                # *
                Data_2D_edges[1:Data_2D_edges.shape[0] - 1, 1:Data_2D_edges.shape[1] - 1] = Data_2D
                print(Data_2D_edges);
                plt.title('P_0: {}, P_1: {}'.format(P_0, P_1))
                plt.imshow(Data_2D_edges, cmap = 'gray', interpolation = 'nearest')
                plt.savefig(Image_path);
                #plt.show()
                plt.close()

            # *
            File_name = 'Image_random_{}_2D.txt'.format(i);
            Path = os.path.join(self.__Folder, File_name);
            np.savetxt(Path, Data_2D_edges, fmt = '%0.0f', delimiter = ',');

    # ?
    @Utilities.time_func
    def generate_euler_samples_settings(self) -> None:
        """
        _summary_

        _extended_summary_
        """

        global Input_2D
        global Output_2D_4_Connectivity
        global Output_2D_8_Connectivity

        # *
        Prediction = EulerNumberML2D(input = Input_2D, output = Output_2D_4_Connectivity, folder = self.__Folder);
        
        # *
        Remove_files = RemoveFiles(folder = self.__Folder)
        Remove_files.remove_all()

        for i in range(self.__Number_of_images):

            #Data_2D = np.random.randint(0, 2, (self._Height * self._Width))

            Euler_number = 0

            # * Initial probabilities values
            P_0 = 0.2
            P_1 = 0.8

            while(Euler_number != self.__Euler_number):

                # *
                Data_2D = np.random.choice(2, self.__Height * self.__Width, p = [P_0, P_1]);
                Data_2D = Data_2D.reshape(self.__Height, self.__Width);

                print(Data_2D);

                # *
                Data_2D_edges = np.zeros((Data_2D.shape[0] + 2, Data_2D.shape[1] + 2))
                
                print(Data_2D_edges);

                # *
                Data_2D_edges[1:Data_2D_edges.shape[0] - 1, 1:Data_2D_edges.shape[1] - 1] = Data_2D

                print(Data_2D_edges);
                print('\n');
                
                # *
                Array = Prediction.obtain_arrays_2D(Data_2D_edges);
                Euler_number = Prediction.model_prediction_2D(self.__Model_trained, Array);

                # *
                Image_name = "Image_2D_Real_Time_{}.png".format(j)
                Image_path = os.path.join(self.__Folder, Image_name)
                plt.title('P_0: {}, P_1: {}, Euler_number: {}'.format(P_0, P_1, Euler_number))
                plt.imshow(Data_2D_edges, cmap = 'gray', interpolation = 'nearest')
                plt.savefig(Image_path)

                # *
                if(Euler_number > self.__Euler_number):

                    if(P_0 != 0.98):

                        P_0 = P_0 - 0.02;
                        P_1 = P_1 + 0.02;

                else:
                    
                    if(P_1 != 0.98):
                        
                        P_0 = P_0 + 0.02;
                        P_1 = P_1 - 0.02;

            # *
            if(self.__Save_image):

                Image_name = "Image_with_euler_{}_2D.png".format(i)
                Image_path = os.path.join(self.__Folder, Image_name)

                plt.imshow(Data_2D_edges, cmap = 'gray', interpolation = 'nearest')
                plt.savefig(Image_path)
                #plt.show()
                plt.close()

            File_name = 'Image_with_euler_{}_2D.txt'.format(i);
            Path = os.path.join(self.__Folder, File_name);
            np.savetxt(Path, Data_2D_edges , fmt = '%0.0f', delimiter = ',');

# ? Extract Euler numbers from 3D arrays.
class EulerObjectGenerator(EulerGenerator):
    
    # ?
    @Utilities.time_func
    def generate_euler_samples_random(self, Prob_0: float = 0.2, Prob_1: float = 0.8) -> None:
        """
        _summary_

        _extended_summary_
        """

        # *
        Remove_files = RemoveFiles(folder = self.__Folder)
        Remove_files.remove_all()

        # *
        for i in range(self.__Number_of_images):

            # *
            #Data_3D = np.random.randint(0, 2, (self._Height * self._Depth * self._Width));
            Data_3D = np.random.choice(2, self.__Height * self.__Depth * self.__Width, p = [Prob_0, Prob_1]);
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

            #print(Data_3D_read);

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
                plt.title('P_0: {}, P_1: {}'.format(Prob_0, Prob_1))
                plt.imshow(Data_3D_edges[j], cmap = 'gray', interpolation = 'nearest')
                plt.savefig(Image_path)
                plt.close()

            File_name = 'Image_random_{}_3D.txt'.format(i);
            Path = os.path.join(self.__Folder, File_name);
            np.savetxt(Path, Data_3D_edges_complete, fmt = '%0.0f', delimiter = ',');

    # ?
    @Utilities.time_func
    def generate_euler_samples_settings(self) -> None:
        """
        _summary_

        _extended_summary_
        """
        DataFrame = pd.DataFrame()
        
        # *
        Prediction = EulerExtractor3D(input = Input_3D_array, output = Output_3D_array, folder = self.__Folder);

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