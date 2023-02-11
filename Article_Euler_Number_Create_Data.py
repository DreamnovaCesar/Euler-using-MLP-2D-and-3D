import os
import numpy as np
import pandas as pd

from Article_Euler_Number_Libraries import *
from Article_Euler_Number_Utilities import Utilities
from Article_Euler_Number_Remove_Files import RemoveFiles
from Article_Euler_Number_2D_And_3D_ML import EulerNumberML2D
from Article_Euler_Number_2D_And_3D_ML import EulerNumberML3D

from Article_Euler_Number_2D_General import Input_2D
from Article_Euler_Number_2D_General import Output_2D_4_Connectivity
from Article_Euler_Number_2D_General import Output_2D_8_Connectivity

from Article_Euler_Number_3D_General import Input_3D_array
from Article_Euler_Number_3D_General import Output_3D_array

from Article_Euler_Number_3D_General import *

import random

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

def get_octovoxel_3D(Object: str) -> list[np.ndarray]:
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
        Array_new = read_image_with_metadata_3D(Object)

        # * Creation of empty numpy arrays 3D

        Qs = table_binary_multi_256(256)
        Qs_value = np.zeros((256), dtype = 'int')

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

                    #Array_new[i:l + i, j:l + j, k:l + k]

                    # *
                    #Array_prediction[0] = Array_new[i + 1][j][k]
                    #Array_prediction[1] = Array_new[i + 1][j][k + 1]

                    #Array_prediction[2] = Array_new[i][j][k]
                    #Array_prediction[3] = Array_new[i][j][k + 1]

                    #Array_prediction[4] = Array_new[i + 1][j + 1][k]
                    #Array_prediction[5] = Array_new[i + 1][j + 1][k + 1]

                    #Array_prediction[6] = Array_new[i][j + 1][k]
                    #Array_prediction[7] = Array_new[i][j + 1][k + 1]
                    #print('\n')

                    for Index in range(len(Qs)):
                    
                        #print('Kernel: {}'.format(Array_new[i:l + i, j:l + j, k:l + k]))
                        #print('Qs: {}'.format(Qs[Index]))
                        #print('\n')
                        #print('\n')

                        if(np.array_equal(np.array(Array_new[i:l + i, j:l + j, k:l + k]), np.array(Qs[Index]))):
                            Qs_value[Index] += 1
                            print('Q{}_value: {}'.format(Index, Qs_value[Index]))

                    print(Qs_value)
                    print('\n')

        #           
        List_string = ''

        for i in range(256):
            List_string = List_string + str(Qs_value[i]) + ', '

        print('[{}]'.format(List_string))

        return Qs_value

# ?
class DataEuler(EulerNumberML2D, EulerNumberML3D):

    def __init__(self, **kwargs) -> None:
        """
        _summary_

        _extended_summary_
        """

        # * General parameters
        self.__Folder = kwargs.get('folder', None);
        self.__Number_of_images = kwargs.get('NI', None);
        
        self.__Number_of_images = int(self.__Number_of_images);

        # * Shape 2D and 3D
        self.__Height = kwargs.get('Height', 8);
        self.__Width = kwargs.get('Width', 8);

        # * 3D
        self.__Depth = kwargs.get('Depth', 4);

        # *
        self.__Save_image = kwargs.get('SI', True);

        # *
        self.__Euler_number = kwargs.get('EN', 1);

        # *
        self.__Model_trained = kwargs.get('MT', None);

        if(isinstance(self.__Height, str)):
            self.__Height = int(self.__Height)
        
        if(isinstance(self.__Width, str)):
            self.__Width = int(self.__Width)
        
        if(isinstance(self.__Depth, str)):
            self.__Depth = int(self.__Depth)

        if(isinstance(self.__Euler_number, str)):
            self.__Euler_number = int(self.__Euler_number)

    def __repr__(self):

        kwargs_info = "{}, {}, {}, {}, {}, {}".format(self.__Folder, self.__Number_of_images, self.__Height, self.__Width, self.__Depth, self.__Save_image);

        return kwargs_info

    def __str__(self):
        pass
    
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

    # ?
    @Utilities.time_func
    def create_data_euler_2D_random(self) -> None:
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
    def create_data_euler_2D_settings(self) -> None:
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

    # ?
    @Utilities.time_func
    def create_data_euler_3D_random(self) -> None:
        """
        _summary_

        _extended_summary_
        """

        P_0 = 0.2
        P_1 = 0.8

        # *
        Remove_files = RemoveFiles(folder = self.__Folder)
        Remove_files.remove_all()

        # *
        for i in range(self.__Number_of_images):

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
                plt.title('P_0: {}, P_1: {}'.format(P_0, P_1))
                plt.imshow(Data_3D_edges[j], cmap = 'gray', interpolation = 'nearest')
                plt.savefig(Image_path)
                plt.close()

            File_name = 'Image_random_{}_3D.txt'.format(i);
            Path = os.path.join(self.__Folder, File_name);
            np.savetxt(Path, Data_3D_edges_complete, fmt = '%0.0f', delimiter = ',');

    # ?
    @Utilities.time_func
    def create_data_euler_3D(self) -> None:
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

