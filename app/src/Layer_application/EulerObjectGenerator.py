import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .EulerGenerator import EulerGenerator
from ..Layer_domain.RemoveFiles.AllFileRemover import AllFileRemover

class EulerObjectGenerator(EulerGenerator):

    def __init__(self, 
                 Folder_path : str, 
                 Number_of_objects : int,
                 Height : int,
                 Width : int,
                 Depth : int) -> None:
        
        super().__init__(Folder_path, Number_of_objects, Height, Width)

        self._Depth = Depth
    
    def generate_euler_samples_random():
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
    
    def generate_euler_samples_settings():
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