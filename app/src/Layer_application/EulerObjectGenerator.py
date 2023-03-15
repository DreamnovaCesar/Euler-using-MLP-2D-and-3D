import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .EulerGenerator import EulerGenerator
from ..Layer_domain.Convertion.BinaryStorageList import BinaryStorageList
from ..Layer_domain.Convertion.ConvertionDecimalBinaryByte import ConvertionDecimalBinaryByte
from ..Layer_domain.Arrays.OctovoxelHander import OctovoxelHandler
from ..Layer_domain.TextDataLoader import TextDataLoader 
from ..Layer_domain.RemoveFiles.AllFileRemover import AllFileRemover
from ..Layer_domain.Model.MLP import MLP

from .ExtractorArrays import ExtractorArrays
from .ExtractorOctovoxels import ExtractorOctovoxels

from .MLPPredictionStandard import MLPPredictionStandard

class EulerObjectGenerator(EulerGenerator):

    def __init__(self, 
                 _Folder_path : str, 
                 _Number_of_objects : int,
                 _Height : int,
                 _Width : int,
                 _Model : str,
                 Depth : int
                 ) -> None:
        
        super().__init__(_Folder_path, _Number_of_objects, _Height, _Width, _Model)

        self._Depth = Depth
        self._Depth = int(self._Depth)

    def generate_euler_samples_random(self, Prob_0: float = 0.2, Prob_1: float = 0.8):
        """
        _summary_

        _extended_summary_
        """

        # *
        Remove_files = AllFileRemover(self._Folder_path)
        Remove_files.remove_files()

        # *
        for i in range(self._Number_of_objects):

            # *
            #Data_3D = np.random.randint(0, 2, (self._Height * self._Depth * self._Width));
            Data_3D = np.random.choice(2, self._Height * self._Depth * self._Width, p = [Prob_0, Prob_1]);
            Data_3D = Data_3D.reshape((self._Height * self._Depth), self._Width);
            Data_3D_plot = Data_3D.reshape((self._Height, self._Depth, self._Width));

            # *
            Data_3D_edges_complete = np.zeros((Data_3D_plot.shape[1] + 2, Data_3D_plot.shape[2] + 2))
            Data_3D_edges_concatenate = np.zeros((Data_3D_plot.shape[1] + 2, Data_3D_plot.shape[2] + 2))
            Data_3D_read = np.zeros((Data_3D.shape[0] + 2, Data_3D.shape[1] + 2))
            
            # * 
            Data_3D_edges = np.zeros((Data_3D_plot.shape[0] + 2, Data_3D_plot.shape[1] + 2, Data_3D_plot.shape[2] + 2))
            
            # * Get 3D image and interpretation of 3D from 2D .txt
            Data_3D_read[1:Data_3D_read.shape[0] - 1, 1:Data_3D_read.shape[1] - 1] = Data_3D
            Data_3D_edges[1:Data_3D_edges.shape[0] - 1, 1:Data_3D_edges.shape[1] - 1, 1:Data_3D_edges.shape[2] - 1] = Data_3D_plot

            # * Concatenate np.zeros
            Data_3D_read = np.concatenate((Data_3D_edges_concatenate, Data_3D_read), axis = 0)
            Data_3D_read = np.concatenate((Data_3D_read, Data_3D_edges_concatenate), axis = 0)

            for k in range(len(Data_3D_edges) - 2):
                Data_3D_edges_complete = np.concatenate((Data_3D_edges_complete, Data_3D_edges[k + 1]), axis = 0)

            Data_3D_edges_complete = np.concatenate((Data_3D_edges_complete, Data_3D_edges_concatenate), axis = 0)

            for j in range(self._Depth + 2):
                
                # *
                Dir_name_images = "Images_random_{}_3D".format(i)

                # *
                Dir_data_images = self._Folder_path + '/' + Dir_name_images

                # *
                Exist_dir_images = os.path.isdir(Dir_data_images)
                
                if Exist_dir_images == False:
                    Folder_path_images = os.path.join(self._Folder_path, Dir_name_images)
                    os.mkdir(Folder_path_images)
                    print(Folder_path_images)
                else:
                    Folder_path_images = os.path.join(self._Folder_path, Dir_name_images)
                    print(Folder_path_images)

                Image_name = "Image_slice_random_{}_{}_3D".format(i, j)
                Image_path = os.path.join(Folder_path_images, Image_name)
                #plt.title('P_0: {}, P_1: {}'.format(P_0, P_1))
                plt.title('P_0: {}, P_1: {}'.format(Prob_0, Prob_1))
                plt.imshow(Data_3D_edges[j], cmap = 'gray', interpolation = 'nearest')
                plt.savefig(Image_path)
                plt.close()

            File_name = 'Image_random_{}_3D.txt'.format(i);
            Path = os.path.join(self._Folder_path, File_name);
            np.savetxt(Path, Data_3D_edges_complete, fmt = '%0.0f', delimiter = ',');
    
    def generate_euler_samples_settings(self):
        """
        _summary_

        _extended_summary_
        """
        DataFrame = pd.DataFrame()
        
        # *
        MLPPrediction = MLPPredictionStandard(ExtractorArrays,
                                              MLP);
        
        Extraction_octovoxels = ExtractorOctovoxels(BinaryStorageList,
                                                    ConvertionDecimalBinaryByte,
                                                    OctovoxelHandler,
                                                    TextDataLoader)

        # *
        Remove_files = AllFileRemover(self._Folder_path);
        Remove_files.remove_files();

        # *
        for i in range(self._Number_of_objects):
            
            P_0 = random.uniform(0, 1)
            P_1 = 1 - P_0

            # *
            #Data_3D = np.random.randint(0, 2, (self._Height * self._Depth * self._Width));
            Data_3D = np.random.choice(2, self._Height * self._Depth * self._Width, p = [P_0, P_1]);
            Data_3D = Data_3D.reshape((self._Height * self._Depth), self._Width);
            Data_3D_plot = Data_3D.reshape((self._Height, self._Depth, self._Width));

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

            File_name = 'Image_random_{}_3D.txt'.format(i);
            Object_path = os.path.join(self._Folder_path, File_name);

            np.savetxt(Object_path, Data_3D_edges_complete, fmt = '%0.0f', delimiter = ',');

            Euler_number = MLPPrediction.prediction(self._Model, Object_path);
            Combination_octovoxels = Extraction_octovoxels.extractor(Object_path);

            print(Combination_octovoxels)
            print('///// ' + str(Euler_number))

            Combination_octovoxels = np.append(Combination_octovoxels, Euler_number)

            print(Combination_octovoxels)

            # * Return the new dataframe with the new data
            DataFrame = DataFrame.append(pd.Series(Combination_octovoxels), ignore_index = True)
                
            Dataframe_name = 'Dataframe_test_1.csv'.format()
            Dataframe_folder = os.path.join(r'app\data\3D\Data', Dataframe_name)
            DataFrame.to_csv(Dataframe_folder)

            #Array = Prediction.obtain_arrays_3D(Data_3D_edges);
            #Euler_number = Prediction.model_prediction_3D(self.__Model_trained, Array);

            #print(Data_3D_read);

            # * 
            for j in range(self._Depth + 2):
                
                # *
                Dir_name_images = "Images_random_{}_3D".format(i)

                # *
                Dir_data_images = self._Folder_path + '/' + Dir_name_images

                # *
                Exist_dir_images = os.path.isdir(Dir_data_images)
                
                if Exist_dir_images == False:
                    Folder_path_images = os.path.join(self._Folder_path, Dir_name_images)
                    os.mkdir(Folder_path_images)
                    print(Folder_path_images)
                else:
                    Folder_path_images = os.path.join(self._Folder_path, Dir_name_images)
                    print(Folder_path_images)

                Image_name = "Image_slice_random_{}_{}_3D".format(i, j)
                Image_path = os.path.join(Folder_path_images, Image_name)
                #plt.title('P_0: {}, P_1: {}'.format(P_0, P_1))
                plt.title('Euler: {}'.format(Euler_number))
                plt.imshow(Data_3D_edges[j], cmap = 'gray', interpolation = 'nearest')
                plt.savefig(Image_path)
                plt.close()
