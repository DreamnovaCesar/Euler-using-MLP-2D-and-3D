import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from abc import ABC
from abc import abstractmethod

from .EulerGenerator import EulerGenerator
from ..Layer_domain.RemoveFiles.AllFileRemover import AllFileRemover

class EulerGenerator(EulerGenerator):

    def __init__(self, 
                 _Folder_path : str, 
                 _Number_of_objects : int,
                 _Height : int,
                 _Width : int,
                 _Model : str) -> None:
        
        super().__init__(_Folder_path, _Number_of_objects, _Height, _Width, _Model)

    
    def generate_euler_samples_random(self, Prob_0: float = 0.2, Prob_1: float = 0.8):

        # *
        Remove_files = AllFileRemover(self._Folder_path)
        Remove_files.remove_files()

        for i in range(self._Number_of_objects):

            # *
            #Data_2D = np.random.randint(0, 2, (self._Height * self._Width))
            Data_2D = np.random.choice(2, self._Height * self._Width, p = [Prob_0, Prob_1]);
            Data_2D = Data_2D.reshape(self._Height, self._Width);

            #print(Data_2D);
            #print('\n');
            
            if(Save_image):

                Image_name = "Image_random_{}_2D.png".format(i)
                Image_path = os.path.join(self._Folder_path, Image_name)
                
                Data_2D_edges = np.zeros((Data_2D.shape[0] + 2, Data_2D.shape[1] + 2))

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
    
    def generate_euler_samples_settings(self):

        # *
        Prediction = EulerNumberML2D(input = Input_2D, output = Output_2D_4_Connectivity, folder = self.__Folder);
        
        # *
        Remove_files = AllFileRemover(self._Folder_path)
        Remove_files.remove_files()

        for i in range(self._Number_of_objects):

            #Data_2D = np.random.randint(0, 2, (self._Height * self._Width))

            Euler_number = 0

            # * Initial probabilities values
            P_0 = 0.2
            P_1 = 0.8

            while(Euler_number != self._Euler_number):

                # *
                Data_2D = np.random.choice(2, self._Height * self._Width, p = [P_0, P_1]);
                Data_2D = Data_2D.reshape(self._Height, self._Width);

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
                Image_path = os.path.join(self._Folder_path, Image_name)
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