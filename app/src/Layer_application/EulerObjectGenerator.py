import os
import random
import numpy as np

from .EulerGenerator import EulerGenerator
from ..Layer_domain.Convertion.BinaryStorageList import BinaryStorageList
from ..Layer_domain.Convertion.ConvertionDecimalBinaryByte import ConvertionDecimalBinaryByte
from ..Layer_domain.Arrays.OctovoxelHander import OctovoxelHandler
from ..Layer_domain.DataLoaderText import DataLoaderText
from ..Layer_domain.RemoveFiles.AllFileRemover import AllFileRemover
from ..Layer_domain.Model.MLP import MLP

from ..Layer_domain.SaverCSV import SaverCSV
from ..Layer_domain.SaverObjectsSettings import SaverObjectsSettings
from ..Layer_domain.SaverObjectsRandomly import SaverObjectsRandomly

from .ExtractorArrays import ExtractorArrays
from .ExtractorOctovoxels import ExtractorOctovoxels

from .MLPPredictionStandard import MLPPredictionStandard

from .GeneratorObject import GeneratorObject

class EulerObjectGenerator(EulerGenerator):
    """
    A class used for generating random 3D images and their combinations
    with octovoxels and Euler numbers.

    Attributes
    ----------
    _Folder_path : str
        The folder path where the images will be saved
    _Number_of_objects : int
        The number of objects to be generated
    _Height : int
        The height of the generated images
    _Width : int
        The width of the generated images
    _Model : str
        The path of the trained model file
    Depth : int
        The depth of the generated images

    Methods
    -------
    generate_euler_samples_random(Prob_0=0.2, Prob_1=0.8)
        Generate random 3D images and save them in the specified folder path.
    generate_euler_samples_settings()
        Generate 3D images with settings and save them in the specified folder path.
    """
    def __init__(
        self, 
        _Folder_path : str, 
        _Number_of_objects : int,
        _Height : int,
        _Width : int,
        _Model : str,
        Depth : int
    ) -> None:
        """
        Constructs all the necessary attributes for the EulerObjectGenerator object.

        Parameters
        ----------
            _Folder_path : str
                The folder path where the images will be saved
            _Number_of_objects : int
                The number of objects to be generated
            _Height : int
                The height of the generated images
            _Width : int
                The width of the generated images
            _Model : str
                The path of the trained model file
            Depth : int
                The depth of the generated images
        """

        super().__init__(
            _Folder_path, 
            _Number_of_objects, 
            _Height, 
            _Width, 
            _Model
        )

        self._Depth = Depth
        self._Depth = int(self._Depth)

    def generate_euler_samples_random(
            self, 
            Prob_0: float = 0.2, 
            Prob_1: float = 0.8
        ):

        """
        Generate random 3D images and save them in the specified folder path.

        Parameters
        ----------
        Prob_0 : float, optional
            The probability of the occurrence of pixel value 0, by default 0.2
        Prob_1 : float, optional
            The probability of the occurrence of pixel value 1, by default 0.8
        """

        # * Object to handle saving files
        Saver_objects = SaverObjectsRandomly()

        # * Remove any existing files from the folder path
        Remove_files = AllFileRemover(self._Folder_path)
        Remove_files.remove_files()

        # * Loop over the number of objects specified and create a 3D array for each one
        for i in range(self._Number_of_objects):
            
            # * Create a unique file name for each 3D array
            File_name = 'Image_random_{}_3D.txt'.format(i);
            Path = os.path.join(self._Folder_path, File_name);

            # * Generate a 3D array using the specified probabilities and dimensions
            Object, Object_plt = GeneratorObject.generator(
                Prob_0, 
                Prob_1,
                self._Width,
                self._Height,
                self._Depth
            )

            # * Save the 3D array to a text file
            np.savetxt(Path, Object, fmt = '%0.0f', delimiter = ',');

            # * Create a directory for the images of each 3D array
            Dir_name_images = "Images_random_{}_3D".format(i)
            Dir_data_images = '{}/{}'.format(self._Folder_path, Dir_name_images)
            Exist_dir_images = os.path.isdir(Dir_data_images)
            
            # * If the directory doesn't exist, create it and print its path
            if Exist_dir_images == False:
                Folder_path_images = os.path.join(self._Folder_path, Dir_name_images)
                os.mkdir(Folder_path_images)
                #print(Folder_path_images)
            else:
                Folder_path_images = os.path.join(self._Folder_path, Dir_name_images)
                #print(Folder_path_images)

            # * Save the images of each 3D array
            for j in range(self._Depth + 2):
        
                Saver_objects.save_file(
                    i, j,
                    Prob_0,
                    Prob_1,
                    Folder_path_images, 
                    Object,
                )
    
    def generate_euler_samples_settings(self):
        """
        Generate 3D images with theirs euler number and save them in the specified folder path.

        Parameters
        ----------
        Prob_0 : float, optional
            The probability of the occurrence of pixel value 0, by default 0.2
        Prob_1 : float, optional
            The probability of the occurrence of pixel value 1, by default 0.8
        """

        # Initialize MLP and extractor objects
        MLPPrediction = MLPPredictionStandard(ExtractorArrays,
                                              MLP);
        
        Extraction_octovoxels = ExtractorOctovoxels(BinaryStorageList,
                                                    ConvertionDecimalBinaryByte,
                                                    OctovoxelHandler,
                                                    DataLoaderText)

        # * Initialize Saver objects
        Saver_CSV = SaverCSV()
        Saver_objects = SaverObjectsSettings()
    
        # * Delete all files in the folder
        Remove_files = AllFileRemover(self._Folder_path);
        Remove_files.remove_files();

        # * Generate the specified number of objects
        for i in range(self._Number_of_objects):
            
            # * Define file path and randomly generate pixel probabilities
            File_name = 'Image_random_{}_3D.txt'.format(i);
            Object_path = os.path.join(self._Folder_path, File_name);

            Prob_0 = random.uniform(0, 1)
            Prob_1 = 1 - Prob_0

            # * Generate the 3D object and save it
            Object, Object_plt = GeneratorObject.generator(
                Prob_0, 
                Prob_1,
                self._Width,
                self._Height,
                self._Depth
            )

            # * Generate a 3D array using the specified probabilities and dimensions
            np.savetxt(Object_path, Object, fmt = '%0.0f', delimiter = ',');

            # * Predict the Euler number of the 3D object and extract octovoxels
            Euler_number = MLPPrediction.prediction(self._Model, Object_path);
            Combination_octovoxels = Extraction_octovoxels.extractor(Object_path);

            # * Save the combination of octovoxels and Euler number in CSV format
            Combination_octovoxels = np.append(Combination_octovoxels, Euler_number)
            Saver_CSV.save_file(r'app\data\3D\Data', Combination_octovoxels)

            # * Create folder to save images if it doesn't exist
            Dir_name_images = "Images_random_{}_3D".format(i)
            Dir_data_images = '{}/{}'.format(self._Folder_path, Dir_name_images)
            Exist_dir_images = os.path.isdir(Dir_data_images)
            
            # * If the directory doesn't exist, create it and print its path
            if(Exist_dir_images == False):
                Folder_path_images = os.path.join(self._Folder_path, Dir_name_images)
                os.mkdir(Folder_path_images)
                print(Folder_path_images)
            else:
                Folder_path_images = os.path.join(self._Folder_path, Dir_name_images)
                print(Folder_path_images)

            # * Save the images of each 3D array
            for j in range(self._Depth + 2):
                
                Saver_objects.save_file(
                    i, j,
                    Folder_path_images, 
                    Euler_number,
                    Object_plt,
                )
