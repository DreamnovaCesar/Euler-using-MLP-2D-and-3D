import os
import random
import numpy as np

from .EulerGenerator import EulerGenerator
from ..Layer_domain.Convertion.BinaryStorageList import BinaryStorageList
from ..Layer_domain.Convertion.ConvertionDecimalBinaryNibble import ConvertionDecimalBinaryNibble
from ..Layer_domain.Arrays.PixelHander import PixelHandler
from ..Layer_domain.DataLoaderText import DataLoaderText
from ..Layer_domain.RemoveFiles.AllFileRemover import AllFileRemover
from ..Layer_domain.Model.MLP import MLP

from ..Layer_domain.SaverCSV import SaverCSV
from ..Layer_domain.SaverImagesSettings import SaverImagesSettings
from ..Layer_domain.SaverImagesRandomly import SaverImagesRandomly

from .ExtractorArrays import ExtractorArrays
from .ExtractorPixels import ExtractorPixels

from .MLPPredictionStandard import MLPPredictionStandard

from .GeneratorImage import GeneratorImage

class EulerImageGenerator(EulerGenerator):
    """
    Generates 2D binary images and their Euler numbers, and stores them in the specified folder path.

    Parameters
    ----------
    _Folder_path : str
        The path to the folder where the generated images and their data will be saved.
    _Number_of_objects : int
        The number of objects to be generated.
    _Height : int
        The height (number of rows) of the 2D image.
    _Width : int
        The width (number of columns) of the 2D image.
    _Model : str
        The path to the pre-trained MLP model that will be used for Euler number prediction.

    Attributes
    ----------
    _Folder_path : str
        The path to the folder where the generated images and their data will be saved.
    _Number_of_objects : int
        The number of objects to be generated.
    _Height : int
        The height (number of rows) of the 2D image.
    _Width : int
        The width (number of columns) of the 2D image.
    _Model : str
        The path to the pre-trained MLP model that will be used for Euler number prediction.

    Methods
    -------
    generate_euler_samples_random(Prob_0=0.2, Prob_1=0.8)
        Generates 2D binary images with random pixel probabilities and their Euler numbers, and stores them in the specified folder path.
    generate_euler_samples_settings()
        Generates 2D binary images with pixel probabilities generated from a normal distribution and their Euler numbers, and stores them in the specified folder path.
    """
    def __init__(self, 
                 _Folder_path : str, 
                 _Number_of_objects : int,
                 _Height : int,
                 _Width : int,
                 _Model : str) -> None:
        """
        Initializes EulerImageGenerator with the given parameters.

        Parameters
        ----------
        _Folder_path : str
            The path to the folder where the generated images and their data will be saved.
        _Number_of_objects : int
            The number of objects to be generated.
        _Height : int
            The height (number of rows) of the 2D image.
        _Width : int
            The width (number of columns) of the 2D image.
        _Model : str
            The path to the pre-trained MLP model that will be used for Euler number prediction.
        """

        super().__init__(
            _Folder_path, 
            _Number_of_objects, 
            _Height, 
            _Width, 
            _Model
        )

    
    def generate_euler_samples_random(
            self, 
            Prob_0: float = 0.2, 
            Prob_1: float = 0.8
        ):

        """
        Generate random 2D images and save them in the specified folder path.

        Parameters
        ----------
        Prob_0 : float, optional
            The probability of the occurrence of pixel value 0, by default 0.2
        Prob_1 : float, optional
            The probability of the occurrence of pixel value 1, by default 0.8
        """

        # * Remove any existing files from the folder path
        Remove_files = AllFileRemover(self._Folder_path);
        Remove_files.remove_files();

        # * Generate the specified number of objects
        for i in range(self._Number_of_objects):
            

            # * Create a unique file name for the 2D numpy array
            Image_name = "Image_random_{}_2D.txt".format(i)
            Image_path = os.path.join(self._Folder_path, Image_name)

            # * Generate a 2D array using the specified probabilities and dimensions
            Image = GeneratorImage.generator(
                Prob_0, 
                Prob_1,
                self._Height,
                self._Width
            );

            # * Save the 2D array to a text file
            np.savetxt(Image_path, Image, fmt = '%0.0f', delimiter = ',');

            # * Create a directory for the images of each 3D array
            Dir_name_images = "Images_random_{}_2D".format(i)
            Dir_data_images = '{}/{}'.format(self._Folder_path, Dir_name_images);
            Exist_dir_images = os.path.isdir(Dir_data_images);
            
            # * If the directory doesn't exist, create it and print its path
            if Exist_dir_images == False:
                Folder_path_images = os.path.join(self._Folder_path, Dir_name_images);
                os.mkdir(Folder_path_images);
                #print(Folder_path_images)
            else:
                Folder_path_images = os.path.join(self._Folder_path, Dir_name_images);
                #print(Folder_path_images)

    
    def generate_euler_samples_settings(self):
        """
        Generate 3D images with theirs euler number and save them in the specified folder path.
        """
        
        # * Initialize MLP and extractor objects
        MLPPrediction = MLPPredictionStandard(
            ExtractorArrays,
            MLP
        );
        
        Extraction_pixels = ExtractorPixels(
            BinaryStorageList,
            ConvertionDecimalBinaryNibble,
            PixelHandler,
            DataLoaderText
        );

        # * Initialize Saver objects
        Saver_CSV = SaverCSV()
        Saver_objects = SaverImagesSettings()
        
        # * Delete all files in the folder
        Remove_files = AllFileRemover(self._Folder_path)
        Remove_files.remove_files()

        # * Generate the specified number of objects
        for i in range(self._Number_of_objects):


            # * Define file path and randomly generate pixel probabilities
            File_name = 'Image_random_{}_2D.txt'.format(i);
            Image_path = os.path.join(self._Folder_path, File_name);

            Prob_0 = random.uniform(0, 1)
            Prob_1 = 1 - Prob_0

            # * Generate the 2D object and save it
            Image = GeneratorImage.generator(
                Prob_0, 
                Prob_1,
                self._Width,
                self._Height,
            )

            # * Generate a 3D array using the specified probabilities and dimensions
            np.savetxt(Image_path, Image, fmt = '%0.0f', delimiter = ',');

            # * Predict the Euler number of the 3D object and extract octovoxels
            Euler_number = MLPPrediction.prediction(self._Model, Image_path);
            Combination_pixels = Extraction_pixels.extractor(Image_path);

            # * Save the combination of octovoxels and Euler number in CSV format
            Combination_pixels = np.append(Combination_pixels, Euler_number)
            Saver_CSV.save_file(r'app\data\2D\Data', Combination_pixels)

            # * Create folder to save images if it doesn't exist
            Dir_name_images = "Images_random_{}_2D".format(i)
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
            
            # * Save the images 2D array
            Saver_objects.save_file(
                i,
                Folder_path_images, 
                Euler_number,
                Image,
            )