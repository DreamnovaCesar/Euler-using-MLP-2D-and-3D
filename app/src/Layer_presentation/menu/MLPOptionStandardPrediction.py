from .MenuOption import MenuOption
from ...Layer_application.MLPPredictionStandard import MLPPredictionStandard
from ...Layer_application.ExtractorArrays import ExtractorArrays
from ...Layer_domain.Model.MLP import MLP

import os

class MLPOptionStandardPrediction(MenuOption):
    """
    A MenuOption class that allows the user to download random images of girls.

    Attributes:
    -----------
    MLP : DownloadGirlsRandom
        The MLP object used to download the images.
    
    Methods
    -------
    execute()
        Prompts the user to input a path to a number of folders and the Nnumber of images to download. 
    """

    def __init__(
        self, 
        MLP_prediction_standard : MLPPredictionStandard
    ):

        """
        Constructs a new DownloadRandomly object.
        """
        
        self.MLP_prediction_standard = MLP_prediction_standard;

    def execute(self):
        """
        Executes the DownloadRandomly option by prompting the user for the number of folders and images to download,
        and then downloading the random images using the MLP object.
        """

        os.system ("cls");

        print("Enter the following parameters to predict the object:")

        print('\n');
        while True:

            user_exit = input("Do you want to go back? (Y/N)").strip().lower()
            print('\n');
            
            if user_exit == "n":

                self.Model = input("Model: ");
                self.Object = input('Object: ');

                print('\n');
                user_input = input("Do you want to change the inputs? (Y/N)").strip().lower()
                print('\n');

                if user_input == "n":
                    
                    MLP_prediction_standard = self.MLP_prediction_standard(
                        ExtractorArrays,
                        MLP
                    );

                    MLP_prediction_standard.prediction(
                        self.Model, 
                        self.Object
                    );
                elif user_input != "y":
                    print("Invalid input, please enter Y or N")
                    print('\n');
            
            elif user_exit == "y":
                break;

            elif user_exit != "y":
                print("Invalid input, please enter Y or N")
                print('\n');