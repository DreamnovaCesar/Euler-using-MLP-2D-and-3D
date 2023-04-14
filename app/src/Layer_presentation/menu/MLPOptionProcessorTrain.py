from .MenuOption import MenuOption
from ...Layer_application.MLPProcessor import MLPProcessor
from ...Layer_domain.Model.MLP import MLP
from ...Layer_domain.DataProcessor import DataProcessor

import os

class MLPOptionProcessorTrain(MenuOption):
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
        MLP_processor : MLPProcessor
    ):

        """
        Constructs a new DownloadRandomly object.
        """
        
        self.MLP_processor = MLP_processor;

    def execute(self):
        """
        Executes the DownloadRandomly option by prompting the user for the number of folders and images to download,
        and then downloading the random images using the MLP object.
        """

        os.system ("cls");

        print("Enter the following parameters to train the model using CSV and JSON:")
        print('\n')
        while True:

            user_exit = input("Do you want to go back? (Y/N)").strip().lower()
            print('\n');

            if user_exit == "n":

                self.CSV_file = input('CSV_file: ');
                self.JSON_file = input('JSON_file: ');
                self.Model_name = input("Model's name: ");
                self.Epochs = input("Epochs: ");
                self.Epochs = int(self.Epochs);

                print('\n');
                user_input = input("Do you want to change the inputs? (Y/N)").strip().lower()
                print('\n');

                if user_input == "n":
                    
                    MLP_processor = self.MLP_processor(
                        DataProcessor, 
                        MLP
                    );

                    MLP_processor.train(
                        self.CSV_file,
                        self.JSON_file,
                        self.Model_name,
                        self.Epochs
                    );
                
                elif user_input != "y":
                    print("Invalid input, please enter Y or N")
                    print('\n');
            
            elif user_exit == "y":
                break;
            
            elif user_exit != "y":
                print("Invalid input, please enter Y or N")
                print('\n');