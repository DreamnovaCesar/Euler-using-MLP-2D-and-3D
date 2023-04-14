from .MenuOption import MenuOption
from ...Layer_application.EulerObjectGenerator import EulerObjectGenerator

import os

class MLPOptionGenerator3D(MenuOption):
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
        Euler_generator_3D : EulerObjectGenerator
    ):

        """
        Constructs a new DownloadRandomly object.
        """
        
        self.Euler_generator_3D = Euler_generator_3D;

    def execute(self):
        """
        Executes the DownloadRandomly option by prompting the user for the number of folders and images to download,
        and then downloading the random images using the MLP object.
        """
        os.system ("cls");

        print("Enter the following parameters to generate the Euler object samples:")
        print('\n');

        while True:

            user_exit = input("Do you want to go back? (Y/N)").strip().lower()
            print('\n');
            
            if user_exit == "n":

                self.Folder_path = input("Folder path: ");
                self.Number_of_objects = input("Number_of_objects: ")
                self.Height = input("Height: ")
                self.Width = input("Width: ")
                self.Depth = input("Depth: ")
                self.Model = input("Model: ")
                
                print('\n');
                user_input = input("Do you want to change the inputs? (Y/N)").strip().lower()
                print('\n');

                if user_input == "n":
                    
                    Euler_generator_3D = self.Euler_generator_3D(
                        self.Folder_path,
                        self.Number_of_objects,
                        self.Height,
                        self.Width,
                        self.Model,
                        self.Depth
                    );

                    Euler_generator_3D.generate_euler_samples_settings();
                
                elif user_input != "y":
                    print("Invalid input, please enter Y or N")
                    print('\n');
            elif user_exit == "y":
                break;
            
            elif user_exit != "y":
                print("Invalid input, please enter Y or N")
                print('\n');

        
