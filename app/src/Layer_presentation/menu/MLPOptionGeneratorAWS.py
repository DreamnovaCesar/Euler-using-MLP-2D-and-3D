from .MenuOption import MenuOption
from ...Layer_application.EulerObjectGeneratorAWS import EulerObjectGeneratorAWS

class MLPOptionGeneratorAWS(MenuOption):
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
        Euler_generator_3D : EulerObjectGeneratorAWS
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
        
        self.Folder_path = input("Folder_path: ");
        self.Number_of_objects = input("Number_of_objects: ");
        self.Height = input("Height: ");
        self.Width = input("Width: ");
        self.Depth = input("Depth: ");
        self.Model = input("Model: ");
        
        Euler_generator_3D = self.Euler_generator_3D(
            self.Folder_path,
            self.Number_of_objects,
            self.Height,
            self.Width,
            self.Model,
            self.Depth
        );
        
        Euler_generator_3D.generate_euler_samples_settings();