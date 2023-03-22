from .MenuOption import MenuOption
from ...Layer_application.MLPStandard import MLPStandard
from ...Layer_domain.Model.MLP import MLP
from ...Utilities import General_Info_3D

class MLPOptionStandardTrain(MenuOption):
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
        MLP_standard : MLPStandard
    ):

        """
        Constructs a new DownloadRandomly object.
        """
        
        self.MLP_standard = MLP_standard;

    def execute(self):
        """
        Executes the DownloadRandomly option by prompting the user for the number of folders and images to download,
        and then downloading the random images using the MLP object.
        """
        
        self.JSON_file = input('JSON_file: ');
        self.Model_name = input("Model's name: ");
        self.Epochs = input("Epochs: ");

        self.Epochs = int(self.Epochs)

        MLP_standard = self.MLP_standard(MLP);

        MLP_standard.train(
            General_Info_3D._INPUT_3D_, 
            General_Info_3D._OUTPUT_3D_, 
            self.JSON_file, 
            self.Model_name,
            self.Epochs
        );