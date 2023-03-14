__author__ = "Cesar Eduardo Munoz Chavez"
__license__ = "Feel free to copy"

# ? Import necessary modules
from .Layer_application.ExtractorPixels import ExtractorPixels
from .Layer_application.ExtractorOctovoxels import ExtractorOctovoxels

from .Layer_application.MLPStandard import MLPStandard
from .Layer_presentation.menu.MLPOption import MLPOption

from .Layer_presentation.menu.Menu import Menu

# ? Define the options for the menu
options = {

    "Train MLP" : MLPOption(MLPStandard)

};

# ? Create and display the menu
def main():
    menu = Menu(options);
    menu.display();

# ? If the script is being run directly, create and display the menu
if __name__ == "__main__":
    main();
