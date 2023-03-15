__author__ = "Cesar Eduardo Munoz Chavez"
__license__ = "Feel free to copy"

# ? Import necessary modules
from .Layer_application.ExtractorPixels import ExtractorPixels
from .Layer_application.ExtractorOctovoxels import ExtractorOctovoxels

from .Layer_application.MLPStandard import MLPStandard
from .Layer_application.MLPProcessor import MLPProcessor
from .Layer_application.MLPPredictionRegression import MLPPredictionRegression
from .Layer_presentation.menu.MLPOptionStandardTrain import MLPOptionStandardTrain
from .Layer_presentation.menu.MLPOptionProcessorTrain import MLPOptionProcessorTrain
from .Layer_presentation.menu.MLPOptionPrediction import MLPOptionPrediction

from .Layer_presentation.menu.Menu import Menu

# ? Define the options for the menu
options = {

    "MLP Train Standard" : MLPOptionStandardTrain(MLPStandard),
    "MLP Train Processor" : MLPOptionProcessorTrain(MLPProcessor),
    "MLP Prediction Processor" : MLPOptionPrediction(MLPPredictionRegression)

};

# ? Create and display the menu
def main():
    menu = Menu(options);
    menu.display();

# ? If the script is being run directly, create and display the menu
if __name__ == "__main__":
    main();
