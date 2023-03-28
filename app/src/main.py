__author__ = "Cesar Eduardo Munoz Chavez"
__license__ = "Feel free to copy"

# ? Import necessary modules
from .Layer_application.ExtractorPixels import ExtractorPixels
from .Layer_application.ExtractorOctovoxels import ExtractorOctovoxels

from .Layer_application.MLPStandard import MLPStandard
from .Layer_application.MLPProcessor import MLPProcessor
from .Layer_application.MLPPredictionRegression import MLPPredictionRegression
from .Layer_application.MLPPredictionStandard import MLPPredictionStandard
from .Layer_application.EulerObjectGenerator import EulerObjectGenerator
from .Layer_application.MLRFTrain import MLRFTrain

from .Layer_presentation.menu.MLPOptionStandardTrain import MLPOptionStandardTrain
from .Layer_presentation.menu.MLPOptionProcessorTrain import MLPOptionProcessorTrain
from .Layer_presentation.menu.MLPOptionStandardPrediction import MLPOptionStandardPrediction
from .Layer_presentation.menu.MLPOptionPrediction import MLPOptionPrediction
from .Layer_presentation.menu.MLPOptionGenerator3D import MLPOptionGenerator3D
from .Layer_presentation.menu.MLPOptionRFTrain import MLPOptionRFTrain

from .Layer_presentation.menu.Menu import Menu

# ? Define the options for the menu
options = {

    "MLP Train Standard" : MLPOptionStandardTrain(MLPStandard),
    "MLP Train Processor" : MLPOptionProcessorTrain(MLPProcessor),
    "MLP Prediction Standard" : MLPOptionStandardPrediction(MLPPredictionStandard),
    "MLP Prediction Processor" : MLPOptionPrediction(MLPPredictionRegression),
    "Euler Generator" : MLPOptionGenerator3D(EulerObjectGenerator),
    "RF Train" : MLPOptionRFTrain(MLRFTrain)
    
};

# ? Create and display the menu
def main():
    menu = Menu(options);
    menu.display();

# ? If the script is being run directly, create and display the menu
if __name__ == "__main__":
    main();
