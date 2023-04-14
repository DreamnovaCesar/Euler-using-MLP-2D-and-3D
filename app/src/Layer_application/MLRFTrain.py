import numpy as np
from .MLTrain import MLTrain
from ..Layer_domain.Model.RF import RF
from ..Layer_domain.Model.ModelSaverML import ModelSaverML
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class MLRFTrain(MLTrain):
    """
    A class for machine learning training using the Random Forest algorithm.
    This class provides an implementation of the abstract base class MLTrain for training a machine learning
    model using the Random Forest algorithm.

    Notes
    -----
    This implementation uses the Random Forest algorithm which is an ensemble learning method for classification,
    regression and other tasks that operate by constructing a multitude of decision trees at training time and
    outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the
    individual trees.

    """
    def __init__(
        self, 
        RF_training : RF
    ):
        """
        Initializes the MLPStandard class.

        Parameters
        ----------
        MLP_training : MLP
            A multilayer perceptron (MLP) model to be trained.
        """

        self.RF_training = RF_training;

    def train(
        self,
        X : np.ndarray, 
        Y : np.ndarray, 
        JSON_file : str, 
        Model_name : str
    ):
        
        # * Instantiates an RF object to train
        RF = self.RF_training(
            X, 
            Y, 
            JSON_file, 
        );

        
        # * Fits the model to the input and target data
        Model = RF.fit_model();

        # * Fits the model to the input and target data
        Pred_Y = RF.predict_model(Model, X);

        # * Calculate accuracy of the model
        Accuracy = accuracy_score(Y, Pred_Y)

        # * Calculate confusion_matrix of the model
        Confusion_matrix = confusion_matrix(Y, Pred_Y)

        # * Print accuracy of the model
        print('Accuracy: {}'.format(Accuracy))
        print('\n')

        # * Print accuracy of the model
        print('Confusion_matrix: {}'.format(Confusion_matrix))
        print('\n')

        # * Save model to the folder
        ModelSaverML.save_model(Model, Model_name);

