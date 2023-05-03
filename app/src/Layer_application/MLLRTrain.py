import numpy as np
from .MLTrain import MLTrain
from ..Layer_domain.Model.LR import LogisticRegression
from ..Layer_domain.DataProcessor import DataProcessor
from ..Layer_domain.DataLoaderCSV import DataLoaderCSV
from ..Layer_domain.Model.ModelSaverML import ModelSaverML
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler

class MLLRTrain(MLTrain):
    """
    A class for machine learning training using the Logistic Regression algorithm.
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
        Data_processor : DataProcessor,
        LogisticRegression_training : LogisticRegression
    ):
        """
        Initializes the MLPStandard class.

        Parameters
        ----------
        MLP_training : MLP
            A multilayer perceptron (MLP) model to be trained.
        """

        self.Data_processor = Data_processor
        self.LogisticRegression_training = LogisticRegression_training;

    def train(
        self,
        CSV_file : str, 
        Model_name : str
    ):
        
        Processor = self.Data_processor(DataLoaderCSV);

        X, Y = Processor.process_data(CSV_file);

        Y = Y.ravel();
        
        Scaler = StandardScaler()
        X_scaled = Scaler.fit_transform(X)

        # * Instantiates an RF object to train
        LR = self.LogisticRegression_training(
            X, 
            Y
        );

        
        # * Fits the model to the input and target data
        Model = LR.fit_model();

        # * Fits the model to the input and target data
        Pred_Y = LR.predict_model(Model, X);

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

