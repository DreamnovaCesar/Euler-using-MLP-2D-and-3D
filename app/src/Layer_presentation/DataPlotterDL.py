import os
import matplotlib.pyplot as plt

from .DataPlotter import DataPlotter

class DataPlotter(DataPlotter):
    
    def plot_data_loss(self, Hist_data: object) -> None:
        """
        Method to plot the loss. plot_data_loss takes the history 
        of the training process and plots the loss of the model as 
        a function of the number of epochs. 

        Parameters
        ----------
        Hist_data : object 
            History data to be plotted (Loss extraction).

        """

        # * Create a figure with size of 8x8
        plt.figure(figsize = (8, 8))
        plt.title('Training loss')
        plt.xlabel ("# Epoch")
        plt.ylabel ("# Loss")
        plt.ylim([0, 2.0])
        plt.plot(Hist_data.history["loss"])
        #plt.show()
        plt.close()

        # * Set the name of the figure to be saved and set the folder to save the figure
        Figure_name = "Figure_Loss_{}.png".format(self._Model_name)
        Figure_name_folder = os.path.join(self._Folder_data, Figure_name)

        # * Save the figure in the specified folder
        plt.savefig(Figure_name_folder)
        plt.close()

    def plot_data_accuracy(self, Hist_data: object) -> None:
        """
        Method to plot the accuracy. plot_data_accuracy also takes the history 
        of the training process as an argument and plots the accuracy of the model as 
        a function of the number of epochs.

        Parameters
        ----------
        Hist_data : object 
            History data to be plotted (Accuracy extraction).
        """

        # * Create a figure with size of 8x8
        plt.figure(figsize = (8, 8))
        plt.title('Training accuracy')
        plt.xlabel ("# Epoch")
        plt.ylabel ("# Acuracy")
        plt.ylim([0, 1])
        plt.plot(Hist_data.history["accuracy"])
        #plt.show()

        # * Set the name of the figure to be saved and set the folder to save the figure
        Figure_name = "Figure_Accuracy_{}.png".format(self._Model_name)
        Figure_name_folder = os.path.join(self._Folder_data, Figure_name)

        # * Save the figure in the specified folder
        plt.savefig(Figure_name_folder)
        plt.close()
