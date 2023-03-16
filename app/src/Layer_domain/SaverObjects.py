import os
import matplotlib.pyplot as plt

from .Saver import Saver

class SaverObjects(Saver):

    def save_file(self, Folder_path: str, Euler_number : int, Data_3D_edges, i : int, j : int):
        # * 

        Image_name = "Image_slice_random_{}_{}_3D".format(i, j)
        Image_path = os.path.join(Folder_path, Image_name)
        #plt.title('P_0: {}, P_1: {}'.format(P_0, P_1))
        plt.title('Euler: {}'.format(Euler_number))
        plt.imshow(Data_3D_edges[j], cmap = 'gray', interpolation = 'nearest')
        plt.savefig(Image_path)
        plt.close()