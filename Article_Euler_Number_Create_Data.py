from Article_Euler_Number_Libraries import *
from Article_Euler_Number_Utilities import Utilities

# ?
class DataEuler(Utilities):

    def __init__(self, **kwargs) -> None:
        """
        _summary_

        _extended_summary_
        """

        # * General parameters
        self._Folder = kwargs.get('folder', None);
        self._Number_of_images = kwargs.get('NI', None);

        # * Shape 2D and 3D
        self._Height = kwargs.get('Height', 8);
        self._Width = kwargs.get('Width', 8);
        # * 3D
        self._Depth = kwargs.get('Depth', 4);

    def __repr__(self):

        kwargs_info = "{}, {}".format(self._Folder , self._Number_of_images);

        return kwargs_info

    def __str__(self):
        pass
    
    # * _Folder attribute
    @property
    def _Folder_property(self):
        return self._Folder

    @_Folder_property.setter
    def _Folder_property(self, New_value):
        print("Changing folder...");
        self._Folder = New_value;
    
    @_Folder_property.deleter
    def _Folder_property(self):
        print("Deleting folder...");
        del self._Folder

    # * _Number_of_images attribute
    @property
    def _Number_of_images_property(self):
        return self._Number_of_images

    @_Number_of_images_property.setter
    def _Number_of_images_property(self, New_value):
        print("Changing number of images...");
        self._Number_of_images = New_value;
    
    @_Number_of_images_property.deleter
    def _Number_of_images_property(self):
        print("Deleting number of images...");
        del self._Number_of_images

    # ?
    @Utilities.time_func
    def create_data_euler_2D(self) -> None:
        
        for i in range(self._Number_of_images):

            Data_2D = np.random.randint(0, 2, (self._Height * self._Width))
            Data_2D = Data_2D.reshape(self._Height, self._Width);

            print(Data_2D);
            print('\n');

            File_name = 'Image_2D_{}.txt'.format(i);
            Path = os.path.join(self._Folder, File_name);
            np.savetxt(Path, Data_2D , fmt = '%0.0f', delimiter = ',');

    # ?
    @Utilities.time_func
    def create_data_euler_3D(self) -> None:
        
        for i in range(self._Number_of_images):

            Data_3D = np.random.randint(0, 2, (self._Height * self._Depth * self._Width));
            Data_3D = Data_3D.reshape((self._Height * self._Depth), self._Width);
            Data_3D_plot = Data_3D.reshape((self._Height, self._Depth, self._Width));

            print(Data_3D_plot);
            print('\n');

            File_name = 'Image_3D_{}.txt'.format(i);
            Path = os.path.join(self._Folder, File_name);
            np.savetxt(Path, Data_3D , fmt = '%0.0f', delimiter = ',');
