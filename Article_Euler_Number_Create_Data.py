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
        self.__Folder = kwargs.get('folder', None);
        self.__Number_of_images = kwargs.get('NI', None);

        # * Shape 2D and 3D
        self.__Height = kwargs.get('Height', 8);
        self.__Width = kwargs.get('Width', 8);

        # * 3D
        self.__Depth = kwargs.get('Depth', 4);

        # *
        self.__Save_image = kwargs.get('SI', True);

        # *
        self.__Euler_number = kwargs.get('EN', True);

    def __repr__(self):

        kwargs_info = "{}, {}, {}, {}, {}, {}".format(self.__Folder, self.__Number_of_images, self.__Height, self.__Width, self.__Depth, self.__Save_image);

        return kwargs_info

    def __str__(self):
        pass
    
    # * _Folder attribute
    @property
    def __Folder_property(self):
        return self._Folder

    @__Folder_property.setter
    def __Folder_property(self, New_value):
        print("Changing folder...");
        self.__Folder = New_value;
    
    @__Folder_property.deleter
    def __Folder_property(self):
        print("Deleting folder...");
        del self.__Folder

    # * _Number_of_images attribute
    @property
    def __Number_of_images_property(self):
        return self.__Number_of_images

    @__Number_of_images_property.setter
    def __Number_of_images_property(self, New_value):
        print("Changing number of images...");
        self.__Number_of_images = New_value;
    
    @__Number_of_images_property.deleter
    def __Number_of_images_property(self):
        print("Deleting number of images...");
        del self.__Number_of_images

    # * _Number_of_images attribute
    @property
    def __Number_of_images_property(self):
        return self.__Number_of_images

    @__Number_of_images_property.setter
    def __Number_of_images_property(self, New_value):
        print("Changing number of images...");
        self.__Number_of_images = New_value;
    
    @__Number_of_images_property.deleter
    def __Number_of_images_property(self):
        print("Deleting number of images...");
        del self.__Number_of_images

    # * __Save_image attribute
    @property
    def __Save_image_property(self):
        return self.__Save_image

    @__Save_image_property.setter
    def __Save_image_property(self, New_value):
        print("Changing number of images...");
        self.__Save_image = New_value;
    
    @__Save_image_property.deleter
    def __Save_image_property(self):
        print("Deleting number of images...");
        del self.__Save_image

    # ?
    @Utilities.time_func
    def create_data_euler_2D_random(self) -> None:
        
        for i in range(self.__Number_of_images):

            #Data_2D = np.random.randint(0, 2, (self._Height * self._Width))
            Data_2D = np.random.choice(2, self.__Height * self.__Width, p = [0.2, 0.8]);
            Data_2D = Data_2D.reshape(self.__Height, self.__Width);

            print(Data_2D);
            print('\n');
            
            if(self.__Save_image):

                Image_name = "Image_2D_{}.png".format(i)
                Image_path = os.path.join(self.__Folder, Image_name)

                plt.imshow(Data_2D, cmap = 'gray', interpolation = 'nearest')
                plt.show()
                plt.savefig(Image_path)

            File_name = 'Image_2D_{}.txt'.format(i);
            Path = os.path.join(self.__Folder, File_name);
            np.savetxt(Path, Data_2D , fmt = '%0.0f', delimiter = ',');

    # ?
    @Utilities.time_func
    def create_data_euler_2D_settings(self) -> None:
        
        for i in range(self.__Number_of_images):

            #Data_2D = np.random.randint(0, 2, (self._Height * self._Width))
            Data_2D = np.random.choice(2, self.__Height * self.__Width, p = [0.2, 0.8]);
            Data_2D = Data_2D.reshape(self.__Height, self.__Width);

            print(Data_2D);
            print('\n');
            
            if(self.__Save_image):

                Image_name = "Image_2D_{}.png".format(i)
                Image_path = os.path.join(self.__Folder, Image_name)

                plt.imshow(Data_2D, cmap = 'gray', interpolation = 'nearest')
                plt.show()
                plt.savefig(Image_path)

            File_name = 'Image_2D_{}.txt'.format(i);
            Path = os.path.join(self.__Folder, File_name);
            np.savetxt(Path, Data_2D , fmt = '%0.0f', delimiter = ',');

    # ?
    @Utilities.time_func
    def create_data_euler_3D_random(self) -> None:
        
        for i in range(self.__Number_of_images):

            #Data_3D = np.random.randint(0, 2, (self._Height * self._Depth * self._Width));
            Data_3D = np.random.choice(2, self.__Height * self.__Depth * self.__Width, p = [0.2, 0.8]);
            Data_3D = Data_3D.reshape((self.__Height * self.__Depth), self.__Width);
            Data_3D_plot = Data_3D.reshape((self.__Height, self.__Depth, self.__Width));

            print(Data_3D_plot);
            print('\n');

            File_name = 'Image_3D_{}.txt'.format(i);
            Path = os.path.join(self.__Folder, File_name);
            np.savetxt(Path, Data_3D , fmt = '%0.0f', delimiter = ',');

    # ?
    @Utilities.time_func
    def create_data_euler_3D_settings(self) -> None:
        
        for i in range(self.__Number_of_images):

            #Data_3D = np.random.randint(0, 2, (self._Height * self._Depth * self._Width));
            Data_3D = np.random.choice(2, self.__Height * self.__Depth * self.__Width, p = [0.2, 0.8]);
            Data_3D = Data_3D.reshape((self.__Height * self.__Depth), self.__Width);
            Data_3D_plot = Data_3D.reshape((self.__Height, self.__Depth, self.__Width));

            print(Data_3D_plot);
            print('\n');

            File_name = 'Image_3D_{}.txt'.format(i);
            Path = os.path.join(self.__Folder, File_name);
            np.savetxt(Path, Data_3D , fmt = '%0.0f', delimiter = ',');