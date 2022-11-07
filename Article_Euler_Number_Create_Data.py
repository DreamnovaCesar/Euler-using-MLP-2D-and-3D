from Article_Euler_Number_Libraries import *
from Article_Euler_Number_Utilities import Utilities
from Article_Euler_Number_2D_And_3D_ML import EulerNumberML2D
from Article_Euler_Number_2D_And_3D_ML import EulerNumberML3D

from Article_Euler_Number_2D_General import Input_2D
from Article_Euler_Number_2D_General import Output_2D_4_Connectivity
from Article_Euler_Number_2D_General import Output_2D_8_Connectivity

from Article_Euler_Number_3D_General import Input_3D_array
from Article_Euler_Number_3D_General import Output_3D_array

# ?
class DataEuler(EulerNumberML2D, EulerNumberML3D):

    def __init__(self, **kwargs) -> None:
        """
        _summary_

        _extended_summary_
        """

        # * General parameters
        self.__Folder = kwargs.get('folder', None);
        self.__Number_of_images = kwargs.get('NI', None);
        
        self.__Number_of_images = int(self.__Number_of_images);

        # * Shape 2D and 3D
        self.__Height = kwargs.get('Height', 8);
        self.__Width = kwargs.get('Width', 8);

        # * 3D
        self.__Depth = kwargs.get('Depth', 4);

        # *
        self.__Save_image = kwargs.get('SI', True);

        # *
        self.__Euler_number = kwargs.get('EN', 1);

        # *
        self.__Model_trained = kwargs.get('MT', None);

        if(isinstance(self.__Height, str)):
            self.__Height = int(self.__Height)
        
        if(isinstance(self.__Width, str)):
            self.__Width = int(self.__Width)
        
        if(isinstance(self.__Depth, str)):
            self.__Depth = int(self.__Depth)

        if(isinstance(self.__Euler_number, str)):
            self.__Euler_number = int(self.__Euler_number)

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
        
        P_0 = 0.2
        P_1 = 0.8

        for i in range(self.__Number_of_images):

            # *
            #Data_2D = np.random.randint(0, 2, (self._Height * self._Width))
            Data_2D = np.random.choice(2, self.__Height * self.__Width, p = [P_0, P_1]);
            Data_2D = Data_2D.reshape(self.__Height, self.__Width);

            #print(Data_2D);
            #print('\n');
            
            if(self.__Save_image):

                # *
                Image_name = "Image_2D_{}.png".format(i)
                Image_path = os.path.join(self.__Folder, Image_name)
                
                # *
                Data_2D_edges = np.zeros((Data_2D.shape[0] + 2, Data_2D.shape[1] + 2))
                
                #print(Data_2D_edges);

                # *
                Data_2D_edges[1:Data_2D_edges.shape[0] - 1, 1:Data_2D_edges.shape[1] - 1] = Data_2D
                print(Data_2D_edges);
                plt.title('P_0: {}, P_1: {}'.format(P_0, P_1))
                plt.imshow(Data_2D_edges, cmap = 'gray', interpolation = 'nearest')
                plt.savefig(Image_path);
                #plt.show()

            # *
            File_name = 'Image_2D_{}.txt'.format(i);
            Path = os.path.join(self.__Folder, File_name);
            np.savetxt(Path, Data_2D_edges, fmt = '%0.0f', delimiter = ',');

    # ?
    @Utilities.time_func
    def create_data_euler_2D_settings(self) -> None:
        
        global Input_2D
        global Output_2D_4_Connectivity
        global Output_2D_8_Connectivity

        Prediction = EulerNumberML2D(input = Input_2D, output = Output_2D_4_Connectivity, folder = self.__Folder);

        for i in range(self.__Number_of_images):

            #Data_2D = np.random.randint(0, 2, (self._Height * self._Width))

            Euler_number = 0

            # * Initial probabilities values
            P_0 = 0.2
            P_1 = 0.8

            while(Euler_number != self.__Euler_number):

                # *
                Data_2D = np.random.choice(2, self.__Height * self.__Width, p = [P_0, P_1]);
                Data_2D = Data_2D.reshape(self.__Height, self.__Width);

                print(Data_2D);

                # *
                Data_2D_edges = np.zeros((Data_2D.shape[0] + 2, Data_2D.shape[1] + 2))
                
                print(Data_2D_edges);

                # *
                Data_2D_edges[1:Data_2D_edges.shape[0] - 1, 1:Data_2D_edges.shape[1] - 1] = Data_2D

                print(Data_2D_edges);
                print('\n');
                
                # *
                Array = Prediction.obtain_arrays_2D(Data_2D_edges);
                Euler_number = Prediction.model_prediction_2D(self.__Model_trained, Array);

                # *
                Image_name = "Image_2D_Real_Time_{}.png".format(j)
                Image_path = os.path.join(self.__Folder, Image_name)
                plt.title('P_0: {}, P_1: {}, Euler_number: {}'.format(P_0, P_1, Euler_number))
                plt.imshow(Data_2D_edges, cmap = 'gray', interpolation = 'nearest')
                plt.savefig(Image_path)

                # *
                if(Euler_number > self.__Euler_number):

                    if(P_0 != 0.98):

                        P_0 = P_0 - 0.02;
                        P_1 = P_1 + 0.02;

                else:
                    
                    if(P_1 != 0.98):
                        
                        P_0 = P_0 + 0.02;
                        P_1 = P_1 - 0.02;

            # *
            if(self.__Save_image):

                Image_name = "Image_2D_{}.png".format(i)
                Image_path = os.path.join(self.__Folder, Image_name)

                plt.imshow(Data_2D_edges, cmap = 'gray', interpolation = 'nearest')
                plt.savefig(Image_path)
                #plt.show()

            File_name = 'Image_2D_{}.txt'.format(i);
            Path = os.path.join(self.__Folder, File_name);
            np.savetxt(Path, Data_2D_edges , fmt = '%0.0f', delimiter = ',');

    # ?
    @Utilities.time_func
    def create_data_euler_3D_random(self) -> None:
        
        P_0 = 0.2
        P_1 = 0.8

        # *
        for i in range(self.__Number_of_images):

            # *
            #Data_3D = np.random.randint(0, 2, (self._Height * self._Depth * self._Width));
            Data_3D = np.random.choice(2, self.__Height * self.__Depth * self.__Width, p = [P_0, P_1]);
            Data_3D = Data_3D.reshape((self.__Height * self.__Depth), self.__Width);
            Data_3D_plot = Data_3D.reshape((self.__Height, self.__Depth, self.__Width));

            # *
            Data_3D_edges_concatenate = np.zeros((Data_3D_plot.shape[1] + 2, Data_3D_plot.shape[2] + 2))
            Data_3D_read = np.zeros((Data_3D.shape[0] + 2, Data_3D.shape[1] + 2))
            
            # *
            Data_3D_edges = np.zeros((Data_3D_plot.shape[0] + 2, Data_3D_plot.shape[1] + 2, Data_3D_plot.shape[2] + 2))
            
            # * Get 3D image and interpretation of 3D from 2D .txt
            Data_3D_read[1:Data_3D_read.shape[0] - 1, 1:Data_3D_read.shape[1] - 1] = Data_3D
            Data_3D_edges[1:Data_3D_edges.shape[0] - 1, 1:Data_3D_edges.shape[1] - 1, 1:Data_3D_edges.shape[2] - 1] = Data_3D_plot[i]

            #print(Data_3D_read);
            #print(Data_3D_edges);
            #print('\n');

            # * Concatenate np.zeros
            Data_3D_read = np.concatenate((Data_3D_edges_concatenate, Data_3D_read), axis = 0)
            Data_3D_read = np.concatenate((Data_3D_read, Data_3D_edges_concatenate), axis = 0)

            #print(Data_3D_read);

            for j in range(self.__Depth):
                
                # *
                Dir_name_images = "Images_{}_3D".format(j)

                # *
                Dir_data_images = self.__Folder + '/' + Dir_name_images

                # *
                Exist_dir_images = os.path.isdir(Dir_data_images)
                
                if Exist_dir_images == False:
                    Folder_path_images = os.path.join(self.__Folder, Dir_name_images)
                    os.mkdir(Folder_path_images)
                    print(Folder_path_images)
                else:
                    Folder_path_images = os.path.join(self.__Folder, Dir_name_images)
                    print(Folder_path_images)

                Image_name = "Image_slice_{}_{}_3D".format(i, j)
                Image_path = os.path.join(Folder_path_images, Image_name)
                plt.title('P_0: {}, P_1: {}'.format(P_0, P_1))
                plt.imshow(Data_3D_edges, cmap = 'gray', interpolation = 'nearest')
                plt.savefig(Image_path)

            File_name = 'Image_3D_{}.txt'.format(i);
            Path = os.path.join(self.__Folder, File_name);
            np.savetxt(Path, Data_3D_read, fmt = '%0.0f', delimiter = ',');

    # ?
    @Utilities.time_func
    def create_data_euler_3D_settings(self) -> None:
        
        # *
        global Input_3D_array
        global Output_3D_array

        # *
        Prediction = EulerNumberML3D(input = Input_3D_array, output = Output_3D_array, folder = self.__Folder);

        # *
        for i in range(self.__Number_of_images):

            #Data_2D = np.random.randint(0, 2, (self._Height * self._Width))

            Euler_number = 0

            P_0 = 0.2
            P_1 = 0.8

            while(Euler_number != self.__Euler_number):
                
                # *
                Data_3D = np.random.choice(2, self.__Height * self.__Depth * self.__Width, p = [P_0, P_1]);
                Data_3D = Data_3D.reshape((self.__Height * self.__Depth), self.__Width);
                Data_3D_plot = Data_3D.reshape((self.__Height, self.__Depth, self.__Width));

                # *
                Data_3D_edges = np.zeros((Data_3D_plot.shape[0] + 2, Data_3D_plot.shape[1] + 2, Data_3D_plot.shape[2] + 2))

                Data_3D_edges[1:Data_3D_edges.shape[0] - 1, 1:Data_3D_edges.shape[1] - 1, 1:Data_3D_edges.shape[2] - 1] = Data_3D_plot[i]

                print(Data_3D_plot);
                print('\n');

                Array = Prediction.obtain_arrays_3D(Data_3D_edges);
                Euler_number = Prediction.model_prediction_3D(self.__Model_trained, Array);

                Image_name = "Image_3D_Real_Time_{}.png".format(j)
                Image_path = os.path.join(self.__Folder, Image_name)
                plt.title('P_0: {}, P_1: {}, Euler_number: {}'.format(P_0, P_1, Euler_number))
                plt.imshow(Data_3D_edges, cmap = 'gray', interpolation = 'nearest')
                plt.savefig(Image_path)

                if(Euler_number > self.__Euler_number):

                    if(P_0 != 0.98):

                        P_0 = P_0 - 0.02;
                        P_1 = P_1 + 0.02;

                else:
                    
                    if(P_1 != 0.98):
                        
                        P_0 = P_0 + 0.02;
                        P_1 = P_1 - 0.02;


            if(self.__Save_image):

                Image_name = "Image_3D_{}.png".format(i)
                Image_path = os.path.join(self.__Folder, Image_name)

                plt.imshow(Data_3D_edges, cmap = 'gray', interpolation = 'nearest')
                plt.savefig(Image_path)
                #plt.show()

            File_name = 'Image_3D_{}.txt'.format(i);
            Path = os.path.join(self.__Folder, File_name);
            np.savetxt(Path, Data_3D , fmt = '%0.0f', delimiter = ',');