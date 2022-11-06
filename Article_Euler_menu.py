from Article_Euler_Number_Create_Data import DataEuler
from Article_Euler_Number_2D_And_3D_ML import Utilities
from Article_Euler_Number_2D_And_3D_ML import EulerNumberML2D
from Article_Euler_Number_2D_And_3D_ML import EulerNumberML3D

from Article_Euler_Number_2D_General import Input_2D
from Article_Euler_Number_2D_General import Output_2D_4_Connectivity
from Article_Euler_Number_2D_General import Output_2D_8_Connectivity

from Article_Euler_Number_3D_General import Input_3D_array
from Article_Euler_Number_3D_General import Output_3D_array

class Menu():

    def __init__(self, **kwargs) -> None:
        """
        _summary_

        _extended_summary_
        """

        # *
        self.__Euler_path_2D = r'Objects\2D';
        self.__Euler_path_images_2D = r'Objects\2D\Images';
        self.__Euler_path_data_2D = r'Objects\2D\Data';

        # *
        self.__Euler_path_3D = r'Objects\3D';
        self.__Euler_path_images_3D = r'Objects\3D\Images';
        self.__Euler_path_data_3D = r'Objects\3D\Data';

        Menu = True

    def __repr__(self):

        kwargs_info = '';

        return kwargs_info

    def __str__(self):
        pass
    
    @staticmethod
    def create_objects_2D(Folder_2D_):

        Cycle = True

        while(Cycle):

            Objects_ = input('How many objects: ');
            Height_ = input('Height of the object: ');
            Width_ = input('Width of the object: ');

            print('These are the settings: Number of Objects: {}, Height: {}, Width: {}'.format(Objects_, Height_, Width_));
            Proceed = input('Do you want to proceed?: [y/n]');

            if(Proceed == 'y'):

                Cycle = False;

            else:

                Cycle = True;

        Images_2D = DataEuler(folder = Folder_2D_, NI = Objects_, Height = Height_, Width = Width_);
        Images_2D.create_data_euler_2D_random();

    @staticmethod
    def create_objects_3D(Folder_3D_):

        Cycle = True

        while(Cycle):

            Objects_ = input('How many objects: ');
            Height_ = input('Height of the object: ');
            Width_ = input('Width of the object: ');
            Depth_ = input('Depth of the object: ');

            print('These are the settings: Number of Objects: {}, Height: {}, Width: {}, Depth {}'.format(Objects_, Height_, Width_, Depth_));
            Proceed = input('Do you want to proceed?: [y/n]');

            if(Proceed == 'y'):

                Cycle = False;

            else:

                Cycle = True;

        Images_3D = DataEuler(folder = Folder_3D_, NI = Objects_, Height = Height_, Width = Width_, Depth = Depth_);
        Images_3D.create_data_euler_3D_random();

    @Utilities.time_func  
    def menu(self):

        Menu = True

        while(Menu):

            print('What do you want to do:')
            print('1: Create object 2D')
            print('2: Create object 3D')

            print('3: Train model 2D')
            print('4: Train model 3D')

            print('5: Prediction 2D')
            print('6: Prediction 3D')

            print('c: Close window')
            
            Options = input('Option: ')


            if(Options == '1'):

                self.create_objects_2D(self.__Euler_path_images_2D)

            elif(Options == '2'):

                self.create_objects_3D(self.__Euler_path_images_3D)

            elif(Options == '3'):

                Train_model = input('Which 4-Connectivity')
                Train_model = input('Train 4-Connectivity')

                #Euler_2D_MLP = EulerNumberML2D(input = Input_2D, output = Output_2D_4_Connectivity, folder = Euler_path_2D, MN = 'Model_MLP_2D_4', epochs = 1000)

            elif(Options == '4'):

                Train_model = input('Train 4-Connectivity')

                #Images_3D = DataEuler(folder = Folder_3D, NI = Objects, Height = 8, Width = 8, Depth = 8);
                #Images_3D.create_data_euler_3D_random();

            elif(Options == 'c'):

                Menu = False
            
            #Create_objects()
            #Euler_2D_test()
            #Euler_3D_test()

        return -1