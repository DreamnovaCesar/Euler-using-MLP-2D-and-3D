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
            print('\n')

            print('These are the settings: Number of Objects: {}, Height: {}, Width: {}'.format(Objects_, Height_, Width_));
            print('\n')

            Proceed = input('Do you want to proceed?: [y/n]');
            print('\n')

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
            print('\n')

            print('These are the settings: Number of Objects: {}, Height: {}, Width: {}, Depth {}'.format(Objects_, Height_, Width_, Depth_));
            print('\n')

            Proceed = input('Do you want to proceed?: [y/n]');

            if(Proceed == 'y'):

                Cycle = False;

            else:

                Cycle = True;

        Images_3D = DataEuler(folder = Folder_3D_, NI = Objects_, Height = Height_, Width = Width_, Depth = Depth_);
        Images_3D.create_data_euler_3D_random();

    @staticmethod
    def Train_model_2D(Euler_path_2D_):

        Cycle = True

        while(Cycle):

            while(True):

                Connectivity_ = input('Which Connectivity will be used: connectivity [4] or [8]: ');
                Model_name_ = input('Name of the model trained: ');
                Epochs_ = input('How many epochs for the model: ');
                print('\n')

                if(Connectivity_ == '4' or Connectivity_ == '8'):
                    print('Connectivity is {}'.format(Connectivity_));
                    print('\n')
                    break;

            print('These are the settings: Connectivity: {}, Model name: {}, Epochs: {}'.format(Connectivity_, Model_name_, Epochs_));
            print('\n')

            Proceed = input('Do you want to proceed? [y/n]: ');
            print('\n')

            if(Proceed == 'y'):

                break;

            else:

                pass

        if(Connectivity_ == '4'):

            Euler_2D_MLP = EulerNumberML2D(input = Input_2D, output = Output_2D_4_Connectivity, folder = Euler_path_2D_, MN = Model_name_, epochs = Epochs_)
            Euler_2D_MLP.model_euler_MLP_2D()

        elif(Connectivity_ == '8'):

            Euler_2D_MLP = EulerNumberML2D(input = Input_2D, output = Output_2D_8_Connectivity, folder = Euler_path_2D_, MN = Model_name_, epochs = Epochs_)
            Euler_2D_MLP.model_euler_MLP_2D()

        else:
            pass

        #Arrays = Euler_2D_MLP.obtain_arrays_from_object_2D(Object_path_4)
        #Euler_2D_MLP.model_prediction_2D('Model_MLP_2D_4.h5', Array_MLP_4)
        #Euler_2D_MLP.connectivity_4_prediction_2D(Array_MLP_4)
    
    @staticmethod
    def Train_model_3D(Euler_path_3D_):
        
        while(True):

            while(True):

                Algorithm_ = input('Which algorithm will be used: Random Forest [RF] or Multi Layer Perceptron [MLP]: ');
                Model_name_ = input('Name of the model trained: ');
                Epochs_ = input('How many epochs for the model: ');
                print('\n')

                if(Algorithm_ == 'RF' or Algorithm_ == 'MLP'):
                    print('Algorithm used is: {}'.format(Algorithm_));
                    print('\n')
                    break;

            print('These are the settings: ML algorithm: {}, Model name: {}, Epochs: {}'.format(Algorithm_, Model_name_, Epochs_));
            print('\n')

            Proceed = input('Do you want to proceed? [y/n]: ');
            print('\n')

            if(Proceed == 'y'):

                break;

            else:

                pass

        if(Algorithm_ == 'RF'):

            Euler_train_3D = EulerNumberML3D(input = Input_3D_array, output = Output_3D_array, folder = Euler_path_3D_, MN = Model_name_, epochs = Epochs_)
            Euler_train_3D.model_euler_RF_3D()

        elif(Algorithm_ == 'MLP'):

            Euler_train_3D = EulerNumberML3D(input = Input_3D_array, output = Output_3D_array, folder = Euler_path_3D_, MN = Model_name_, epochs = Epochs_)
            Euler_train_3D.model_euler_MLP_3D()

        else:
            
            pass

        
        #Arrays = Euler_2D_MLP.obtain_arrays_from_object_2D(Object_path_4)
        #Euler_2D_MLP.model_prediction_2D('Model_MLP_2D_4.h5', Array_MLP_4)
        #Euler_2D_MLP.connectivity_4_prediction_2D(Array_MLP_4)

    @Utilities.time_func  
    def menu(self):

        Menu = True

        while(Menu):

            print('What do you want to do:')
            print('1: Create object 2D')
            print('2: Create object 3D')
            print('\n')

            print('3: Train model 2D')
            print('4: Train model 3D')
            print('\n')

            print('5: Prediction 2D')
            print('6: Prediction 3D')
            print('\n')

            print('c: Close window')
            print('\n')

            Options = input('Option: ')


            if(Options == '1'):

                self.create_objects_2D(self.__Euler_path_images_2D)

            elif(Options == '2'):

                self.create_objects_3D(self.__Euler_path_images_3D)

            elif(Options == '3'):

                self.Train_model_2D(self.__Euler_path_data_2D)

            elif(Options == '4'):

                self.Train_model_3D(self.__Euler_path_data_3D)

            elif(Options == 'c'):

                Menu = False
            
            #Create_objects()
            #Euler_2D_test()
            #Euler_3D_test()

        return -1