import tkinter
import tkinter.messagebox
import customtkinter

from Article_Euler_Number_Create_Data import DataEuler
from Article_Euler_Number_2D_And_3D_ML import Utilities
from Article_Euler_Number_2D_And_3D_ML import EulerNumberML2D
from Article_Euler_Number_2D_And_3D_ML import EulerNumberML3D

from Article_Euler_Number_2D_General import Input_2D
from Article_Euler_Number_2D_General import Output_2D_4_Connectivity
from Article_Euler_Number_2D_General import Output_2D_8_Connectivity

from Article_Euler_Number_3D_General import Input_3D_array
from Article_Euler_Number_3D_General import Output_3D_array

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

#https://github.com/TomSchimansky/CustomTkinter

class Menu(Utilities):
    """

    """
    def __init__(self) -> None:
        """


        Args:
            Image_cropped (ndarray): Raw image cropped that is use.

        Returns:
            ndarray: The image after the safe rotation transformation.
        """

        # *
        self.__Euler_path_2D = r'Objects\2D';
        self.__Euler_path_images_2D = r'Objects\2D\Images';
        self.__Euler_path_images_settings_2D = r'Objects\2D\Images_with_euler'
        self.__Euler_path_data_2D = r'Objects\2D\Data';

        # *
        self.__Euler_path_3D = r'Objects\3D';
        self.__Euler_path_images_3D = r'Objects\3D\Images';
        self.__Euler_path_images_settings_3D = r'Objects\3D\Images_with_euler'
        self.__Euler_path_data_3D = r'Objects\3D\Data';

    @staticmethod
    def create_objects_2D(Folder_2D_):

        while(True):

            Objects_ = input('How many objects: ');
            Height_ = input('Height of the object: ');
            Width_ = input('Width of the object: ');
            print('\n')

            print('These are the settings: Number of Objects: {}, Height: {}, Width: {}'.format(Objects_, Height_, Width_));
            print('\n')

            Proceed = input('Do you want to proceed?: [y/n]: ');
            print('\n')

            if(Proceed == 'y'):

                break;

            else:

                pass;

        Images_2D = DataEuler(folder = Folder_2D_, NI = Objects_, Height = Height_, Width = Width_);
        Images_2D.create_data_euler_2D_random();

    @staticmethod
    def create_objects_3D(Folder_3D_):
        
        while(True):

            Objects_ = input('How many objects: ');
            Height_ = input('Height of the object: ');
            Width_ = input('Width of the object: ');
            Depth_ = input('Depth of the object: ');
            print('\n')

            print('These are the settings: Number of Objects: {}, Height: {}, Width: {}, Depth {}'.format(Objects_, Height_, Width_, Depth_));
            print('\n')

            Proceed = input('Do you want to proceed?: [y/n]: ');
            print('\n')
            
            if(Proceed == 'y'):

                break;

            else:

                pass;

        Images_3D = DataEuler(folder = Folder_3D_, NI = Objects_, Height = Height_, Width = Width_, Depth = Depth_);
        Images_3D.create_data_euler_3D_random();
    
    @staticmethod
    def create_objects_settings_2D(Folder_2D_):

        while(True):

            Objects_ = input('How many objects: ');
            Euler_Number_ = input('Which euler number do you want: ');
            Height_ = input('Height of the object: ');
            Width_ = input('Width of the object: ');
            print('\n')

            print('These are the settings: Number of Objects: {}, Height: {}, Width: {}'.format(Objects_, Height_, Width_));
            print('\n')

            Proceed = input('Do you want to proceed?: [y/n]: ');
            print('\n')

            if(Proceed == 'y'):

                break;

            else:

                pass;

        Images_2D = DataEuler(folder = Folder_2D_, NI = Objects_, Height = Height_, Width = Width_, EN = Euler_Number_);
        Images_2D.create_data_euler_2D_settings();

    @staticmethod
    def create_objects_settings_3D(Folder_3D_):

        while(True):

            Objects_ = input('How many objects: ');
            Euler_Number_ = input('Which euler number do you want: ');
            Height_ = input('Height of the object: ');
            Width_ = input('Width of the object: ');
            Depth_ = input('Depth of the object: ');
            print('\n')

            print('These are the settings: Number of Objects: {}, Height: {}, Width: {}, Depth {}'.format(Objects_, Height_, Width_, Depth_));
            print('\n')

            Proceed = input('Do you want to proceed?: [y/n]: ');

            if(Proceed == 'y'):

                break;

            else:

                pass;

        Images_3D = DataEuler(folder = Folder_3D_, NI = Objects_, Height = Height_, Width = Width_, Depth = Depth_, EN = Euler_Number_);
        Images_3D.create_data_euler_3D_settings();

    @staticmethod
    def Train_model_2D(Euler_path_2D_):

        while(True):

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

            Euler_MLP_2D = EulerNumberML2D(input = Input_2D, output = Output_2D_4_Connectivity, folder = Euler_path_2D_, MN = Model_name_, epochs = Epochs_)
            Euler_MLP_2D.model_euler_MLP_2D()

        elif(Connectivity_ == '8'):

            Euler_MLP_2D = EulerNumberML2D(input = Input_2D, output = Output_2D_8_Connectivity, folder = Euler_path_2D_, MN = Model_name_, epochs = Epochs_)
            Euler_MLP_2D.model_euler_MLP_2D()

        else:
            pass
    
    @staticmethod
    def Train_model_3D(Euler_path_3D_):
        
        while(True):

            while(True):

                Algorithm_ = input('Which algorithm will be used: Random Forest [RF] or Multi Layer Perceptron [MLP]: ');
                Model_name_ = input('Name of the model trained: ');
                
                if(Algorithm_ == 'RF' or Algorithm_ == 'MLP'):
                    print('Algorithm used is: {}'.format(Algorithm_));
                    print('\n')
                    break;

                if(Algorithm_ == 'MLP'):
                    Epochs_ = input('How many epochs for the model: ');

                    print('These are the settings: ML algorithm: {}, Model name: {}, Epochs: {}'.format(Algorithm_, Model_name_, Epochs_));
                    print('\n')
                elif(Algorithm_ == 'MLP'):
                    
                    print('These are the settings: ML algorithm: {}, Model name: {}'.format(Algorithm_, Model_name_));

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

    @staticmethod
    def Prediction_2D():

        while(True):

            Model_path_ = input('Model path: ');
            Images_path_ = input('Image path: ');
            print('\n')

            print('These are the paths: Model_path_: {}, Images_path_: {}'.format(Model_path_, Images_path_));
            print('\n')

            Proceed = input('Do you want to proceed? [y/n]: ');
            print('\n')

            if(Proceed == 'y'):

                break;

            else:

                pass

        Euler_MLP_2D = EulerNumberML2D()

        Array = Euler_MLP_2D.obtain_arrays_from_object_2D(Images_path_)
        Euler_MLP_2D.model_prediction_2D(Model_path_, Array)

    @staticmethod
    def Prediction_3D():

        while(True):

            Model_path_ = input('Model path: ');
            Images_path_ = input('Image path: ');
            print('\n')

            print('These are the paths: Model_path_: {}, Images_path_: {}'.format(Model_path_, Images_path_));
            print('\n')

            Proceed = input('Do you want to proceed? [y/n]: ');
            print('\n')

            if(Proceed == 'y'):

                break;

            else:

                pass

        Euler_MLP_3D = EulerNumberML3D()

        Array = Euler_MLP_3D.obtain_arrays_from_object_3D(Images_path_)
        Euler_MLP_3D.model_prediction_3D(Model_path_, Array)
    
    @staticmethod
    def Show_3D():

        while(True):

            Path_model_3D = input('Path 3D model: ');
            print('\n')

            print('This is the path: Path_model_3D: {}'.format(Path_model_3D));
            print('\n')

            Proceed = input('Do you want to proceed? [y/n]: ');
            print('\n')

            if(Proceed == 'y'):

                break;

            else:

                pass
        
        Euler_3D = EulerNumberML3D()
        Euler_3D.Show_array_3D(Path_model_3D)

    @Utilities.time_func  
    def menu(self):

        app = App()
        app.mainloop()


class App(customtkinter.CTk):

    WIDTH = 780
    HEIGHT = 520

    def __init__(self):
        super().__init__()

        # *
        self.__Euler_path_2D = r'Objects\2D';
        self.__Euler_path_images_2D = r'Objects\2D\Images';
        self.__Euler_path_images_settings_2D = r'Objects\2D\Images_with_euler'
        self.__Euler_path_data_2D = r'Objects\2D\Data';

        # *
        self.__Euler_path_3D = r'Objects\3D';
        self.__Euler_path_images_3D = r'Objects\3D\Images';
        self.__Euler_path_images_settings_3D = r'Objects\3D\Images_with_euler'
        self.__Euler_path_data_3D = r'Objects\3D\Data';
        
        self.title("Article Euler 2D and 3D")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed
        
        # ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)

        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # ============ frame_left ============

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)   # empty row with minsize as spacing
        #self.frame_left.grid_rowconfigure(5, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Options",
                                              text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_1.grid(row=1, column=0, pady=10, padx=10)

        self.button_1 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Create 2D images",
                                                command=self.button_change_menu_2D)
        self.button_1.grid(row=2, column=0, pady=10, padx=20)

        self.button_3 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Create 3D images",
                                                command=self.button_change_menu_3D)
        self.button_3.grid(row=3, column=0, pady=10, padx=20)

        self.button_4 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Training 2D model",
                                                command=self.button_)
        self.button_4.grid(row=4, column=0, pady=10, padx=20)

        self.button_5 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Training 3D model",
                                                command=self.button_)
        self.button_5.grid(row=5, column=0, pady=10, padx=20)

        self.button_6 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Prediction 2D",
                                                command=self.button_)
        self.button_6.grid(row=6, column=0, pady=10, padx=20)

        self.button_7 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Prediction 3D",
                                                command=self.button_)
        self.button_7.grid(row=7, column=0, pady=10, padx=20)

        self.button_6 = customtkinter.CTkButton(master=self.frame_left,
                                                text="BACK",
                                                command=self.button_back_menu)
        self.button_6.grid(row=8, column=0, pady=10, padx=20)

        self.label_mode = customtkinter.CTkLabel(master=self.frame_left, text="Appearance Mode:")
        self.label_mode.grid(row=9, column=0, pady=0, padx=20, sticky="w")

        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                        values=["Light", "Dark", "System"],
                                                        command=self.change_appearance_mode)
        self.optionmenu_1.grid(row=10, column=0, pady=10, padx=20, sticky="w")


    # set default values
        self.optionmenu_1.set("Dark")
        
    def button_change_menu_2D(self):

        # ============ frame_right ============

        # configure grid layout (3x7)
        
        self.frame_right.destroy()

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        self.frame_info = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info.grid(row=0, column=0, pady=20, padx=20, sticky="nsew")

        # ============ frame_info ============

        # configure grid layout (1x1)
        self.frame_info.rowconfigure(0, weight=2)
        self.frame_info.columnconfigure(1, weight=2)

        self.Labelframe = customtkinter.CTkFrame(self.frame_info)
        self.Labelframe.grid(row=0, column=0, pady=20, padx=20, sticky="nsew")

        self.Labelframe1 = customtkinter.CTkFrame(self.frame_info)
        self.Labelframe1.grid(row=0, column=1, pady=20, padx=20, sticky="nsew")

        self.my_entry = customtkinter.CTkEntry(self.Labelframe, width=150,
                                            placeholder_text="Number of objects")

        self.my_entry.grid(row=0, column=0, padx=10, pady=10)

        self.my_entry1 = customtkinter.CTkEntry(self.Labelframe, width=150,
                                            placeholder_text="Height")
        self.my_entry1.grid(row=1, column=0, padx=10, pady=10)

        self.my_entry2 = customtkinter.CTkEntry(self.Labelframe, width=150,
                                            placeholder_text="Width")
        self.my_entry2.grid(row=2, column=0, padx=10, pady=10)

        self.button = customtkinter.CTkButton(self.Labelframe,
                                            text="Go!",
                                            command=self.create_objects_2D)

        self.button.grid(row=4, column=0, pady=10, padx=20)

    def button_change_menu_3D(self):

        # ============ frame_right ============

        # configure grid layout (3x7)
        self.frame_right.destroy()

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        self.frame_info1 = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info1.grid(row=0, column=0, pady=20, padx=20, sticky="nsew")

        # ============ frame_info ============

        # configure grid layout (1x1)
        self.frame_info1.rowconfigure(0, weight=1)
        self.frame_info1.columnconfigure(1, weight=1)

        self.Labelframe = customtkinter.CTkFrame(self.frame_info1)
        self.Labelframe.grid(row=0, column=0, pady=20, padx=20, sticky="nsew")

        self.Labelframe1 = customtkinter.CTkFrame(self.frame_info1)
        self.Labelframe1.grid(row=0, column=1, pady=20, padx=20, sticky="nsew")

        self.my_entry = customtkinter.CTkEntry(self.Labelframe, width=150,
                                            placeholder_text="Number of objects")

        self.my_entry.grid(row=0, column=0, padx=10, pady=10)

        self.my_entry1 = customtkinter.CTkEntry(self.Labelframe, width=150,
                                            placeholder_text="Height")
        self.my_entry1.grid(row=1, column=0, padx=10, pady=10)

        self.my_entry2 = customtkinter.CTkEntry(self.Labelframe, width=150,
                                            placeholder_text="Width")
        self.my_entry2.grid(row=2, column=0, padx=10, pady=10)

        self.my_entry3 = customtkinter.CTkEntry(self.Labelframe, width=150,
                                            placeholder_text="Depth")
        self.my_entry3.grid(row=3, column=0, padx=10, pady=10)

        self.button = customtkinter.CTkButton(self.Labelframe,
                                            text="Go!",
                                            command=self.create_objects_3D)

        self.button.grid(row=4, column=0, pady=10, padx=20)

    def button_back_menu(self):

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

    def button_get_value(self):
        print(self.my_entry.get())
        print(self.my_entry1.get())
        print(self.my_entry2.get())

    def button_(self):
        print('No use')

    def create_objects_2D(self):
        Images_2D = DataEuler(folder = self.__Euler_path_images_2D, NI = self.my_entry.get(), 
                                Height = self.my_entry1.get(), Width = self.my_entry2.get());

        Images_2D.create_data_euler_2D_random();

    def create_objects_3D(self):
        Images_3D = DataEuler(folder = self.__Euler_path_images_3D, NI = self.my_entry.get(), 
                                Height = self.my_entry1.get(), Width = self.my_entry2.get(), Depth = self.my_entry3.get());

        Images_3D.create_data_euler_3D_random();

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_closing(self, event=0):
        self.destroy()

def main():
    """Main function
    """
    #Create_objects()

    config = Menu()
    config.menu()

if __name__ == "__main__":
    main()