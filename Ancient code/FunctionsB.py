# ? Method to obtain 1D arrays from a 3D array
    @Utilities.time_func
    @profile
    def obtain_arrays_from_object_3D(self, Object: str) -> list[np.ndarray]:
        """
        Method to obtain 1D arrays from a 3D array

        Args:
            Object (str): description

        """

        # *
        Arrays = []
        Asterisks = 30

        l = 2

        # *
        Array_new = self.read_image_with_metadata_3D(Object)

        # * Creation of empty numpy arrays 3D
        Array_prediction_octov = np.zeros((l, l, l))
        Array_prediction = np.zeros((8))

        # *
        for i in range(Array_new.shape[0] - 1):
            for j in range(Array_new.shape[1] - 1):
                for k in range(Array_new.shape[2] - 1):

                    Array_new[i:l + i, j:l + j, k:l + k]

                    # *
                    Array_prediction[0] = Array_new[i + 1][j][k]
                    Array_prediction[1] = Array_new[i + 1][j][k + 1]

                    Array_prediction[2] = Array_new[i][j][k]
                    Array_prediction[3] = Array_new[i][j][k + 1]

                    Array_prediction[4] = Array_new[i + 1][j + 1][k]
                    Array_prediction[5] = Array_new[i + 1][j + 1][k + 1]

                    Array_prediction[6] = Array_new[i][j + 1][k]
                    Array_prediction[7] = Array_new[i][j + 1][k + 1]
                    print('\n')

                    # *
                    print("*" * Asterisks)
                    Array_prediction_list = Array_prediction.tolist()
                    Array_prediction_list_int = [int(i) for i in Array_prediction_list]

                    # *
                    print("Kernel array")
                    print(Array_new[i:l + i, j:l + j, k:l + k])
                    print('\n')
                    print("Prediction array")
                    print(Array_prediction)
                    print('\n')
                    Arrays.append(Array_prediction_list_int)
                    print("*" * Asterisks)
                    print('\n')

        # *
        for i in range(len(Arrays)):
            print('{} ---- {}'.format(i, Arrays[i]))
        print('\n')
        
        return Arrays

# ? Method to obtain 1D arrays from a 2D array
    @Utilities.time_func
    def obtain_arrays_from_object_2D(self, Object) -> list[np.ndarray]:
        """
        Method to obtain 1D arrays from a 2D array

        Args:
            Object (str): description

        """

        # *
        Arrays = []
        Asterisks = 30

        k = 2

        # *
        Array = self.read_image_with_metadata_2D(Object)

        # *
        Array_comparison = np.zeros((k, k), dtype = 'int')
        Array_prediction = np.zeros((4), dtype = 'int')

        for i in range(Array.shape[0] - 1):
            for j in range(Array.shape[1] - 1):

                #Array_comparison[0][0] = Array[i][j]
                #Array_comparison[1][0] = Array[i + 1][j]
                #Array_comparison[0][1] = Array[i][j + 1]
                #Array_comparison[1][1] = Array[i + 1][j + 1]

                Array[i:k + i, j:k + j]

                Array_prediction[0] = Array[i][j]
                Array_prediction[1] = Array[i][j + 1]
                Array_prediction[2] = Array[i + 1][j]
                Array_prediction[3] = Array[i + 1][j + 1]

                print('\n')
                #print("*" * Asterisks)

                # *
                print("*" * Asterisks)
                Array_prediction_list = Array_prediction.tolist()
                Array_prediction_list_int = [int(i) for i in Array_prediction_list]

                # *
                print("Kernel array")
                print(Array[i:k + i, j:k + j])
                print('\n')
                print("Prediction array")
                print(Array_prediction)
                print('\n')
                Arrays.append(Array_prediction_list_int)
                print("*" * Asterisks)
                print('\n')

        # *
        for i in range(len(Arrays)):
            print('{} ---- {}'.format(i, Arrays[i]))
        print('\n')
        
        return Arrays