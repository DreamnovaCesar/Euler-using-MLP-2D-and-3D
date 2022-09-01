import numpy as np
from PIL import Image

from General_2D import read_image_with_metadata

from General_2D import Connectivity_4_first_array
from General_2D import Connectivity_4_second_array
from General_2D import Connectivity_4_third_array

from General_2D import Connectivity_8_first_array
from General_2D import Connectivity_8_second_array
from General_2D import Connectivity_8_third_array

#image = Image.open(r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\X_2D.png")
#Array = np.asarray(image)

Asterisks = 30

Connectivity_4 = 0 
Connectivity_8 = 0 

Array_MLP = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Example_2D_1.txt"
#Array_MLP = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Example_2D_2.txt"
#Array_MLP = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Example_2D_3.txt"
#Array_MLP = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Example_2D_4.txt"

#Array_MLP = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Example_2D_5.txt"

Array = read_image_with_metadata(Array_MLP)

Array_comparison = np.zeros((2, 2), dtype = 'int')
Array_prediction = np.zeros((1, 4), dtype = 'int')

for i in range(Array.shape[0] - 1):
    for j in range(Array.shape[1] - 1):

        Array_comparison[0][0] = Array[i][j]
        Array_comparison[1][0] = Array[i + 1][j]
        Array_comparison[0][1] = Array[i][j + 1]
        Array_comparison[1][1] = Array[i + 1][j + 1]

        Array_prediction[0][0] = Array[i][j]
        Array_prediction[0][2] = Array[i + 1][j]
        Array_prediction[0][1] = Array[i][j + 1]
        Array_prediction[0][3] = Array[i + 1][j + 1]

        print("Kernel array")
        print(Array_comparison)
        print('\n')

        if(np.all(Array_comparison == Connectivity_4_first_array)):
            Connectivity_4 += 1
            print('Connectivity 4: {}'.format(Connectivity_4))

        if(np.all(Array_comparison == Connectivity_4_second_array)):
            Connectivity_4 -= 1
            print('Connectivity 4: {}'.format(Connectivity_4))

        if(np.all(Array_comparison == Connectivity_4_third_array)):
            Connectivity_4 += 1
            print('Connectivity 4: {}'.format(Connectivity_4))

        if(np.all(Array_comparison == Connectivity_8_first_array)):
            Connectivity_8 += 1
            print('Connectivity 8: {}'.format(Connectivity_8))

        if(np.all(Array_comparison == Connectivity_8_second_array)):
            Connectivity_8 -= 1
            print('Connectivity 8: {}'.format(Connectivity_8))

        if(np.all(Array_comparison == Connectivity_8_third_array)):
            Connectivity_8 -= 1
            print('Connectivity 8: {}'.format(Connectivity_8))

        print('\n')

print('\n')
print('Connectivity 4: {}'.format(Connectivity_4))
print('Connectivity 8: {}'.format(Connectivity_8))
