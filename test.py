import numpy as np
import matplotlib.pyplot as plt

import numpy as np
  
Data = np.genfromtxt(r"C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\Handcraft\3D\Example_3D_3.txt", delimiter=",")

Data = Data.reshape((4, 8, 8));

print(Data)

for i in range(1):

    Data_3D = np.random.randint(0, 2, (8 * 8 * 8));
    Data_3D_plot = Data_3D.reshape((8, 8, 8));

    Data_3D_plot = np.array(Data_3D_plot)

    #print(Data_3D_plot);
    print('\n');

filled = np.array([
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 1, 1], [1, 0, 0], [1, 0, 1]],
    [[1, 1, 0], [1, 1, 1], [0, 0, 0]]
])

print(filled);

print(Data_3D_plot.shape);
print(filled.shape);

ax = plt.figure().add_subplot(projection = '3d')
ax.voxels(Data, edgecolors = 'gray')
plt.show()
