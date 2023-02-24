import numpy as np
import pyvista as pv

Data = np.genfromtxt(r'Objects\3D\Images\Image_random_0_3D.txt', delimiter = ",")
        
Height = Data.shape[0]/Data.shape[1]
values = Data.reshape((int(Height), int(Data.shape[1]), int(Data.shape[1])));

Shape = values.shape[0] * values.shape[1] * values.shape[2]

print(values)
# Create a cube object
#cube = pv.Cube(center=(0.0, 0.0, 0.0), x_length=0.1, y_length=0.1, z_length=0.1)
#cube1 = pv.Cube(center=(0.1, 0.1, 0.1), x_length=0.1, y_length=0.1, z_length=0.1)
#cube2 = pv.Cube(center=(0.0, 0.1, 0.1), x_length=0.1, y_length=0.1, z_length=0.1)
"""
# Plot the cube using PyVista's plotter
p = pv.Plotter()

for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        for k in range(values.shape[2]):
            if(values[i][j][k] == 1):
                cube = pv.Cube(center=(i, j, k), x_length=1, y_length=1, z_length=1)
                p.add_mesh(cube, show_edges=True)
p.show()
"""
# Plot the cube using PyVista's plotter
p = pv.Plotter()

for i in range(16):
    for j in range(16):
        for k in range(16):

            cube = pv.Cube(center=(i, j, k), x_length=1, y_length=1, z_length=1)
            p.add_mesh(cube, show_edges=True)
p.show()