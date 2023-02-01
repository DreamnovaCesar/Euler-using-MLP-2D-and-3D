import pyvista as pv
import numpy as np

Data = np.genfromtxt(r'Objects\3D\Images\Image_random_0_3D.txt', delimiter = ",")
        
Height = Data.shape[0]/Data.shape[1]
values = Data.reshape((int(Height), int(Data.shape[1]), int(Data.shape[1])));

#values = np.where(values==0, 1, 0)

#values = np.linspace(0, 4096, 4096).reshape((16, 16, 16))
print(values)
print(values.shape)

# Create the spatial reference
grid = pv.UniformGrid()

# Set the grid dimensions: shape + 1 because we want to inject our values on
#   the CELL data
grid.dimensions = np.array(values.shape)

# Edit the spatial reference
grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis

# Add the data values to the cell data
grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array!

# Now plot the grid!
grid.plot(show_edges=True)