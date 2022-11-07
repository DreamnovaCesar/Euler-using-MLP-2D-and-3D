import numpy as np

Data_3D_edges_complete = np.zeros((8 + 2, 8 + 2))
Data_3D_edges = np.ones((2 + 2, 8 + 2, 8 + 2))

#print(Data_3D_edges)

for k in range(len(Data_3D_edges)):
    Data_3D_edges_complete = np.concatenate((Data_3D_edges_complete, Data_3D_edges[k]), axis = 0)
    print('\n')

print(Data_3D_edges_complete)