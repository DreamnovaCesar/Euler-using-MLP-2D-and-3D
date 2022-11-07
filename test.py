import numpy as np

#Data_3D = np.random.randint(0, 2, (self._Height * self._Depth * self._Width));
Data_3D = np.random.choice(2, 8 * 8 * 8, p = [0.2, 0.8]);
Data_3D = Data_3D.reshape((8 * 8), 8);
Data_3D_plot = Data_3D.reshape((8, 8, 8));
#Data_3D_plot = Data_3D.reshape((self.__Height, self.__Depth, self.__Width));

#print(Data_3D_plot[0])

# *
Data_3D_edges_init = np.zeros((Data_3D_plot.shape[1] + 2, Data_3D_plot.shape[2] + 2))
Data_3D_edges = np.zeros((Data_3D.shape[0] + 2, Data_3D.shape[1] + 2))

#print(Data_3D_edges_init)
#print(Data_3D_edges)


Data_3D_edges[1:Data_3D_edges.shape[0] - 1, 1:Data_3D_edges.shape[1] - 1] = Data_3D

# *
#print(Data_3D_edges)

gfg = np.concatenate((Data_3D_edges_init, Data_3D_edges), axis = 0)
gfg = np.concatenate((gfg, Data_3D_edges_init), axis = 0)

print(gfg)