import numpy as np

Array = np.loadtxt(r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Example_3D_1.txt", delimiter = ',')

print(Array.shape[0])
print(Array.shape[1])

Height = Array.shape[0]/Array.shape[1]

Array_new = Array.reshape(int(Height), int(Array.shape[1]), int(Array.shape[1]))

print(Array_new)

print(Array_new.shape[0])
print(Array_new.shape[1])
print(Array_new.shape[2])

Array_quad = np.zeros((2, 2, 2))

#print(Array_quad)
print('\n')
print('Empieza lo chido')
print('\n')

for i in range(Array_new.shape[0] - 1):
    for j in range(Array_new.shape[1] - 1):
        for k in range(Array_new.shape[2] - 1):

            Array_quad[0][0][0] = Array_new[i][j][k]
            Array_quad[0][1][0] = Array_new[i][j + 1][k]
            Array_quad[0][0][1] = Array_new[i][j][k + 1]
            Array_quad[0][1][1] = Array_new[i][j + 1][k + 1]
            Array_quad[1][0][0] = Array_new[i + 1][j][k] 
            Array_quad[1][1][0] = Array_new[i + 1][j + 1][k]
            Array_quad[1][0][1] = Array_new[i + 1][j][k + 1]
            Array_quad[1][1][1] = Array_new[i + 1][j + 1][k + 1]

            print(Array_quad)
            print('\n')
