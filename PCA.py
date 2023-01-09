import scipy as sp
from scipy import io as sio
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


data = sio.loadmat('iris_data.mat', struct_as_record=True)
x1 = data['x']
x2 = np.array([x1[0] , x1[1] , x1[2] , x1[3]] ) # (4, 150) shaped array

m = np.mean(x2.T , axis = 0)
centered = np.subtract(x2.T, m) # substract column wise

fig1 = plt.figure(1)
plt.plot(centered[0:99,2], centered[0:99,3], "r.", linewidth = 4)
plt.plot(centered[100:149,2], centered[100:149,3], "b.", linewidth = 4)

plt.grid()

#plt.show()

cov_mat = np.cov(centered[:, 2:4].T) #features are in rows?
w, v = np.linalg.eig(cov_mat)

projected = np.dot(centered[:, 2:4], v)

fig2 = plt.figure(2)
plt.plot(projected[0:99,0], projected[0:99,1], "r.", linewidth = 4)
plt.plot(projected[100:149,0], projected[100:149,1], "b.", linewidth = 4)
plt.grid()

plt.show()


# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(1)
# ax = plt.axes(projection='3d')

# ax.plot3D(x[0,0:50], x[1,0:50], x[2,0:50], "r.", linewidth = 4)
# ax.plot3D(x[0,50:100], x[1,50:100], x[2,50:100],  "b.", linewidth = 4)
# ax.plot3D(x[0,100:150], x[1,100:150], x[2,100:150], "y.", linewidth = 4)



print( projected )