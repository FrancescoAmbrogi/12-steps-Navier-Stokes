## 2D Linear convection

from mpl_toolkits.mplot3d import Axes3D   # New library required for 3D plots
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import time

### Variable declarations
nx = 81   # x dimension
ny = 81   # y dimension
nt = 100
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .2
dt = sigma * dx # In this case in both directions sigma di the same

x = np.linspace(0, 2, nx) # create the vector x
y = np.linspace(0, 2, ny) # create the vector y

u = np.ones((ny,nx)) ## Creates the 2D array for the solution
un = np.ones((ny,nx)) ## Creates the temporary array

## Assign the initial conditions
u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2

## Plto the initial condition
fig = plt.figure(figsize=(11,7), dpi=100)
ax = fig.gca(projection='3d')
X,Y = np.meshgrid(x,y)
surf = ax.plot_surface(X,Y, u[:], cmap=cm.viridis)
plt.show()

### Calculation with loops
t = time.time()
un = np.ones((ny,nx))
u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2

for n in range(nt + 1):  #loop across number of time steps
    un = u.copy()
    row, col = u.shape
    for j in range(1,row):
        for i in range(1,col):
            u[j,i] = (un[j,i] - (c*dt/dx * (un[j,i] - un[j,i-1])) - (c*dt/dy * (un[j,i] - un[j-1,i])))
            # Boundary conditions
            u[0, :] = 1
            u[-1,:] = 1
            u[:,0] = 1
            u[:,-1] = 1

fig = plt.figure(figsize=(11,7), dpi=100)
ax = fig.gca(projection='3d')
surf2 = ax.plot_surface(X,Y,u[:], cmap=cm.viridis)
plt.show()

print('The calculation with loops took: ' + str('{0:2f}'.format(time.time() -t)) + 's')
