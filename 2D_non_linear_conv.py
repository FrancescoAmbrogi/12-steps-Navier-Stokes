## 2D non Linear convection

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import time

### Variable declarations
nx = 81   # x dimension
ny = 81   # y dimension
nt = 50
c = 1  # This is now not needeed
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .2
dt = sigma * dx # In this case in both directions sigma di the same

x = np.linspace(0, 2, nx) # create the vector x
y = np.linspace(0, 2, ny) # create the vector y

u = np.ones((ny,nx)) ## Creates the 2D array for the solution
v = np.ones((ny,nx)) ## Now we also have the convection due to the v component

# This also set the boundary conditions
un = np.ones((ny,nx)) ## Creates the temporary array
vn = np.ones((ny,nx))

# Set the initial conditions for both u and v
u[int(.5/dy):int(1 / dy + 1), int(.5/dx):int(1 / dx + 1)]=2
v[int(.5/dy):int(1 / dy + 1), int(.5/dx):int(1 / dx + 1)]=2

## Let us plot the initial conditions
fig = plt.figure(figsize=(11,7), dpi=100)
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x,y)
ax.plot_surface(X,Y,u, cmap=cm.viridis, rstride = 2, cstride = 2)
ax.set_xlabel('X')
ax.set_ylabel('Y')

### Starting the calculation with arrays
t = time.time()
for n in range(nt+1): ## Starting the integration in time
    row, col = u.shape
    row, col = v.shape
    for j in range(1,row):
        for i in range(1,col):
            un = u.copy()
            vn = v.copy()
            u[j,i] = (un[j,i] - (un[j,i]*c*dt/dx * (un[j,i] - un[j,i-1])) - (vn[j,i]*c*dt/dy * (un[j,i] - un[j-1,i])))
            v[j,i] = (vn[j,i] - (un[j,i]*c*dt/dx * (vn[j,i] - vn[j,i-1])) - (vn[j,i]*c*dt/dy * (vn[j,i] - vn[j-1,i])))
            
            # Boundary conditions
            u[0, :] = 1
            u[-1,:] = 1
            u[:,0] = 1
            u[:,-1] = 1
        
            v[0, :] = 1
            v[-1,:] = 1
            v[:,0] = 1
            v[:,-1] = 1

fig = plt.figure(figsize=(11,7), dpi=100)
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x,y)
ax.plot_surface(X,Y,u, cmap=cm.viridis, rstride = 2, cstride = 2)
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()
print('The calculation with loops took: ' + str('{0:2f}'.format(time.time() -t)) + 's')