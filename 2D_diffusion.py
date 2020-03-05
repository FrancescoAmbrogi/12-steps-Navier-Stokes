### Created by Francesco Ambrogi upon Lorena Barba notes
### 5th March 2020
### This program solve the 2D diffusion equation using
### An explicit time advancement scheme (Euler)
### And a second order finite differences scheme (Spatial)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


### Variable declaration
nx = 101    # x grid number
ny = 101    # y grid number
nt = 20    # Number of time steps
nu = .05   # Viscosity
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .25
dt = sigma * dx * dy / nu  # Time step size

x = np.linspace(0, 2, nx)  # x vector
y = np.linspace(0, 2, ny)  # y vector

u = np.ones((nx,ny))
un = np.ones((nx,ny))

### Assign the initial condition
# Set an hat function I.C. : u(.5<= x <= 1 and .5<= y <= 1) = 2
u[int(.75 / dy):int(1.25 / dy + 1), int(.75 / dx):int(1.25 / dx + 1)]=2

# Let us plot the initial condition
fig = plt.figure(figsize=(11,7), dpi=100)
ax = fig.gca(projection='3d')
X,Y = np.meshgrid(x,y)
surf = ax.plot_surface(X,Y,u,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0,antialiased=False)

ax.set_xlim([0, 2])
ax.set_ylim([0, 2])
ax.set_zlim([1, 2.5])

### Beginning of the calculation with a function
u[int(.75 / dy):int(1.25 / dy + 1), int(.75 / dx):int(1.25 / dx + 1)]=2

for n in range(nt+1):
    row, col = u.shape
    for j in range(1,row-1):
        for i in range(1,col-1):
            un = u.copy()
            u[j,i] = un[j,i] + (nu*dt/dx**2)*(un[j,i+1] -2*un[j,i] + un[j,i-1]) + (nu*dt/dy**2)*(un[j+1,i] -2*un[j,i] + un[j-1,i])

            ### Boundary conditions
            u[0, :] = 1
            u[-1,:] = 1
            u[:,0] = 1
            u[:,-1] = 1

fig = plt.figure(figsize=(11,7), dpi=100)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, u[:], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim([1, 2.5])
ax.set_xlabel('X')
ax.set_ylabel('Y');
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()