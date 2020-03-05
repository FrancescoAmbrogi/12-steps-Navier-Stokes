### Created by Francesco Ambrogi upon Lorena Barba notes
### 5th March 2020
### This program solve the 2D Burgers equation using
### An explicit time advancement scheme (Euler)
### And a second order finite differences scheme - Diffusion (Spatial)
### And a first order backward fifferences - Convection (Spatial)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

### Variable declarations

nx = 41
ny = 41
nt = 120
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .009
nu = 0.01
dt = sigma * dx * dy / nu

x = np.linspace(0, 2, nx)  # Vector x
y = np.linspace(0, 2, ny)  # Vector y

u = np.ones((nx,ny))
v = np.ones((nx,ny))
un = np.ones((nx,ny))
vn = np.ones((nx,ny))

comb = np.ones((nx,ny))

### Assign initial conditions
u[int(.75 / dy):int(1.25 / dy + 1), int(.75 / dx):int(1.25 / dx + 1)]=2
v[int(.75 / dy):int(1.25 / dy + 1), int(.75 / dx):int(1.25 / dx + 1)]=2

### Let's plot the ICs
fig = plt.figure(figsize=(11,7), dpi = 100)
ax = fig.gca(projection='3d')
X,Y = np.meshgrid(x,y)
surf1 = ax.plot_surface(X, Y, u[:], cmap=cm.coolwarm, rstride=1,cstride=1)
surf2 = ax.plot_surface(X, Y, v[:], cmap=cm.coolwarm, rstride=1,cstride=1)
fig.colorbar(surf1, shrink=0.5, aspect=5)
ax.set_xlabel('X')
ax.set_ylabel('Y')


### Starting the calculation
for n in range(nt+1): ## Starting the integration in time
    row, col = u.shape
    row, col = v.shape
    for j in range(1,row-1):
        for i in range(1,col-1):
            un = u.copy()
            vn = v.copy()
            u[j,i] = un[j,i] - (dt/dx * un[j,i]* (un[j,i] - un[j,i-1])) - (dt/dy * vn[j,i] * (un[j,i] - un[j-1,i])) + \
            (nu * dt /dx**2 * (un[j,i+1] - 2*un[j,i] + un[j,i-1])) + (nu * dt /dy**2 * (un[j+1,i] - 2 * un[j,i] + un[j-1,i]))
            
            v[j,i] = vn[j,i] - (dt/dx * un[j,i]* (vn[j,i] - vn[j,i-1])) - (dt/dy * vn[j,i] * (vn[j,i] - vn[j-1,i])) + \
            (nu * dt /dx**2 * (vn[j,i+1] - 2*vn[j,i] + vn[j,i-1])) + (nu * dt /dy**2 * (vn[j+1,i] - 2 * vn[j,i] + vn[j-1,i]))
            
            ### Boundary conditions
            u[0, :] = 1
            u[-1,:] = 1
            u[:,0] = 1
            u[:,-1] = 1
        
            v[0, :] = 1
            v[-1,:] = 1
            v[:,0] = 1
            v[:,-1] = 1

fig = plt.figure(figsize=(11,7), dpi = 100)
ax = fig.gca(projection='3d')
X,Y = np.meshgrid(x,y)
surf1 = ax.plot_surface(X, Y, u[:], cmap=cm.coolwarm, rstride=1,cstride=1)
surf2 = ax.plot_surface(X, Y, v[:], cmap=cm.coolwarm, rstride=1,cstride=1)
ax.set_zlim([1, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
clb = fig.colorbar(surf1, shrink=0.5, aspect=5)
clb.set_label('Velocity', labelpad=-40, y=1.10, rotation=0)
plt.title('2D Burgers Equation', {'fontsize': 20})
plt.show()
