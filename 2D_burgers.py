### Created by Francesco Ambrogi upon Lorena Barba notes
### 5th March 2020
### This program solve the 2D Burgers equation using
### An explicit time advancement scheme (Euler)
### And a second order finite differences scheme - Diffusion (Spatial)
### And a first order backward fifferences - Convection (Spatial)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

### Variable declarations

nx = 41
ny = 41
nt = 120
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .0009
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
ax.plot_surface(X, Y, u[:], cmap=cm.viridis, rstride=1,cstride=1)
ax.plot_surface(X, Y, v[:], cmap=cm.viridis, rstride=1,cstride=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
