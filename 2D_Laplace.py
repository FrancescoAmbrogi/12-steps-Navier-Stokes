### This program aim at resolving the 2D Laplace equation
### Created by Francesco Ambrogi on March the 5th 2020

### IMPORTANT the calculation using array is correct
### There are some issues in the matrix calculation
### That need to be fixed.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D   # New library required for 3D plots


### Let's create a function
def plot2D(x,y,p):
    fig = plt.figure(figsize=(11,7), dpi=100)
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid(x,y)
    surf = ax.plot_surface(X,Y,p[:], rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0,antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.view_init(30,225)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
def laplace2D(p, y, dx, dy, l1norm_target): ### the l1norm target will define the tolerance
    l1norm = 1
    pn = np.empty_like(p)
    
    while l1norm > l1norm_target:
        # row, col = p.shape
        # for j in range(1,row-1):
        #     for i in range(1,col-1):
        #         pn = p.copy()
        #         p[j,i] = ((dy**2 * (pn[j,i+1] + pn[j,i-1]) + dx**2 * (pn[j+1,i] + pn[j-1,i])) / (2 * (dx**2 + dy**2)))
                
        #         ## Boundary conditions
        #         p[:, 0] = 0      # p = 0 @ x = 0
        #         p[:, -1] = y     # p = y @ x = 2
        #         p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
        #         p[-1,:] = p[-2, :] # dp/dy = 0 @ y = 1
                
        #         l1norm = (np.sum(np.abs(p[:]) - np.abs(pn[:])) / np.sum(np.abs(pn[:])))
        
        pn = p.copy()
        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2]) +
                         dx**2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1])) /
                        (2 * (dx**2 + dy**2)))
            
        p[:, 0] = 0  # p = 0 @ x = 0
        p[:, -1] = y  # p = y @ x = 2
        p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
        p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 1
        l1norm = (np.sum(np.abs(p[:]) - np.abs(pn[:])) /
                np.sum(np.abs(pn[:])))
                
    return p
            
### Variable declaration
nx = 31
ny = 31
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

# Initaial condition
p = np.zeros((nx,ny))

# plotting features
x = np.linspace(0, 2, nx)
y = np.linspace(0, 1, ny)

### Boundary conditions
p[:,0] = 0      # p = 0 @ x = 0
p[:,-1] = y     # p = y @ x = 2
p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
p[-1,:] = p[-2, :] # dp/dy = 0 @ y = 1      

plot2D(x, y, p)
p = laplace2D(p, y, dx, dy, 1e-4)
plot2D(x, y, p)
plt.show();
        
    
    
