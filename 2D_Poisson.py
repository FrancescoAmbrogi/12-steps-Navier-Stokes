### This is the 10th step and its aims at solving the
### 2D Poisson equation nabla^2(P) = f(x,y)
### Poisson equation is a fundamental step of solving the Navier-Stokes eqns
### It is often solve to be sure that the velocity at the next time step
### will be divergence free. Enforce mass conservation.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D   # New library required for 3D plots

# Simulation parameters
nx = 100
ny = 100
nt = 100  # number of time steps
xmin = 0
xmax = 2
ymin = 0
ymax = 1

dx = (xmax - xmin) / (nx - 1) # grid size x direction
dy = (ymax - ymin) / (ny - 1) # grid size y direction

# Initialization
p = np.zeros((ny,nx))  # The initial cond for pressure is 0 everywhere
pn = np.zeros((ny,nx))
b = np.zeros((ny,nx))  # Initialization of the rhs
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

# RHS
b[int(ny / 4), int(nx / 4)] = 100
b[int(3 * ny / 4), int(nx / 4)] = 100
b[int(3 * ny / 4), int(3 * nx / 4)] = -100
b[int(ny / 4), int(3 * nx / 4)] = -100

### Calculation in two different ways

# for it in range(nt):
#     pn = p.copy()
#     p[1:-1,1:-1] = (((dy**2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2])) +
#                           (dx**2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1])) - dx**2 * dy**2 * b[1:-1,1:-1]) /
#                         (2 * (dx**2 + dy**2)))
    
#     # Boundary conditions
#     p[0, :] = 0
#     p[ny-1, :] = 0
#     p[:, 0] = 0
#     p[:, nx-1] = 0
    
for n in range(nt):
    row, col = p.shape
    row, col = b.shape
    for j in range(1,row-1):
        for i in range(1,col-1):
            pn = p.copy()
            p[j,i] = (((dy**2 * (pn[j,i+1] + pn[j,i-1])) + 
                        (dx**2 * (pn[j+1,i] + pn[j-1,i])) - dx**2 * dy**2 * b[j,i]) /
                      (2 * (dx**2 + dy**2)))
    
            # Boundary conditions
            p[0, :] = 0
            p[-1,:] = 0
            p[:,0] = 0
            p[:,-1] = 0
    
def plot2D(x,y,p):
    fig = plt.figure(figsize=(11,7), dpi=100)
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid(x,y)
    surf = ax.plot_surface(X,Y,p[:], rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0,antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(30,225)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
plot2D(x,y,p);
plt.show();
 