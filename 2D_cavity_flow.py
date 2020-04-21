import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D   # New library required for 3D plots

nx = 41
ny = 41
nt = 500
nit = 50
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = np.linspace(0,2,nx)
y = np.linspace(0,2,nx)
X, Y = np.meshgrid(x, y)

rho = 1
nu = .1
dt = .001

u = np.zeros((ny,nx))
v = np.zeros((ny,nx))
p = np.zeros((ny,nx))
b = np.zeros((ny,nx))
nt = 100   # Number of iterations for the Poisson solver

# Here we call the function that solve the cavity flow
u,v,p = cavity(nt,u,dt,dx,dy,p,rho,nu)

fig = plt.figure(figsize(22,14),dpi=100)
# Let us plot the pressure field as a contour
plt.contourf(X,Y,p,alpha=0.5,cmap=cm.coolwarm)
plt.colorbar(surf, shrink=0.5, aspect=10)
# Let us plot the pressure field outlines
plt.contour(X,Y,p,cmap=cm.coolwarm)

# Here let us plot the velocity field
plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

